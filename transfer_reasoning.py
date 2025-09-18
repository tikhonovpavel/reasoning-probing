import logging
import sqlite3
import json
import re
from typing import Optional, List, Tuple

import fire
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Import the necessary functions from rc_utils
from rc_utils import extract_think_content, split_reasoning_chain

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# This prompt is used to force the model to stop thinking and generate a final answer
OSS_SPECIAL_STOPPING_PROMPT = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.<|end|><|start|>assistant<|channel|>final<|message|>"


def extract_final_answer_letter(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Extracts the raw final answer and the letter from the full model output."""
    # The final answer is between <|channel|>final<|message|> and <|return|> or <|end|>
    match = re.search(r"<\|channel\|>final<\|message\|>(.*?)(\<\|return\|\>|\<\|end\|\>)", text, re.DOTALL)
    if not match:
        # If no final block, maybe the model just outputted a letter directly after the prompt
        letter_match_direct = re.search(r"^\s*([A-D])", text)
        if letter_match_direct:
            return text.strip(), letter_match_direct.group(1)
        return text.strip(), None

    raw_final_answer = match.group(1).strip()
    
    # Find the first single letter A, B, C, or D in the final answer block
    letter_match = re.search(r"\b([A-D])\b", raw_final_answer)
    if letter_match:
        return raw_final_answer, letter_match.group(1)
        
    return raw_final_answer, None


def load_transfer_model(model_path: str):
    """Loads the target model and tokenizer for the transfer analysis."""
    logger.info(f"Loading transfer model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:1",
    )
    logger.info("Transfer model loaded successfully.")
    return model, tokenizer


def create_db_table(conn):
    """Creates the results table if it doesn't exist."""
    try:
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reasoning_transfer_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id INTEGER NOT NULL,
                    transfer_model_path TEXT NOT NULL,
                    original_prompt TEXT,
                    qwen_reasoning_at_k TEXT,
                    qwen_reasoning_at_k_minus_1 TEXT,
                    qwen_reasoning_at_k_minus_2 TEXT,
                    oss_continuation_text TEXT,
                    oss_continuation_text_from_k_minus_2 TEXT,
                    oss_prediction_at_k_minus_2_raw TEXT,
                    oss_prediction_at_k_minus_2_letter TEXT,
                    oss_prediction_at_k_minus_1_raw TEXT,
                    oss_prediction_at_k_minus_1_letter TEXT,
                    oss_prediction_at_k_raw TEXT,
                    oss_prediction_at_k_letter TEXT,
                    oss_prediction_after_continuation_raw TEXT,
                    oss_prediction_after_continuation_letter TEXT,
                    oss_prediction_after_continuation_from_k_minus_2_raw TEXT,
                    oss_prediction_after_continuation_from_k_minus_2_letter TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    UNIQUE(trace_id, transfer_model_path)
                )
            """)
        logger.info("Database table 'reasoning_transfer_results' is ready.")
    except sqlite3.Error as e:
        logger.error(f"Database error while creating table: {e}")
        raise


def get_processed_trace_ids(conn, transfer_model_path: str) -> set:
    """Gets the set of trace_ids that have already been processed for this model."""
    try:
        with conn:
            rows = conn.execute(
                "SELECT trace_id FROM reasoning_transfer_results WHERE transfer_model_path = ?",
                (transfer_model_path,)
            ).fetchall()
        processed_ids = {row[0] for row in rows}
        logger.info(f"Found {len(processed_ids)} already processed trace IDs for {transfer_model_path}.")
        return processed_ids
    except sqlite3.Error as e:
        logger.error(f"Database error while fetching processed IDs: {e}")
        return set()


def parse_json_safe(raw: str, default_val=None):
    if not raw:
        return default_val
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return default_val


def find_first_change_index(predictions: List[Optional[str]]) -> Optional[int]:
    last_pred = None
    for i, pred in enumerate(predictions):
        if pred and pred in "ABCD":
            if last_pred is None:
                last_pred = pred
            elif pred != last_pred:
                return i
    return None


def fetch_transfer_data(sqlite_db: str, model_name: str, where_model_path: str, system_prompt: str, head_limit: Optional[int]) -> pd.DataFrame:
    """Fetches traces with exactly one answer change from the database."""
    logger.info("Fetching data for transfer analysis...")
    conn = sqlite3.connect(sqlite_db)

    sql = """
    SELECT
        m.trace_id,
        t.question_text,
        t.choices,
        m.predictions_json,
        m.continuation_texts_json,
        t.full_prompt_text
    FROM
        reasoning_trace_forced_solution_metrics AS m
    JOIN
        reasoning_traces_qpqa AS t ON m.trace_id = t.id
    WHERE
        m.num_changes = 1
        AND m.model_name = ?
        AND m.model_path = ?
        AND m.system_prompt = ?
    """
    params = [model_name, where_model_path, system_prompt]

    if head_limit:
        sql += f" LIMIT {int(head_limit)}"

    try:
        df = pd.read_sql_query(sql, conn, params=params)
        logger.info(f"Fetched {len(df)} rows.")
        return df
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        return pd.DataFrame()
    finally:
        conn.close()


def run(
    sqlite_db: str,
    model_name: str = "Qwen/Qwen3-32B",
    where_model_path: str = "Qwen/Qwen3-32B",
    system_prompt: str = "Answer only with a letter of a correct choice.",
    transfer_model_path: str = "openai/gpt-oss-20b",
    head_limit: Optional[int] = 50,
    out_file: str = "transfer_reasoning_results.jsonl",
    test_prompt_construction: bool = False,
    # Scenario flags
    run_scenario_at_k: bool = True,
    run_scenario_at_k_minus_1: bool = True,
    run_scenario_at_k_minus_2: bool = True,
    run_scenario_after_continuation: bool = True,
    run_scenario_after_continuation_from_k_minus_2: bool = True,
    continuation_tokens: int = 128,
    max_token_limit: int = 8192,
):
    """
    Analyzes the transferability of reasoning chains across three scenarios.
    """
    tokenizer = AutoTokenizer.from_pretrained(transfer_model_path)
    
    # --- TEST MODE ---
    if test_prompt_construction:
        logger.info("--- Running in Prompt Construction Test Mode ---")
        df = fetch_transfer_data(sqlite_db, model_name, where_model_path, system_prompt, head_limit=1)
        if df.empty:
            return
        
        row = df.iloc[0]
        # Common data preparation
        reasoning_chain = extract_think_content(row['full_prompt_text'])
        reasoning_chunks = split_reasoning_chain(reasoning_chain or "")
        k = find_first_change_index(parse_json_safe(row['predictions_json'], []))
        
        choices = parse_json_safe(row['choices'], [])
        choices_formatted = "\n".join([f"{chr(65+i)}) {choice}" for i, choice in enumerate(choices)])
        user_content = f"{row['question_text']}\n\n{choices_formatted}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        prompt_history = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)

        print("\n" + "="*80)
        print("                PROMPT CONSTRUCTION TEST")
        print("="*80)
        
        # Scenario k-2
        if run_scenario_at_k_minus_2 and k is not None and k > 1:
            reasoning_part = "\n\n".join(reasoning_chunks[:k-1])
            prompt = prompt_history + f"<|start|>assistant<|channel|>analysis<|message|>{reasoning_part}" + OSS_SPECIAL_STOPPING_PROMPT
            print("\n--- SCENARIO: At Pre-Pre-Critical Chunk (k-2) ---\n", prompt)

        # Scenario k-1
        if run_scenario_at_k_minus_1 and k is not None and k > 0:
            reasoning_part = "\n\n".join(reasoning_chunks[:k])
            prompt = prompt_history + f"<|start|>assistant<|channel|>analysis<|message|>{reasoning_part}" + OSS_SPECIAL_STOPPING_PROMPT
            print("\n--- SCENARIO: At Pre-Critical Chunk (k-1) ---\n", prompt)

        # Scenario A
        if run_scenario_at_k and k is not None:
            reasoning_part = "\n\n".join(reasoning_chunks[:k+1])
            prompt = prompt_history + f"<|start|>assistant<|channel|>analysis<|message|>{reasoning_part}" + OSS_SPECIAL_STOPPING_PROMPT
            print("\n--- SCENARIO: At Critical Chunk (k) ---\n", prompt)

        # Scenario B (from k-1)
        if run_scenario_after_continuation and k is not None and k > 0:
            reasoning_part = "\n\n".join(reasoning_chunks[:k])
            prompt = prompt_history + f"<|start|>assistant<|channel|>analysis<|message|>{reasoning_part}"
            print(f"\n--- SCENARIO: After Continuation from (k-1) --- (will generate {continuation_tokens} tokens then add stopping prompt)\n", prompt)
            
        # Scenario B2 (from k-2)
        if run_scenario_after_continuation_from_k_minus_2 and k is not None and k > 1:
            reasoning_part = "\n\n".join(reasoning_chunks[:k-1])
            prompt = prompt_history + f"<|start|>assistant<|channel|>analysis<|message|>{reasoning_part}"
            print(f"\n--- SCENARIO: After Continuation from (k-2) --- (will generate {continuation_tokens} tokens then add stopping prompt)\n", prompt)

        print("="*80)
        return

    # --- FULL RUN MODE ---
    logger.info(f"Starting reasoning transfer analysis from '{model_name}' to '{transfer_model_path}'")
    
    conn = sqlite3.connect(sqlite_db)
    try:
        create_db_table(conn)
        processed_ids = get_processed_trace_ids(conn, transfer_model_path)

        model, tokenizer = load_transfer_model(transfer_model_path)
        df = fetch_transfer_data(sqlite_db, model_name, where_model_path, system_prompt, head_limit)
        
        # Filter out already processed traces
        if not df.empty:
            original_count = len(df)
            df = df[~df['trace_id'].isin(processed_ids)]
            logger.info(f"Skipping {original_count - len(df)} processed traces. Starting with {len(df)} new traces.")

        if df.empty:
            logger.info("No new traces to process.")
            return

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing traces"):
            # Prepare common data for the trace
            reasoning_chain = extract_think_content(row['full_prompt_text'])
            if not reasoning_chain: continue
            reasoning_chunks = split_reasoning_chain(reasoning_chain)
            continuations = parse_json_safe(row['continuation_texts_json'], [])
            if len(reasoning_chunks) != len(continuations): continue

            k = find_first_change_index(parse_json_safe(row['predictions_json'], []))
            if k is None: continue

            choices = parse_json_safe(row['choices'], [])
            choices_formatted = "\n".join([f"{chr(65+i)}) {choice}" for i, choice in enumerate(choices)])
            user_content = f"{row['question_text']}\n\n{choices_formatted}"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
            prompt_history = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
            
            # Check token length of the longest potential prompt (Scenario A) before running any scenarios
            longest_reasoning_part = "\n\n".join(reasoning_chunks[:k+1])
            longest_prompt = prompt_history + f"<|start|>assistant<|channel|>analysis<|message|>{longest_reasoning_part}" + OSS_SPECIAL_STOPPING_PROMPT
            token_count = len(tokenizer(longest_prompt)['input_ids'])

            if token_count > max_token_limit:
                logger.warning(
                    f"Skipping trace_id {row['trace_id']} because its longest prompt ({token_count} tokens) "
                    f"exceeds the limit of {max_token_limit}."
                )
                continue

            result_item = {
                "trace_id": int(row['trace_id']), 
                "transfer_model_path": transfer_model_path,
                "original_prompt": user_content
            }

            # --- Run Scenario C -> k-1 ---
            if run_scenario_at_k_minus_1 and k > 0:
                reasoning_part = "\n\n".join(reasoning_chunks[:k])
                result_item['qwen_reasoning_at_k_minus_1'] = reasoning_part
                prompt = prompt_history + f"<|start|>assistant<|channel|>analysis<|message|>{reasoning_part}" + OSS_SPECIAL_STOPPING_PROMPT
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                logger.info(f"Trace {row['trace_id']}, Scenario k-1: {len(inputs['input_ids'][0])} tokens")
                outputs = model.generate(**inputs, max_new_tokens=32, pad_token_id=tokenizer.eos_token_id)
                generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=False)
                raw_answer, letter = extract_final_answer_letter(generated_text)
                result_item['oss_prediction_at_k_minus_1_raw'] = raw_answer
                result_item['oss_prediction_at_k_minus_1_letter'] = letter

            # --- Run Scenario k-2 ---
            if run_scenario_at_k_minus_2 and k > 1:
                reasoning_part = "\n\n".join(reasoning_chunks[:k-1])
                result_item['qwen_reasoning_at_k_minus_2'] = reasoning_part
                prompt = prompt_history + f"<|start|>assistant<|channel|>analysis<|message|>{reasoning_part}" + OSS_SPECIAL_STOPPING_PROMPT
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                logger.info(f"Trace {row['trace_id']}, Scenario k-2: {len(inputs['input_ids'][0])} tokens")
                outputs = model.generate(**inputs, max_new_tokens=32, pad_token_id=tokenizer.eos_token_id)
                generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=False)
                raw_answer, letter = extract_final_answer_letter(generated_text)
                result_item['oss_prediction_at_k_minus_2_raw'] = raw_answer
                result_item['oss_prediction_at_k_minus_2_letter'] = letter

            # --- Run Scenario A -> k ---
            if run_scenario_at_k:
                reasoning_part = "\n\n".join(reasoning_chunks[:k+1])
                result_item['qwen_reasoning_at_k'] = reasoning_part
                prompt = prompt_history + f"<|start|>assistant<|channel|>analysis<|message|>{reasoning_part}" + OSS_SPECIAL_STOPPING_PROMPT
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                logger.info(f"Trace {row['trace_id']}, Scenario A: {len(inputs['input_ids'][0])} tokens")
                outputs = model.generate(**inputs, max_new_tokens=32, pad_token_id=tokenizer.eos_token_id)
                generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=False)
                raw_answer, letter = extract_final_answer_letter(generated_text)
                result_item['oss_prediction_at_k_raw'] = raw_answer
                result_item['oss_prediction_at_k_letter'] = letter

            # --- Run Scenario B -> after k-1 ---
            if run_scenario_after_continuation and k > 0:
                reasoning_part = "\n\n".join(reasoning_chunks[:k])
                # Note: reasoning for this is already saved under qwen_reasoning_at_k_minus_1 if that scenario ran
                if 'qwen_reasoning_at_k_minus_1' not in result_item:
                    result_item['qwen_reasoning_at_k_minus_1'] = reasoning_part
                
                prompt_start = prompt_history + f"<|start|>assistant<|channel|>analysis<|message|>{reasoning_part}"
                inputs = tokenizer(prompt_start, return_tensors="pt").to(model.device)
                logger.info(f"Trace {row['trace_id']}, Scenario B (cont): {len(inputs['input_ids'][0])} tokens")
                # Generate a short continuation
                continuation_outputs = model.generate(**inputs, max_new_tokens=continuation_tokens, pad_token_id=tokenizer.eos_token_id)
                generated_continuation = tokenizer.decode(continuation_outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
                result_item['oss_continuation_text'] = generated_continuation
                # Now, force the answer
                prompt_with_continuation = prompt_start + generated_continuation + OSS_SPECIAL_STOPPING_PROMPT
                inputs_final = tokenizer(prompt_with_continuation, return_tensors="pt").to(model.device)
                logger.info(f"Trace {row['trace_id']}, Scenario B (final): {len(inputs_final['input_ids'][0])} tokens")
                outputs_final = model.generate(**inputs_final, max_new_tokens=32, pad_token_id=tokenizer.eos_token_id)
                generated_text_final = tokenizer.decode(outputs_final[0][inputs_final["input_ids"].shape[-1]:], skip_special_tokens=False)
                raw_answer, letter = extract_final_answer_letter(generated_text_final)
                result_item['oss_prediction_after_continuation_raw'] = raw_answer
                result_item['oss_prediction_after_continuation_letter'] = letter

            # --- Run Scenario B2: After continuation from k-2 ---
            if run_scenario_after_continuation_from_k_minus_2 and k > 1:
                reasoning_part = "\n\n".join(reasoning_chunks[:k-1])
                if 'qwen_reasoning_at_k_minus_2' not in result_item:
                    result_item['qwen_reasoning_at_k_minus_2'] = reasoning_part
                
                prompt_start = prompt_history + f"<|start|>assistant<|channel|>analysis<|message|>{reasoning_part}"
                inputs = tokenizer(prompt_start, return_tensors="pt").to(model.device)
                logger.info(f"Trace {row['trace_id']}, Scenario B2 (cont): {len(inputs['input_ids'][0])} tokens")

                continuation_outputs = model.generate(**inputs, max_new_tokens=continuation_tokens, pad_token_id=tokenizer.eos_token_id)
                generated_continuation = tokenizer.decode(continuation_outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
                result_item['oss_continuation_text_from_k_minus_2'] = generated_continuation
                
                prompt_with_continuation = prompt_start + generated_continuation + OSS_SPECIAL_STOPPING_PROMPT
                inputs_final = tokenizer(prompt_with_continuation, return_tensors="pt").to(model.device)
                logger.info(f"Trace {row['trace_id']}, Scenario B2 (final): {len(inputs_final['input_ids'][0])} tokens")
                
                outputs_final = model.generate(**inputs_final, max_new_tokens=32, pad_token_id=tokenizer.eos_token_id)
                generated_text_final = tokenizer.decode(outputs_final[0][inputs_final["input_ids"].shape[-1]:], skip_special_tokens=False)
                raw_answer, letter = extract_final_answer_letter(generated_text_final)
                result_item['oss_prediction_after_continuation_from_k_minus_2_raw'] = raw_answer
                result_item['oss_prediction_after_continuation_from_k_minus_2_letter'] = letter

            # Insert results into the database
            columns = ', '.join(result_item.keys())
            placeholders = ', '.join('?' * len(result_item))
            insert_sql = f'INSERT INTO reasoning_transfer_results ({columns}) VALUES ({placeholders})'
            try:
                with conn:
                    conn.execute(insert_sql, list(result_item.values()))
            except sqlite3.Error as e:
                logger.error(f"Failed to insert row for trace_id {row['trace_id']}: {e}")


    finally:
        conn.close()
        logger.info(f"Analysis complete. Database connection closed.")


if __name__ == "__main__":
    fire.Fire(run)
