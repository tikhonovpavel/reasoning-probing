import sqlite3
import os
import json
import fire
import re
from tqdm import tqdm
from transformers import AutoTokenizer
from gpqa_dataset import GPQADataset
import time
import requests
import concurrent.futures
from logiqa_dataset import LogiqaDataset
import traceback

def setup_database(db_path: str, table_name: str):
    """Initializes the database and creates the specified table if it doesn't exist."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_path TEXT NOT NULL,
            question_id INTEGER NOT NULL,
            question_text TEXT NOT NULL,
            choices TEXT NOT NULL,
            correct_answer_letter TEXT NOT NULL,
            full_prompt_text TEXT NOT NULL,
            full_prompt_token_ids TEXT NOT NULL,
            extracted_answer TEXT,
            is_correct INTEGER
        )
    """)
    conn.commit()
    return conn

def process_item(args):
    """
    Processes a single item: sends a request to OpenRouter, retries on failure,
    parses the response, and returns a dictionary for database insertion.
    """
    question_id, item, model_name, api_key, tokenizer, with_reasoning, include_no_think_instruction = args
    max_retries = 3
    retries = 0
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    messages = [
        {"role": "system", "content": "Respond only with the letter of the correct option. Don't write any explanations"},
        {"role": "user", "content": item['prompt']},
    ]
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.6,
        "reasoning": {"enabled": True},
    }
    if not with_reasoning:
        payload["reasoning"] = {"max_tokens": 1}

    response = None
    while retries < max_retries:
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=180)
            response.raise_for_status()
            
            response_data = response.json()
            choice = response_data['choices'][0]
            final_answer = choice['message'].get('content', '').strip()
            
            # if with_reasoning:
            reasoning = choice['message'].get('reasoning', '')
            if not reasoning or len(reasoning.strip()) < 50:
                # Reasoning is too short or missing, retry
                retries += 1
                time.sleep(2) # Wait a bit before retrying
                continue
            
            reasoning = reasoning.strip()
            
            if 'Qwen3' in model_name:
                assistant_content = f"<think>\n{reasoning}\n</think>\n{final_answer}"

                # Construct the full conversation and apply the chat template
                full_conversation = [
                    {"role": "user", "content": item['prompt']},
                    {"role": "assistant", "content": assistant_content},
                ]

                if include_no_think_instruction:
                    full_conversation.insert(0, {"role": "system", "content": "Respond only with the letter of the correct option. Don't write any explanations"})

                full_text = tokenizer.apply_chat_template(
                    full_conversation,
                    tokenize=False,
                    add_generation_prompt=False
                )
                if with_reasoning:
                    full_text = full_text.replace(
                        '<think>\n\n</think>\n\n<think>', '<think>'
                    )

            elif 'deepseek/' in model_name:
                assistant_content = f"<think>\n{reasoning}\n</THINK>\n{final_answer}"

                # Construct the full conversation and apply the chat template
                full_conversation = [
                    {"role": "user", "content": item['prompt']},
                    {"role": "assistant", "content": assistant_content},
                ]

                if include_no_think_instruction:
                    full_conversation.insert(0, {"role": "system", "content": "Respond only with the letter of the correct option. Don't write any explanations"})

                full_text = tokenizer.apply_chat_template(
                    full_conversation,
                    tokenize=False,
                    add_generation_prompt=False
                )
                if with_reasoning:
                    full_text = full_text.replace(
                        '</THINK>', '</think>'
                    )

            elif 'gpt-oss-20b' in model_name:
                assistant_response_text = f"<|start|>analysis<|message|>{reasoning}<|end|><|start|>final<|message|>{final_answer}<|return|>"
                prompt_conversation = [
                    {"role": "user", "content": item['prompt']}
                ]
                if include_no_think_instruction:
                    prompt_conversation.insert(0, {"role": "system", "content": "Respond only with the letter of the correct option. Don't write any explanations"})
                
                prompt_text = tokenizer.apply_chat_template(
                    prompt_conversation,
                    tokenize=False,
                    add_generation_prompt=False
                )
                full_text = prompt_text + assistant_response_text
            else:
                raise ValueError(f"Model family for '{model_name}' is not supported for prompt reconstruction.")

            # Extract the letter from the final answer
            matches = re.findall(r'[A-D]', final_answer)
            extracted_answer = matches[-1] if matches else None
            
            is_correct = None
            if extracted_answer is not None:
                is_correct = 1 if extracted_answer == item['answer_letter'] else 0
                
            return {
                "question_id": question_id,
                "model_path": model_name,
                "question_text": item["question"],
                "choices": json.dumps(item["choices"]),
                "correct_answer_letter": item["answer_letter"],
                "full_prompt_text": full_text,
                "full_prompt_token_ids": json.dumps([]), # OpenRouter doesn't provide token IDs
                "extracted_answer": extracted_answer,
                "is_correct": is_correct,
            }
        
        except json.JSONDecodeError as e:
            print(f"\n--- ERROR: Failed to decode JSON response from OpenRouter ---")
            print(f"Error: {e}")
            if response is not None:
                print(f"Response Status Code: {response.status_code}")
                print(f"Response Body (first 500 chars):\n{response.text[:500]}")
            print("------------------------------------------------------------")
            retries += 1
            time.sleep(5)
        
        except requests.RequestException as e:
            print(f"\n--- ERROR: Request to OpenRouter failed (Network/HTTP Error) ---")
            print(f"Error: {e}")
            if response is not None:
                print(f"Response Status Code: {response.status_code}")
                print(f"Response Body (first 500 chars):\n{response.text[:500]}")
            print("Full Traceback:")
            traceback.print_exc()
            print("------------------------------------------------------------")
            retries += 1
            time.sleep(5)

    # If all retries fail, return None
    return None


def main(
    model_name: str = "openai/gpt-oss-20b", #"deepseek/deepseek-r1-distill-llama-70b", #"Qwen/Qwen3-32B",
    dataset_name: str = "gpqa",
    dataset_config: str = "gpqa_main",
    dataset_split: str = "train",
    db_path: str = "reasoning_traces.sqlite",
    with_reasoning: bool = True,
    include_no_think_instruction: bool = True,
    num_samples: int = -1,
):
    """
    Runs inference on the GPQA dataset using the OpenRouter API with multiple threads,
    and saves results to an SQLite database.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set.")
        return

    # Load tokenizer to correctly format the final prompt string
    tokenizer = AutoTokenizer.from_pretrained(model_name.replace('deepseek/', 'deepseek-ai/'))

    # Load dataset
    if dataset_name == "gpqa":
        dataset = GPQADataset(config_name=dataset_config, split=dataset_split)
        table_base_name = "reasoning_traces_gpqa"
    elif dataset_name == "logiqa":
        dataset = LogiqaDataset(split=dataset_split)
        table_base_name = "reasoning_traces_logiqa"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Setup database
    if with_reasoning:
        table_name = table_base_name
    else:
        table_name = f"{table_base_name}_no_reasoning"
    
    conn = setup_database(db_path, table_name)
    cursor = conn.cursor()

    # Get already processed question texts to avoid re-generating
    cursor.execute(f"SELECT question_text FROM {table_name} WHERE model_path = ?", (model_name,))
    processed_texts = {row[0] for row in cursor.fetchall()}
    
    total_samples_to_consider = len(dataset)
    if num_samples > 0:
        total_samples_to_consider = min(num_samples, len(dataset))
        print(f"Limiting processing to the first {total_samples_to_consider} samples.")

    indices_to_process = []
    for i in range(total_samples_to_consider):
        if dataset[i]["question"] not in processed_texts:
            indices_to_process.append(i)

    if processed_texts:
        print(f"Found {len(processed_texts)} already processed entries for model '{model_name}' in table '{table_name}'.")
    
    if not indices_to_process:
        print(f"All questions for this model are already processed in table '{table_name}'.")
        conn.close()
        return
        
    print(f"Processing the remaining {len(indices_to_process)} questions.")

    tasks = [(i, dataset[i], model_name, api_key, tokenizer, with_reasoning, include_no_think_instruction) for i in indices_to_process]
    
    successful_writes = 0
    with concurrent.futures.ThreadPoolExecutor(10) as executor:
        # Submit all tasks and create a map from future to question_id for error reporting
        future_to_id = {executor.submit(process_item, task): task[0] for task in tasks}

        for future in tqdm(concurrent.futures.as_completed(future_to_id), total=len(tasks), desc=f"Processing with {model_name}"):
            question_id = future_to_id[future]
            try:
                result = future.result()
                if result:
                    cursor.execute(
                        f"""
                        INSERT INTO {table_name} (model_path, question_id, question_text, choices, correct_answer_letter, full_prompt_text, full_prompt_token_ids, extracted_answer, is_correct)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            result["model_path"],
                            result["question_id"],
                            result["question_text"],
                            result["choices"],
                            result["correct_answer_letter"],
                            result["full_prompt_text"],
                            result["full_prompt_token_ids"],
                            result["extracted_answer"],
                            result["is_correct"],
                        ),
                    )
                    conn.commit()  # Commit after each successful insert
                    successful_writes += 1
            except Exception as e:
                print(f"An error occurred processing question {question_id}: {e}")

    conn.close()
    
    print(f"\nProcessing complete. Successfully saved {successful_writes} new entries to {db_path} in table '{table_name}'")


if __name__ == "__main__":
    fire.Fire(main) 
