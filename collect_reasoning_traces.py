import sqlite3
import os
import json
import torch
import fire
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from gpqa_dataset import GPQADataset
import time


def setup_database(db_path: str):
    """Initializes the database and creates the reasoning_traces_qpqa table if it doesn't exist."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reasoning_traces_qpqa (
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

def main(
    model_name: str = "Qwen/Qwen3-32B",
    dataset_config: str = "gpqa_diamond",
    dataset_split: str = "train",
    db_path: str = "reasoning_traces.sqlite",
):
    """
    Runs inference on the GPQA dataset, streams the output, and saves results to an SQLite database.

    Args:
        model_name: The name of the Hugging Face model to use.
        dataset_config: The configuration for the GPQA dataset.
        dataset_split: The split of the dataset to use (e.g., 'train').
        db_path: The path to the SQLite database file.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
    ).to(device)

    # Load dataset
    dataset = GPQADataset(config_name=dataset_config, split=dataset_split)

    # Setup database
    conn = setup_database(db_path)
    cursor = conn.cursor()

    # Get already processed question IDs to avoid re-generating
    cursor.execute("SELECT question_id FROM reasoning_traces_qpqa")
    processed_ids = {row[0] for row in cursor.fetchall()}
    indices_to_process = [i for i in range(len(dataset)) if i not in processed_ids]

    if processed_ids:
        print(f"Found {len(processed_ids)} already processed entries. Processing the remaining {len(indices_to_process)}.")

    # Main loop
    start_time = time.time()
    total_to_process = len(indices_to_process)
    for loop_idx, question_id in enumerate(indices_to_process):
        os.system('cls' if os.name == 'nt' else 'clear')

        elapsed_time = time.time() - start_time
        elapsed_str = tqdm.format_interval(elapsed_time)

        remaining_str = 'N/A'
        # Calculate ETA after at least one item is fully processed
        if loop_idx > 0:
            rate = loop_idx / elapsed_time
            if rate > 0:
                remaining_items = total_to_process - loop_idx
                eta_seconds = remaining_items / rate
                remaining_str = tqdm.format_interval(eta_seconds)
        
        print(f"Processing: {loop_idx + 1}/{total_to_process} (Question ID: {question_id}) | Elapsed: {elapsed_str} | ETA: {remaining_str}")
        
        item = dataset[question_id]
        
        print(f"Question: {item['question']}")
        print(f"Correct Answer: {item['answer_letter']}")
        print("-" * 20)
        print("Model generation:")

        messages = [
            {"role": "system", "content": "Respond only with the letter of the correct option. Don't write any explanations"},
            {"role": "user", "content": item['prompt']},
            {"role": "assistant", "content": "<think>"},
        ]

        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            continue_final_message=True, 
            enable_thinking=False, 
            add_generation_prompt=False
        )
        
        prompt = prompt.replace(
            '<think>\n\n</think>\n\n<think>', '<think>'
        )

        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
        streamer = TextStreamer(tokenizer, skip_prompt=False, skip_special_tokens=False)

        output_ids = model.generate(
            inputs, 
            max_new_tokens=12800, 
            temperature=0.6, 
            streamer=streamer
        )

        full_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
        token_ids_list = output_ids[0].tolist()
        
        # Extract answer from generation
        generated_text = tokenizer.decode(output_ids[0][inputs.shape[-1]:], skip_special_tokens=False)
        matches = re.findall(r'[A-D]', generated_text)
        extracted_answer = matches[-1] if matches else None

        # Check if correct
        is_correct = None
        if extracted_answer is not None:
            is_correct = 1 if extracted_answer == item['answer_letter'] else 0
        
        # Save to database
        cursor.execute(
            """
            INSERT INTO reasoning_traces_qpqa (question_id, model_path, question_text, choices, correct_answer_letter, full_prompt_text, full_prompt_token_ids, extracted_answer, is_correct)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                question_id,
                model_name,
                item["question"],
                json.dumps(item["choices"]),
                item["answer_letter"],
                full_text,
                json.dumps(token_ids_list),
                extracted_answer,
                is_correct,
            ),
        )
        conn.commit()

    conn.close()
    print("\n\nProcessing complete. Results saved to", db_path)

if __name__ == "__main__":
    fire.Fire(main) 