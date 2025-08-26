import sqlite3
import os
import json
import torch
import fire
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, Dataset
import logging
from collections import defaultdict
from typing import List, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ProbeDataPreparationDataset(Dataset):
    """
    Dataset for loading reasoning traces from the SQLite database for probe data generation.
    It loads all samples from the database for a given model.
    """
    def __init__(self, db_path: str, tokenizer: AutoTokenizer, model_name: str):
        self.tokenizer = tokenizer
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

        # Fetch all data for the model to perform deduplication in memory based on question_text
        self.cursor.execute(
            "SELECT question_id, full_prompt_text, correct_answer_letter, extracted_answer, question_text FROM reasoning_traces_qpqa WHERE model_path=?",
            (model_name,)
        )
        all_rows = self.cursor.fetchall()
        logging.info(f"Found {len(all_rows)} candidate rows for model {model_name}. Deduplicating by question text...")

        self.data = []
        seen_question_texts = set()
        for row in tqdm(all_rows, desc="Loading and deduplicating data"):
            question_id, full_prompt_text, correct_answer_letter, extracted_answer, question_text = row
            
            if question_text in seen_question_texts:
                continue
            seen_question_texts.add(question_text)

            token_ids = tokenizer.encode(full_prompt_text)
            if len(token_ids) > 12_500: # Same filter as in training script
                logging.warning(f"Skipping question '{question_text[:50]}...' due to long prompt ({len(token_ids)} tokens).")
                continue

            self.data.append({
                "question_id": question_id, # Kept for compatibility, but uniqueness is now based on text
                "token_ids": token_ids,
                "correct_answer_letter": correct_answer_letter,
                "extracted_answer": extracted_answer,
            })
        
        logging.info(f"Loaded {len(self.data)} unique questions after deduplication.")
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "question_id": item['question_id'],
            "token_ids": torch.tensor(item['token_ids'], dtype=torch.long),
            "correct_answer_letter": item['correct_answer_letter'],
            "extracted_answer": item['extracted_answer'],
        }

def main(
    model_name: str = "Qwen/Qwen3-32B",
    db_path: str = "reasoning_traces.sqlite",
    layer_indices: Union[int, List[int]] = 32,
):
    """
    Generates a dataset for training a probe by running a model over reasoning traces,
    extracting hidden states, and saving them to a file.
    
    Args:
        model_name (str): The name of the Hugging Face model to use.
        db_path (str): Path to the SQLite database with reasoning traces.
        layer_indices (Union[int, List[int]]): The layer or layers from which to extract hidden states.
    """
    if isinstance(layer_indices, int):
        layer_indices = [layer_indices]

    layer_indices = [int(layer_index) for layer_index in layer_indices]
    print(f"Generating probe data for layers: {layer_indices}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # Load Tokenizer and Model
    logging.info(f"Loading model and tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
    ).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    print(model)

    # Setup dataset and dataloader
    dataset = ProbeDataPreparationDataset(db_path, tokenizer, model_name)
    # Use a custom collate_fn to handle single items from the dataset
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])

    # Answer mapping
    answer_map = {letter: i for i, letter in enumerate(['A', 'B', 'C', 'D'])}
    think_token_id = tokenizer.convert_tokens_to_ids("<think>")
    assert think_token_id is not None, "Think token not found"

    # Forward hook setup
    hidden_states_containers = defaultdict(list)
    def create_hook_fn(layer_idx):
        def hook_fn(module, input, output):
            hidden_states_containers[layer_idx].append(output[0].cpu())
        return hook_fn

    hooks = []
    for layer_idx in layer_indices:
        try:
            hook = model.model.layers[layer_idx].register_forward_hook(create_hook_fn(layer_idx))
            hooks.append(hook)
        except IndexError as e:
            print(model)
            raise e

    # Data collection loop
    logging.info(f"Starting hidden state extraction for layers: {layer_indices}...")
    all_probe_data_by_layer = defaultdict(list)
    for sample in tqdm(dataloader, desc="Extracting hidden states"):
        input_ids = sample["token_ids"].to(device)

        for container in hidden_states_containers.values():
            container.clear()
        
        with torch.no_grad():
            model(input_ids.unsqueeze(0))

        if not any(hidden_states_containers.values()):
            logging.warning(f"No hidden states captured for question_id {sample['question_id']}. Skipping.")
            continue
        
        think_token_positions = (input_ids == think_token_id).nonzero(as_tuple=True)[0]
        if think_token_positions.numel() == 0:
            continue

        start_pos = think_token_positions[0]
        
        ground_truth_letter = sample['correct_answer_letter']
        ground_truth_idx = answer_map.get(ground_truth_letter)

        model_answer_letter = sample['extracted_answer']
        model_answer_idx = answer_map.get(model_answer_letter, -1)
        
        if model_answer_idx == -1:
            print(f'Warning: model answer is {model_answer_letter} for question_id {sample["question_id"]}')

        if ground_truth_idx is None:
            continue
            
        for layer_idx in layer_indices:
            if not hidden_states_containers[layer_idx]:
                logging.warning(f"No hidden states captured for question_id {sample['question_id']} at layer {layer_idx}. Skipping.")
                continue
            
            seq_hiddens = hidden_states_containers[layer_idx][0].squeeze(0)
            relevant_hiddens = seq_hiddens[start_pos:]
            
            if relevant_hiddens.numel() == 0:
                continue

            relevant_token_ids = input_ids[start_pos:]

            all_probe_data_by_layer[layer_idx].append({
                "hiddens": relevant_hiddens.cpu(),
                "token_ids": relevant_token_ids.cpu(),
                "ground_truth_idx": ground_truth_idx,
                "model_answer_idx": model_answer_idx,
                "question_id": sample['question_id'],
                'prompt_len': len(input_ids)
            })

    for hook in hooks:
        hook.remove()

    # Save the collected data
    for layer_idx, data in all_probe_data_by_layer.items():
        if data:
            output_path = f"probe_data/probe_data_{layer_idx}.pt"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            logging.info(f"Saving {len(data)} processed samples for layer {layer_idx} to {output_path}...")
            torch.save(data, output_path)
        else:
            logging.warning(f"No data was collected for layer {layer_idx}. The output file will not be created.")
    
    logging.info("Data generation finished.")


if __name__ == "__main__":
    fire.Fire(main)
