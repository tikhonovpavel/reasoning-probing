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
import re
from datasets import load_dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def to_float(s: str) -> Union[float, None]:
    """Safely converts a string to a float, handling commas and stripping non-numeric characters."""
    if s is None:
        return None
    try:
        # Handle cases like '1,000' -> '1000'
        s = s.replace(',', '')
        # Use regex to find the first valid number (int or float)
        match = re.search(r'-?\d+(\.\d+)?', s)
        if match:
            return float(match.group(0))
    except (ValueError, TypeError):
        return None
    return None

def extract_boxed_answer(text: str) -> Union[str, None]:
    """Extracts the content from a \\boxed{...} expression."""
    if text is None:
        return None
    match = re.search(r'\\boxed\{(.+?)\}', text)
    return match.group(1) if match else text # Return original text if no box found


def main(
    model_name: str = "Qwen/QwQ-32B",
    dataset_name: str = "PrimeIntellect/NuminaMath-QwQ-CoT-5M",
    dataset_split: str = "train",
    layer_indices: Union[int, List[int]] = 40,
    max_prompt_length: int = 4096,
    stride: int = 10,
    num_samples: int = 5000,
    cache_dir: str = "/mnt/nfs_share/tikhonov/hf_cache",
    chunk_size: int = 200,
    extraction_mode: str = "stride",
    min_paragraph_breaks: int = 10,
    max_paragraph_breaks: int = 300,
):
    """
    Generates a dataset for training a regression probe by running a model over reasoning traces
    from a Hugging Face dataset, extracting hidden states, and saving them to a file.

    Args:
        model_name (str): The name of the Hugging Face model to use.
        dataset_name (str): The name of the Hugging Face dataset to use.
        dataset_split (str): The split of the dataset to process.
        layer_indices (Union[int, List[int]]): Layer(s) from which to extract hidden states.
        max_prompt_length (int): Maximum number of tokens in a prompt to be processed.
        stride (int): The step size for subsampling hidden states (used in 'stride' mode).
        num_samples (int): The number of samples to process from the dataset.
        chunk_size (int): How many samples to save in each output file chunk.
        extraction_mode (str): 'stride' or 'paragraph_break'. Determines how hidden states are extracted.
        min_paragraph_breaks (int): Minimum paragraph breaks for a sample to be included in 'paragraph_break' mode.
        max_paragraph_breaks (int): Maximum paragraph breaks for a sample to be included in 'paragraph_break' mode.
    """
    if extraction_mode not in ["stride", "paragraph_break"]:
        raise ValueError("extraction_mode must be either 'stride' or 'paragraph_break'")

    if isinstance(layer_indices, int):
        layer_indices = [layer_indices]

    layer_indices = [int(layer_index) for layer_index in layer_indices]
    print(f"Generating probe data for layers: {layer_indices}")

    # --- Stage 1: Data Preprocessing (Tokenizer only) ---
    logging.info(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load dataset
    logging.info(f"Loading dataset {dataset_name} (split: {dataset_split})...")
    dataset = load_dataset(dataset_name, split=dataset_split, cache_dir=cache_dir)
    
    # --- Pre-filtering and Preprocessing Step ---
    logging.info(f"Searching for {num_samples} valid samples...")

    def process_and_validate(example):
        """Processes a single example, returning it with new fields if valid, otherwise None."""
        # 1. Parse and validate answers
        ground_truth_value = to_float(example['ground_truth'])
        model_answer_value = to_float(extract_boxed_answer(example['response']))
        if ground_truth_value is None or model_answer_value is None:
            return None

        # 2. Construct prompt and check length
        conversation = [
            {"role": "user", "content": example['prompt']},
            {"role": "assistant", "content": example['response']},
        ]
        token_ids = tokenizer.encode(tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        ))
        if len(token_ids) > max_prompt_length:
            return None

        # 3. Determine start position
        conversation_user_only = [{"role": "user", "content": example['prompt']}]
        text_user_only = tokenizer.apply_chat_template(conversation_user_only, tokenize=False, add_generation_prompt=True)
        tokens_user_only = tokenizer.encode(text_user_only, add_special_tokens=False)
        
        example['input_ids'] = token_ids
        example['start_pos'] = len(tokens_user_only)
        example['ground_truth_value'] = ground_truth_value
        example['model_answer_value'] = model_answer_value
        return example

    # Define the paragraph break filter function once
    def paragraph_break_filter(example):
        response_text = example.get('response', '')
        if not isinstance(response_text, str):
            return False
        # Use regex to find sequences of 2 or more newlines
        matches = re.findall(r'\n{2,}', response_text)
        return min_paragraph_breaks <= len(matches) <= max_paragraph_breaks

    final_dataset = []
    total_checked = 0
    with tqdm(total=num_samples, desc="Finding valid samples") as pbar:
        for example in dataset:
            total_checked += 1
            
            # Initial validation (numerical answer, etc.)
            processed_example = process_and_validate(example)
            if not processed_example:
                continue

            # In paragraph_break mode, we need to pre-calculate token indices for breaks
            if extraction_mode == 'paragraph_break':
                # This logic is now correct and robust, using offset_mapping.
                conversation = [
                    {"role": "user", "content": processed_example['prompt']},
                    {"role": "assistant", "content": processed_example['response']},
                ]
                full_text = tokenizer.apply_chat_template(
                    conversation, tokenize=False, add_generation_prompt=False
                )
                
                response_text = processed_example['response']
                response_char_start = full_text.find(response_text)
                if response_char_start == -1:
                    logging.warning(f"Could not find response text in chat template for sample {processed_example.get('problem_id', 'N/A')}. Skipping.")
                    continue

                encoding = tokenizer(full_text, return_offsets_mapping=True)
                offset_mapping = encoding['offset_mapping']

                break_char_spans = [
                    (m.start() + response_char_start, m.end() + response_char_start)
                    for m in re.finditer(r'\n{2,}', response_text)
                ]
                
                break_token_indices = []
                for start_char, end_char in break_char_spans:
                    current_break_tokens = []
                    for i, (token_start, token_end) in enumerate(offset_mapping):
                        if token_end > start_char and token_start < end_char:
                            current_break_tokens.append(i)
                    if current_break_tokens:
                        break_token_indices.append(current_break_tokens)
                
                processed_example['break_token_indices'] = break_token_indices
                
                if not (min_paragraph_breaks <= len(break_token_indices) <= max_paragraph_breaks):
                    continue

            final_dataset.append(processed_example)
            pbar.update(1)
            
            pbar.set_postfix({"checked": f"{total_checked}/{len(dataset)}"})

            if len(final_dataset) >= num_samples:
                break
    
    logging.info(f"Search complete. Found {len(final_dataset)} valid samples after checking {total_checked} examples.")

    if len(final_dataset) == 0:
        logging.warning("No valid samples found after filtering. Exiting.")
        return
    # This warning is no longer needed as the new logic ensures we find the target number if they exist.
    # if num_samples > 0 and len(final_dataset) < num_samples:
    #     logging.warning(f"Found only {len(final_dataset)} samples, which is less than the requested {num_samples}.")

    # --- Stage 2: Model Loading and Hidden State Extraction ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # --- Resume Logic ---
    output_dir = f"probe_data/layer_{layer_indices[0]}"
    os.makedirs(output_dir, exist_ok=True)
    start_sample_idx = 0
    start_chunk_idx = 0
    
    try:
        existing_chunks = [f for f in os.listdir(output_dir) if f.startswith('chunk_') and f.endswith('.pt')]
        if existing_chunks:
            # Sort chunks numerically to process them in order
            existing_chunks.sort(key=lambda f: int(re.search(r'chunk_(\d+).pt', f).group(1)))
            
            total_samples_found = 0
            for chunk_file in tqdm(existing_chunks, desc="Counting existing samples"):
                chunk_path = os.path.join(output_dir, chunk_file)
                try:
                    # This is the most reliable way to count: load and check length
                    total_samples_found += len(torch.load(chunk_path))
                except Exception as e:
                    logging.warning(f"Could not load or read length of chunk {chunk_file}: {e}. Skipping this chunk in count.")
            
            start_sample_idx = total_samples_found
            start_chunk_idx = len(existing_chunks)

            logging.info(f"Resuming from previous run. Found {len(existing_chunks)} completed chunks with exactly {start_sample_idx} samples.")
            logging.info(f"Skipping the first {start_sample_idx} samples.")
    except Exception as e:
        logging.warning(f"Could not parse existing chunk files for resuming: {e}. Starting from scratch.")

    dataset_to_process = final_dataset[start_sample_idx:]
    if not dataset_to_process:
        logging.info("All samples have already been processed. Exiting.")
        return

    logging.info(f"Loading model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        cache_dir=cache_dir
    ).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Forward hook setup
    hidden_states_containers = defaultdict(list)
    def create_hook_fn(layer_idx):
        def hook_fn(module, input, output):
            hidden_states_containers[layer_idx].append(output[0].cpu())
        return hook_fn

    hooks = []
    for layer_idx in layer_indices:
        try:
            # Adjusting for common model architectures (e.g., Llama, Qwen)
            hook = model.model.layers[layer_idx].register_forward_hook(create_hook_fn(layer_idx))
            hooks.append(hook)
        except AttributeError:
            raise RuntimeError(f"Could not find 'model.layers' attribute. Please check the model architecture of {model_name}.")
        except IndexError as e:
            print(model)
            raise e

    # Data collection loop
    logging.info(f"Starting hidden state extraction for {len(dataset_to_process)} samples (resuming at sample {start_sample_idx})...")
    all_probe_data_by_layer = defaultdict(list)
    chunk_counter = start_chunk_idx
    
    for i, sample in enumerate(tqdm(dataset_to_process, desc="Extracting hidden states")):
        
        input_ids = torch.tensor(sample['input_ids'], dtype=torch.long).to(device)
        start_pos = sample['start_pos']
        ground_truth_value = sample['ground_truth_value']
        model_answer_value = sample['model_answer_value']

        # Run model and extract hidden states
        for container in hidden_states_containers.values():
            container.clear()
        
        with torch.no_grad():
            model(input_ids.unsqueeze(0))

        if not any(hidden_states_containers.values()):
            # This warning is kept as it indicates a problem with the model pass, not filtering.
            logging.warning(f"No hidden states captured for sample {sample.get('problem_id', 'N/A')}. Skipping.")
            continue
        
        # Process and save data for each layer
        for layer_idx in layer_indices:
            if not hidden_states_containers[layer_idx]:
                logging.warning(f"No hidden states captured for sample {sample.get('problem_id', 'N/A')} at layer {layer_idx}. Skipping.")
                continue
            
            seq_hiddens = hidden_states_containers[layer_idx][0].squeeze(0)
            
            relevant_hiddens = None
            relevant_token_ids = None

            if extraction_mode == 'stride':
                # Subsample with stride
                relevant_hiddens = seq_hiddens[start_pos::stride]
                if relevant_hiddens.numel() > 0:
                    relevant_token_ids = input_ids[start_pos::stride]

            elif extraction_mode == 'paragraph_break':
                break_indices = sample.get('break_token_indices', [])
                
                if break_indices:
                    averaged_hiddens = []
                    # Placeholder for token_ids metadata.
                    newline_token_id = tokenizer.encode('\n', add_special_tokens=False)[0]

                    for break_group in break_indices:
                        if not break_group: continue
                        hiddens_to_average = seq_hiddens[break_group]
                        averaged_hidden = torch.mean(hiddens_to_average, dim=0)
                        averaged_hiddens.append(averaged_hidden)
                    
                    if averaged_hiddens:
                        relevant_hiddens = torch.stack(averaged_hiddens)
                        num_breaks = relevant_hiddens.shape[0]
                        relevant_token_ids = torch.full((num_breaks,), newline_token_id, dtype=torch.long)

            if relevant_hiddens is None or relevant_hiddens.numel() == 0:
                continue

            all_probe_data_by_layer[layer_idx].append({
                "hiddens": relevant_hiddens.cpu(),
                "token_ids": relevant_token_ids.cpu(),
                "ground_truth_value": ground_truth_value,
                "model_answer_value": model_answer_value,
                "question_id": sample.get('problem_id', -1),
                'prompt_len': len(input_ids)
            })

        # --- Chunk-based Saving ---
        is_last_sample = (i + 1) == len(dataset_to_process)
        # Use the buffer length to decide when to save, as it's independent of the loop index
        current_buffer_size = len(next(iter(all_probe_data_by_layer.values()), []))
        
        if (current_buffer_size >= chunk_size) or (is_last_sample and current_buffer_size > 0):
            logging.info(f"\nSaving chunk {chunk_counter} at sample {start_sample_idx + i + 1}/{len(final_dataset)}...")
            for layer_idx, data in all_probe_data_by_layer.items():
                if data:
                    # output_dir is defined in resume logic
                    output_path = os.path.join(output_dir, f"chunk_{chunk_counter}.pt")
                    
                    # Save to a temporary file first to prevent corruption
                    temp_output_path = output_path + ".tmp"
                    torch.save(data, temp_output_path)
                    os.rename(temp_output_path, output_path)
            
            logging.info(f"Chunk {chunk_counter} saved. Clearing memory buffer.")
            all_probe_data_by_layer.clear()
            chunk_counter += 1

    for hook in hooks:
        hook.remove()

    # The final save is now handled inside the loop
    logging.info("Data generation finished.")


if __name__ == "__main__":
    fire.Fire(main)
