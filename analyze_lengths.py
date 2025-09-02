import sys
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
import re
from tqdm import tqdm
import logging

# Suppress some of the verbose logging from other libraries
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def extract_final_number(text: str):
    """
    Flexible function for extracting a single numerical answer from a string.
    """
    if not isinstance(text, str):
        return None
    try:
        matches = re.findall(r'\\boxed\{(.*?)\}', text)
        if len(matches) == 1:
            content = matches[0].strip().replace(',', '')
            if '/' in content:
                parts = content.split('/')
                if len(parts) == 2:
                    return float(parts[0]) / float(parts[1])
            return float(content)
    except (ValueError, TypeError):
        pass
    try:
        cleaned_text = text.strip().replace(',', '')
        if '/' in cleaned_text:
                parts = cleaned_text.split('/')
                if len(parts) == 2:
                    return float(parts[0]) / float(parts[1])
        return float(cleaned_text)
    except (ValueError, TypeError):
        return None

MODEL_NAME = "Qwen/QwQ-32B"
DATASET_NAME = "PrimeIntellect/NuminaMath-QwQ-CoT-5M"
CACHE_DIR = "/mnt/nfs_share/tikhonov/hf_cache/"
NUM_SAMPLES_TO_ANALYZE = 10000

logger.info(f"Loading tokenizer for {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)

logger.info(f"Loading dataset {DATASET_NAME} (streaming)...")
dataset = load_dataset(DATASET_NAME, split='train', streaming=True, cache_dir=CACHE_DIR)

token_lengths = []
valid_samples_found = 0

logger.info(f"Analyzing token lengths for the first {NUM_SAMPLES_TO_ANALYZE} valid samples...")

for example in tqdm(dataset, desc="Scanning dataset"):
    if valid_samples_found >= NUM_SAMPLES_TO_ANALYZE:
        logger.info(f"\nCollected {NUM_SAMPLES_TO_ANALYZE} valid samples. Stopping analysis.")
        break
        
    if extract_final_number(example['ground_truth']) is not None:
        full_chat = [
            {"role": "user", "content": example['prompt']},
            {"role": "assistant", "content": example['response']},
        ]
        
        token_ids = tokenizer.apply_chat_template(full_chat, tokenize=True, add_generation_prompt=False)
        token_lengths.append(len(token_ids))
        valid_samples_found += 1

print("\n" + "="*50)
print("--- Token Length Distribution Analysis ---")
print("="*50)

if token_lengths:
    lengths_arr = np.array(token_lengths)
    print(f"Total valid samples analyzed: {len(lengths_arr)}")
    print(f"Min length:    {np.min(lengths_arr)}")
    print(f"Max length:    {np.max(lengths_arr)}")
    print(f"Mean length:   {np.mean(lengths_arr):.2f}")
    print(f"Median length: {np.median(lengths_arr)}")
    print("\n" + "-"*20 + " Percentiles " + "-"*20)
    print(f"50th (Median): {np.percentile(lengths_arr, 50)}")
    print(f"75th:          {np.percentile(lengths_arr, 75)}")
    print(f"90th:          {np.percentile(lengths_arr, 90)}")
    print(f"95th:          {np.percentile(lengths_arr, 95)}")
    print(f"98th:          {np.percentile(lengths_arr, 98)}")
    print(f"99th:          {np.percentile(lengths_arr, 99)}")
    print(f"99.5th:        {np.percentile(lengths_arr, 99.5)}")
    print(f"99.9th:        {np.percentile(lengths_arr, 99.9)}")
    print("="*50)
else:
    print("No valid samples with numerical answers were found in the scanned portion of the dataset.")
