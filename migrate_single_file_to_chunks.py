import torch
import os
import fire
import logging
import re
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def migrate(
    input_file: str = "probe_data/probe_data_regression_45.pt",
    chunk_size: int = 200,
):
    """
    Loads a large single .pt file containing a list of data and splits it
    into smaller chunked .pt files in a layer-specific directory.
    """
    if not os.path.exists(input_file):
        logging.error(f"Input file not found: {input_file}")
        return

    # Extract layer index from filename to create the correct output directory
    match = re.search(r'_(\d+)\.pt$', input_file)
    if not match:
        logging.error(f"Could not extract layer index from filename: {input_file}")
        logging.error("Expected format like 'probe_data_regression_LAYER.pt'")
        return
    
    layer_idx = match.group(1)
    output_dir = f"probe_data/layer_{layer_idx}"

    logging.info(f"Loading data from {input_file}...")
    try:
        data = torch.load(input_file)
        if not isinstance(data, list):
            logging.error("Expected input file to contain a list of samples.")
            return
        logging.info(f"Loaded {len(data)} samples.")
    except Exception as e:
        logging.error(f"Failed to load data file: {e}")
        return

    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Splitting data into chunks of size {chunk_size} and saving to {output_dir}...")

    num_chunks = (len(data) + chunk_size - 1) // chunk_size
    for i in tqdm(range(num_chunks), desc="Saving chunks"):
        chunk_data = data[i * chunk_size : (i + 1) * chunk_size]
        chunk_path = os.path.join(output_dir, f"chunk_{i}.pt")
        try:
            torch.save(chunk_data, chunk_path)
        except Exception as e:
            logging.error(f"Failed to save chunk {i}: {e}")
            return

    migrated_path = input_file + ".migrated"
    logging.info(f"Successfully migrated {len(data)} samples into {num_chunks} chunks.")
    logging.warning(f"Migration complete. Please manually rename the original file to avoid re-processing.")
    logging.warning(f"Suggested command: mv {input_file} {migrated_path}")

if __name__ == "__main__":
    fire.Fire(migrate)
