import torch
import os
import fire
import logging
import random
import glob
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def analyze(
    probe_data_path: str = "probe_data/layer_45",
    target_type: str = "ground_truth",
    val_split_size: float = 0.2,
):
    """
    Analyzes the distribution of target values in the validation set
    of a chunked probe dataset.

    Args:
        probe_data_path (str): Path to the DIRECTORY containing probe data chunks.
        target_type (str): What to analyze: 'ground_truth' or 'model_answer'.
        val_split_size (float): Fraction of data used for validation.
    """
    assert target_type in ['ground_truth', 'model_answer'], "target_type must be 'ground_truth' or 'model_answer'"
    target_key = 'ground_truth_value' if target_type == 'ground_truth' else 'model_answer_value'

    logging.info(f"Searching for probe data chunks in {probe_data_path}...")
    chunk_files = sorted(glob.glob(os.path.join(probe_data_path, "chunk_*.pt")))
    if not chunk_files:
        logging.error(f"No chunk files found in {probe_data_path}")
        return
    
    logging.info(f"Found {len(chunk_files)} chunk files.")

    # --- Replicate the exact train/val split ---
    random.Random(42).shuffle(chunk_files)
    split_idx = int(len(chunk_files) * (1 - val_split_size))
    val_chunk_files = chunk_files[split_idx:]
    
    if not val_chunk_files:
        logging.warning("No files allocated for validation set. Check val_split_size.")
        return
        
    logging.info(f"Analyzing {len(val_chunk_files)} chunks corresponding to the validation set.")

    # --- Load data and collect target values ---
    target_values = []
    for chunk_file in tqdm(val_chunk_files, desc="Loading validation data"):
        try:
            chunk_data = torch.load(chunk_file)
            for sample in chunk_data:
                target_values.append(sample[target_key])
        except Exception as e:
            logging.error(f"Could not load or process {chunk_file}: {e}")
            continue
            
    if not target_values:
        logging.error("No target values were extracted from the validation chunks.")
        return

    # --- Calculate and Print Statistics ---
    values_np = np.array(target_values)
    logging.info("\n" + "="*50)
    logging.info(f"Statistics for '{target_key}' in the validation set:")
    logging.info(f"  Count: {len(values_np)}")
    logging.info(f"  Mean: {np.mean(values_np):.4f}")
    logging.info(f"  Std Dev: {np.std(values_np):.4f}")
    logging.info(f"  Min: {np.min(values_np):.4f}")
    logging.info(f"  25% (Q1): {np.percentile(values_np, 25):.4f}")
    logging.info(f"  50% (Median): {np.median(values_np):.4f}")
    logging.info(f"  75% (Q3): {np.percentile(values_np, 75):.4f}")
    logging.info(f"  90% (Q9): {np.percentile(values_np, 90):.4f}")
    logging.info(f"  95% (Q95): {np.percentile(values_np, 95):.4f}")
    logging.info(f"  Max: {np.max(values_np):.4f}")
    logging.info("="*50 + "\n")

    # --- Generate and Save Histogram ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    values_np_filtered = values_np[values_np < np.percentile(values_np, 95)]
    
    sns.histplot(values_np_filtered, bins=50, kde=False, ax=ax)
    
    ax.set_title(f'Distribution of "{target_key}" in Validation Set (filtered at Q95)', fontsize=16)
    ax.set_xlabel("Value", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_yscale('log')
    
    # Add a text box with statistics
    stats_str = (
        f"Count (filtered at Q95): {len(values_np_filtered)}\n"
        f"Mean: {np.mean(values_np):.2f}\n"
        f"Median: {np.median(values_np_filtered):.2f}\n"
        f"Std Dev: {np.std(values_np_filtered):.2f}\n"
        f"Min: {np.min(values_np_filtered):.2f}\n"
        f"Max: {np.max(values_np_filtered):.2f}"
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.95, 0.95, stats_str, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    
    output_filename = f"validation_distribution_{target_key}_filtered_at_Q95.png"
    plt.savefig(output_filename)
    logging.info(f"Histogram saved to {output_filename}")


if __name__ == "__main__":
    fire.Fire(analyze)
