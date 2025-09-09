import fire
import datasets
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import os

from numina_math_dataset import extract_final_number

# Increase matplotlib font size for readability
plt.rcParams.update({'font.size': 14})


def generate_text_histogram(data, bins=50, width=80):
    """Generates a string containing a text-based histogram."""
    if not data:
        return "No data to display."

    counts, bin_edges = np.histogram(data, bins=bins)
    max_count = counts.max()
    
    hist_str = ""
    for i in range(len(counts)):
        bar_len = int((counts[i] / max_count) * (width - 25)) if max_count > 0 else 0
        bar = '█' * bar_len
        range_str = f"{bin_edges[i]:>8.2f} - {bin_edges[i+1]:<8.2f}"
        hist_str += f"{range_str} | {bar} ({counts[i]})\n"
        
    return hist_str

def main(
    cache_dir: str = "/mnt/nfs_share/tikhonov/hf_cache",
    output_dir: str = "analysis_results",
    num_proc: int = 32
):
    """
    Analyzes the distribution of numerical answers in the NuminaMath dataset
    after filtering outliers.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to '{output_dir}'")

    # 1. Load dataset
    print("Loading NuminaMath dataset...")
    dataset = datasets.load_dataset(
        'PrimeIntellect/NuminaMath-QwQ-CoT-5M',
        cache_dir=cache_dir,
        split='train'
    )
    print(f"Dataset loaded with {len(dataset):,} examples.")

    # 2. Filter for numerical answers using datasets.map for parallelism
    print(f"Extracting all numerical answers using {num_proc} processes...")
    
    # Use .map to create a new column with the extracted number
    mapped_dataset = dataset.map(
        lambda example: {'number': extract_final_number(example['ground_truth'])},
        num_proc=num_proc,
        desc="Extracting numbers"
    )

    # Filter out examples where no number was found.
    # Instead of a slow, disk-based .filter(), we pull the column into memory
    # and use a fast list comprehension.
    print("Extracting column and filtering in memory (this should be fast)...")
    all_extracted_values = mapped_dataset['number']
    all_numbers = [num for num in tqdm(all_extracted_values, desc="Filtering Nones in-memory") if num is not None]

    print(f"Found {len(all_numbers):,} numerical answers.")
    if not all_numbers:
        print("No numerical answers found. Exiting.")
        return

    # 3. Calculate percentile bounds
    lower_perc = 1.0
    upper_perc = 95.0
    print(f"Calculating {lower_perc}% and {upper_perc}% percentiles...")
    lower_bound = np.percentile(all_numbers, lower_perc)
    upper_bound = np.percentile(all_numbers, upper_perc)
    print(f"Percentile bounds: lower={lower_bound:.2f}, upper={upper_bound:.2f}")

    # 4. Filter out outliers
    print("Filtering out outliers...")
    filtered_numbers = [num for num in all_numbers if lower_bound <= num <= upper_bound]
    print(f"Data after outlier removal: {len(filtered_numbers):,} examples remaining.")

    # 5. Print statistics
    print("\n--- Statistics of Filtered Data ---")
    print(f"Count:    {len(filtered_numbers):,}")
    print(f"Mean:     {np.mean(filtered_numbers):.2f}")
    print(f"Std Dev:  {np.std(filtered_numbers):.2f}")
    print(f"Min:      {np.min(filtered_numbers):.2f}")
    print(f"25%:      {np.percentile(filtered_numbers, 25):.2f}")
    print(f"Median:   {np.median(filtered_numbers):.2f}")
    print(f"75%:      {np.percentile(filtered_numbers, 75):.2f}")
    print(f"Max:      {np.max(filtered_numbers):.2f}")
    print("-------------------------------------\n")

    # 6. Generate and save graphical histogram
    print("Generating graphical histogram...")
    plt.figure(figsize=(14, 7))
    plt.hist(filtered_numbers, bins=100, color='skyblue', edgecolor='black')
    plt.title(f'Distribution of NuminaMath Answers\n(After filtering between {lower_perc}% and {upper_perc}% percentiles)')
    plt.xlabel('Numerical Answer')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    
    hist_path = os.path.join(output_dir, "numina_distribution_histogram.png")
    plt.savefig(hist_path)
    print(f"Histogram saved to {hist_path}")
    plt.close()

    # 7. Generate and print text histogram
    print("\n--- Text Histogram ---")
    text_hist = generate_text_histogram(filtered_numbers, bins=30, width=100)
    print(text_hist)
    
    text_hist_path = os.path.join(output_dir, "numina_distribution_histogram.txt")
    with open(text_hist_path, "w") as f:
        f.write(text_hist)
    print(f"Text histogram saved to {text_hist_path}")


if __name__ == '__main__':
    fire.Fire(main)
