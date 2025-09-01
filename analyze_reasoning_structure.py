import datasets
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import fire
import os
import re
from typing import Union

def extract_final_number(text: str) -> Union[float, None]:
    """
    Flexible function for extracting a single numerical answer.
    Priority 1: Looks for \\boxed{...}.
    Priority 2: Checks if the entire string is a number.
    """
    if not isinstance(text, str):
        return None
    try:
        matches = re.findall(r'\\boxed\{(.*?)\}', text)
        if len(matches) == 1:
            content = matches[0].strip().replace(',', '')
            return float(content)
    except (ValueError, TypeError):
        pass
    try:
        cleaned_text = text.strip().replace(',', '')
        return float(cleaned_text)
    except (ValueError, TypeError):
        return None

def is_numerical(example):
    return extract_final_number(example['ground_truth']) is not None

def analyze_dataset(
    dataset_name: str = "PrimeIntellect/NuminaMath-QwQ-CoT-5M",
    dataset_split: str = "train",
    num_samples_to_analyze: int = 10000,
    cache_dir: str = "/mnt/nfs_share/tikhonov/hf_cache",
    output_dir: str = "analysis_results"
):
    """
    Analyzes the structure of reasoning text in a dataset.

    This script counts the occurrences of '.' and '\\n' in the 'response' field
    of each sample and generates histograms and summary statistics for both
    the full and filtered datasets.

    Args:
        dataset_name (str): The name of the Hugging Face dataset.
        dataset_split (str): The split of the dataset to process.
        num_samples_to_analyze (int): The number of samples to analyze.
        cache_dir (str): Directory for Hugging Face cache.
        output_dir (str): Directory to save analysis results (e.g., plots).
    """
    print(f"Loading dataset {dataset_name}...")
    dataset = datasets.load_dataset(
        dataset_name, 
        split=dataset_split, 
        cache_dir=cache_dir,
        streaming=True
    )
    
    print(f"Analyzing {num_samples_to_analyze:,} samples with numerical ground truth...")

    dot_counts = []
    newline_counts = []
    double_newline_counts = []
    paragraph_break_counts = []

    pbar = tqdm(total=num_samples_to_analyze, desc="Finding and analyzing samples")
    
    analyzed_count = 0
    for example in dataset:
        if analyzed_count >= num_samples_to_analyze:
            break

        if is_numerical(example):
            response = example.get('response')
            if isinstance(response, str):
                dot_counts.append(response.count('.'))
                newline_counts.append(response.count('\n'))
                double_newline_counts.append(response.count('\n\n'))
                # Count sequences of 2 or more newlines as one entity
                paragraph_break_counts.append(len(re.findall(r'\n{2,}', response)))
                analyzed_count += 1
                pbar.update(1)

    pbar.close()

    dot_counts = np.array(dot_counts)
    newline_counts = np.array(newline_counts)
    double_newline_counts = np.array(double_newline_counts)
    paragraph_break_counts = np.array(paragraph_break_counts)

    os.makedirs(output_dir, exist_ok=True)
    print(f"\nAnalysis results will be saved to '{output_dir}/'")

    def print_stats(name, data):
        if len(data) == 0:
            print(f"\n--- {name} ---")
            print("No data found.")
            return
        print(f"\n--- {name} ---")
        print(f"  Samples considered: {len(data):,}")
        print(f"  Average: {np.mean(data):.2f}")
        print(f"  Median: {np.median(data):.2f}")
        print(f"  Standard Deviation: {np.std(data):.2f}")
        print(f"  Min: {np.min(data)}")
        print(f"  Max: {np.max(data)}")

    def plot_histogram(data, title, filename, original_sample_count=None):
        if len(data) == 0:
            print(f"Skipping plot for '{title}' as no data is available.")
            return
            
        final_title = title
        if original_sample_count is not None and len(data) != original_sample_count:
            percentage = (len(data) / original_sample_count) * 100
            final_title += f"\n(Displaying {len(data):,} / {original_sample_count:,} samples - {percentage:.1f}%)"

        plt.figure(figsize=(12, 6))
        max_val = int(np.max(data))
        # Use a reasonable number of bins for clarity, especially for wide ranges
        bins = min(100, max_val + 1) if max_val > 0 else 1
        plt.hist(data, bins=bins, edgecolor='black', alpha=0.7)
        plt.title(final_title, fontsize=16)
        plt.xlabel("Count per Response")
        plt.ylabel("Number of Samples")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath)
        plt.close()
        print(f"  Histogram saved to: {filepath}")

    print("\n" + "="*50)
    print("Dot ('.') Analysis")
    print("="*50)
    # Full
    print_stats("Dot Counts [Full Data]", dot_counts)
    plot_histogram(dot_counts, "Distribution of Dot ('.') Counts [Full Data]", "dot_counts_histogram_full.png")
    # Filtered
    filtered_dot_counts = dot_counts[dot_counts <= 400]
    print_stats("Dot Counts [Filtered, <= 400]", filtered_dot_counts)
    plot_histogram(filtered_dot_counts, "Distribution of Dot ('.') Counts [Filtered, <= 400]", "dot_counts_histogram_filtered.png", original_sample_count=len(dot_counts))

    print("\n" + "="*50)
    print("Newline ('\\n') Analysis")
    print("="*50)
    # Full
    print_stats("Newline Counts [Full Data]", newline_counts)
    plot_histogram(newline_counts, "Distribution of Newline ('\\n') Counts [Full Data]", "newline_counts_histogram_full.png")
    # Filtered
    newline_filter_mask = (newline_counts >= 10) & (newline_counts <= 1000)
    filtered_newline_counts = newline_counts[newline_filter_mask]
    print_stats("Newline Counts [Filtered, 10 <= n <= 1000]", filtered_newline_counts)
    plot_histogram(filtered_newline_counts, "Distribution of Newline ('\\n') Counts [Filtered, 10-1000]", "newline_counts_histogram_filtered.png", original_sample_count=len(newline_counts))
    
    print("\n" + "="*50)
    print("Double Newline ('\\n\\n') Analysis")
    print("="*50)
    # Full
    print_stats("Double Newline Counts [Full Data]", double_newline_counts)
    plot_histogram(double_newline_counts, "Distribution of Double Newline ('\\n\\n') Counts [Full Data]", "double_newline_counts_histogram_full.png")
    # Filtered
    double_newline_filter_mask = (double_newline_counts > 0) & (double_newline_counts <= 400)
    filtered_double_newline_counts = double_newline_counts[double_newline_filter_mask]
    print_stats("Double Newline Counts [Filtered, 0 < n <= 400]", filtered_double_newline_counts)
    plot_histogram(filtered_double_newline_counts, "Distribution of Double Newline ('\\n\\n') Counts [Filtered, 0 < n <= 400]", "double_newline_counts_histogram_filtered.png", original_sample_count=len(double_newline_counts))

    print("\n" + "="*50)
    print("Paragraph Break (\\n{2,}) Analysis")
    print("="*50)
    # Full
    print_stats("Paragraph Break Counts [Full Data]", paragraph_break_counts)
    plot_histogram(paragraph_break_counts, "Distribution of Paragraph Break ('\\n{2,}') Counts [Full Data]", "paragraph_break_counts_histogram_full.png")

    print("\n" + "="*50)


if __name__ == "__main__":
    fire.Fire(analyze_dataset)
