import datasets
from tqdm import tqdm
import re
from typing import Union
import fire

def extract_final_number(text: str) -> Union[float, None]:
    """Extracts a numerical answer from text, prioritizing \\boxed expressions."""
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
    """Checks if the ground_truth in an example is numerical."""
    return extract_final_number(example.get('ground_truth')) is not None

def verify_quad_newline(
    dataset_name: str = "PrimeIntellect/NuminaMath-QwQ-CoT-5M",
    dataset_split: str = "train",
    num_samples_to_check: int = 10000,
    cache_dir: str = "/mnt/nfs_share/tikhonov/hf_cache",
):
    """
    Checks for the occurrence of quadruple newlines ('\\n\\n\\n\\n') in a specified
    number of samples with numerical ground truth.
    """
    print(f"Loading dataset {dataset_name}...")
    dataset = datasets.load_dataset(
        dataset_name, 
        split=dataset_split, 
        cache_dir=cache_dir,
        streaming=True
    )
    
    print(f"Searching for '\\n\\n\\n\\n' in the first {num_samples_to_check:,} samples with numerical ground truth...")

    found_count = 0
    checked_numerical_samples = 0
    
    pbar = tqdm(total=num_samples_to_check, desc="Checking numerical samples")

    for example in dataset:
        if checked_numerical_samples >= num_samples_to_check:
            break

        if is_numerical(example):
            response = example.get('response')
            if isinstance(response, str) and '\n\n\n\n' in response:
                found_count += 1
                problem_id = example.get('problem_id', 'N/A')
                print(f"\n----> Found '\\n\\n\\n\\n' in problem_id: {problem_id}")

            checked_numerical_samples += 1
            pbar.update(1)

    pbar.close()

    print("\n" + "="*50)
    print("Verification Complete")
    print("="*50)
    print(f"Total numerical samples checked: {checked_numerical_samples:,}")
    print(f"Total samples containing '\\n\\n\\n\\n': {found_count}")
    print("="*50)

if __name__ == "__main__":
    fire.Fire(verify_quad_newline)
