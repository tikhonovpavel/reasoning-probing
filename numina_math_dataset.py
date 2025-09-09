import datasets
import re
import fire



def extract_final_number(text: str):
    """
    Flexible function for extracting a single numerical answer.
    Priority 1: Looks for \boxed{...}.
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

def main(cache_dir='/mnt/nfs_share/tikhonov/hf_cache', num_examples=3):
    print("Loading dataset (this may take some time as streaming=False)...")

    dataset = datasets.load_dataset(
        'PrimeIntellect/NuminaMath-QwQ-CoT-5M', 
        cache_dir=cache_dir,
        streaming=False
    )
    train_dataset = dataset['train']
    print(f"Dataset loaded. Total examples in 'train': {len(train_dataset):,}")

    print("\nFiltering first 1000 examples of dataset... (progress bar is automatic)")

    def is_numerical(example):
        return extract_final_number(example['ground_truth']) is not None

    filtered_dataset = train_dataset.select(range(1000)).filter(is_numerical, num_proc=100) 

    print("\nFiltering complete!")
    print(f"Total examples found with a numerical answer: {len(filtered_dataset):,}")
    print("=" * 100)

    print(f"Displaying the first {num_examples} found examples:\n")

    for i, d in enumerate(filtered_dataset.select(range(num_examples))):
        print(f"----------- Example {i+1} -----------")
        print("Item keys:")
        print(d.keys())
        print("Plain dataset item:")
        print(d)
        extracted_num = extract_final_number(d['ground_truth'])
        print(f"----> Extracted Number for Regression: {extracted_num}")
        print("=" * 100)


if __name__ == '__main__':
    fire.Fire(main)
