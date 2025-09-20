import random
from pprint import pprint
from datasets import load_dataset
from torch.utils.data import Dataset

class LogiqaDataset(Dataset):
    """
    Class for the LogiQA dataset.
    """
    def __init__(self, split="train", seed=42, cache_dir=None):
        """
        Loads and processes the data once.

        Args:
            split (str): Dataset split (e.g., 'train', 'validation', 'test').
            seed (int): Seed for reproducibility.
            cache_dir (str, optional): Directory to cache the dataset. Defaults to None.
        """
        self.split = split
        self.seed = seed
        self.cache_dir = cache_dir
        self.processed_data = self._load_and_process_data()

    def _load_and_process_data(self):
        """Loads the raw dataset and converts it into a list of processed dictionaries."""
        raw_dataset = load_dataset("lucasmccabe/logiqa", "default", split=self.split, cache_dir=self.cache_dir)
        
        processed_list = []
        
        for doc in raw_dataset:
            # The question is a combination of context and the query itself
            question = f"{doc['context']}\n{doc['query']}"
            choices = doc["options"]
            correct_answer_index = doc["correct_option"]
            correct_answer_letter = chr(65 + correct_answer_index)
            correct_answer_text = choices[correct_answer_index]

            processed_list.append({
                "question": question,
                "choices": choices,
                "answer_text": correct_answer_text,
                "answer_index": correct_answer_index,
                "answer_letter": correct_answer_letter
            })
        return processed_list

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.processed_data)

    def __getitem__(self, idx):
        """
        Forms and returns a single data sample by index.
        """
        item_data = self.processed_data[idx]
        
        question = item_data["question"]
        choices = item_data["choices"]

        # Compose the prompt in the standard multiple-choice format
        prompt = (
            f"{question}\n\n"
            f"A) {choices[0]}\n"
            f"B) {choices[1]}\n"
            f"C) {choices[2]}\n"
            f"D) {choices[3]}"
        )
        
        return {
            "prompt": prompt,
            "question": question,
            "choices": choices,
            "answer_letter": item_data["answer_letter"],
            "answer_index": item_data["answer_index"],
            "answer_text": item_data["answer_text"]
        }

# --- Example usage ---
if __name__ == "__main__":
    logiqa_dataset = LogiqaDataset(split='train')
    print(f"Dataset size: {len(logiqa_dataset)}\n")
    
    first_sample = logiqa_dataset[0]
    print("Structure of a sample:")
    pprint(first_sample)
