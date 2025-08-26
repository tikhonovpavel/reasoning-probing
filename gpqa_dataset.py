import random
import re
from pprint import pprint
from datasets import load_dataset
from torch.utils.data import Dataset

class GPQADataset(Dataset):
    """
    Class for the GPQA dataset, which loads and processes the data upon initialization,
    and then returns samples in the required format.
    """
    def __init__(self, config_name="gpqa_diamond", split="train", seed=42):
        """
        Loads and processes the data once.

        Args:
            config_name (str): GPQA dataset configuration ('gpqa_main', 'gpqa_diamond', etc.).
            split (str): Dataset split (e.g., 'train').
            seed (int): Seed for reproducible shuffling of answer choices.
        """
        self.config_name = config_name
        self.split = split
        self.seed = seed
        self.processed_data = self._load_and_process_data()

    def _preprocess_text(self, text):
        """Helper function to clean text from artifacts."""
        if text is None:
            return ""
        text = text.strip()
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")
        return text

    def _load_and_process_data(self):
        """Loads the raw dataset and converts it into a list of processed dictionaries."""
        # Load the raw dataset
        raw_dataset = load_dataset("Idavidrein/gpqa", self.config_name, split=self.split)
        
        processed_list = []
        random.seed(self.seed)

        for doc in raw_dataset:
            # Collect all answer choices and clean them
            correct_answer_text = self._preprocess_text(doc["Correct Answer"])
            choices = [
                self._preprocess_text(doc["Incorrect Answer 1"]),
                self._preprocess_text(doc["Incorrect Answer 2"]),
                self._preprocess_text(doc["Incorrect Answer 3"]),
                correct_answer_text,
            ]

            # Shuffle the choices for each question
            random.shuffle(choices)
            
            # Find the new index and letter of the correct answer
            correct_answer_index = choices.index(correct_answer_text)
            correct_answer_letter = chr(65 + correct_answer_index)

            # Save all required fields in processed form
            processed_list.append({
                "question": doc["Question"],
                "choices": choices, # Already shuffled list of 4 strings
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
        # Get the preprocessed sample
        item_data = self.processed_data[idx]
        
        question = item_data["question"]
        choices = item_data["choices"]

        # Compose the prompt in the requested format
        prompt = (
            f"{question}\n\n"
            f"A) {choices[0]}\n"
            f"B) {choices[1]}\n"
            f"C) {choices[2]}\n"
            f"D) {choices[3]}"
        )
        
        # Return a dictionary with all necessary fields
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
    # Create an instance of the dataset
    gpqa_dataset = GPQADataset()

    # Check the size
    print(f"Dataset size: {len(gpqa_dataset)}\n")

    # Get and print the first sample
    first_sample = gpqa_dataset[0]
    
    print("Structure of a sample:")
    pprint(first_sample)