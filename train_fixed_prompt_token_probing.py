import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from gpqa_dataset import GPQADataset
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import random
import fire
import json
import os
import neptune
import logging
from neptune.types import File
import matplotlib.pyplot as plt
import seaborn as sns
import re
import datasets
from typing import Optional
from sklearn.preprocessing import StandardScaler

from numina_math_dataset import extract_final_number
from torch.utils.data import Dataset
import glob

# --- Configure logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =================================================================================
# 1. Probe Definition
# =================================================================================

class ProbeMlpClassifier(nn.Module):
    """An MLP-based probe for classification."""
    def __init__(self, hidden_size: int, mlp_hidden_dim: int, num_classes: int, device: str, dtype: torch.dtype):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, num_classes)
        ).to(device=device, dtype=dtype)

    def forward(self, hidden_states: torch.Tensor):
        """
        Args:
            hidden_states (torch.Tensor): Shape (batch_size, hidden_size).
        Returns:
            torch.Tensor: Predicted logits, shape (batch_size, num_classes).
        """
        return self.classifier(hidden_states)

class ProbeMlpRegressor(nn.Module):
    """An MLP-based probe for regression."""
    def __init__(self, hidden_size: int, mlp_hidden_dim: int, device: str, dtype: torch.dtype):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 1)
        ).to(device=device, dtype=dtype)

    def forward(self, hidden_states: torch.Tensor):
        """
        Args:
            hidden_states (torch.Tensor): Shape (batch_size, hidden_size).
        Returns:
            torch.Tensor: Predicted values, shape (batch_size, 1).
        """
        return self.regressor(hidden_states)

class ProbeAttention(nn.Module):
    """A single-head attention-based probe."""
    def __init__(self, hidden_size: int, attention_dim: int, device: str, dtype: torch.dtype):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_dim = attention_dim
        
        self.q_proj = nn.Linear(hidden_size, attention_dim, bias=False).to(device=device, dtype=dtype)
        self.k_proj = nn.Linear(hidden_size, attention_dim, bias=False).to(device=device, dtype=dtype)
        self.v_proj = nn.Linear(hidden_size, attention_dim, bias=False).to(device=device, dtype=dtype)
        
        self.scale = torch.sqrt(torch.tensor(attention_dim, dtype=dtype, device=device))

    def forward(self, hidden_states: torch.Tensor):
        """
        Args:
            hidden_states (torch.Tensor): Shape (batch_size, seq_len, hidden_size).
        Returns:
            torch.Tensor: Context vector, shape (batch_size, attention_dim).
        """
        # Query is based on the last token's hidden state
        last_token_hidden = hidden_states[:, -1, :].unsqueeze(1) # (batch_size, 1, hidden_size)
        
        q = self.q_proj(last_token_hidden) # (batch_size, 1, attention_dim)
        k = self.k_proj(hidden_states)     # (batch_size, seq_len, attention_dim)
        v = self.v_proj(hidden_states)     # (batch_size, seq_len, attention_dim)
        
        # Scaled dot-product attention
        attn_weights = torch.bmm(q, k.transpose(1, 2)) / self.scale # (batch_size, 1, seq_len)
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        context = torch.bmm(attn_weights, v) # (batch_size, 1, attention_dim)
        
        return context.squeeze(1) # (batch_size, attention_dim)

class ProbeAttentionClassifier(nn.Module):
    """An Attention-based probe for classification."""
    def __init__(self, hidden_size: int, attention_dim: int, num_classes: int, device: str, dtype: torch.dtype):
        super().__init__()
        self.attention = ProbeAttention(hidden_size, attention_dim, device, dtype)
        self.classifier = nn.Linear(attention_dim, num_classes).to(device=device, dtype=dtype)

    def forward(self, hidden_states: torch.Tensor):
        context_vector = self.attention(hidden_states)
        return self.classifier(context_vector)

class ProbeAttentionRegressor(nn.Module):
    """An Attention-based probe for regression."""
    def __init__(self, hidden_size: int, attention_dim: int, device: str, dtype: torch.dtype):
        super().__init__()
        self.attention = ProbeAttention(hidden_size, attention_dim, device, dtype)
        self.regressor = nn.Linear(attention_dim, 1).to(device=device, dtype=dtype)

    def forward(self, hidden_states: torch.Tensor):
        context_vector = self.attention(hidden_states)
        return self.regressor(context_vector)


# =================================================================================
# 2. Data Handling for Precomputed States
# =================================================================================

class PrecomputedHiddenStateDataset(Dataset):
    """
    A PyTorch dataset that loads pre-computed hidden states and labels from disk.
    Assumes data is stored as individual .pt files in a directory.
    """
    def __init__(self, data_path: str):
        self.data_path = data_path
        # Find all sample files and sort them numerically to ensure order
        self.sample_files = sorted(
            glob.glob(os.path.join(data_path, "sample_*.pt")),
            key=lambda f: int(re.search(r"sample_(\d+)\.pt", f).group(1))
        )
        self.label_files = sorted(
            glob.glob(os.path.join(data_path, "label_*.pt")),
            key=lambda f: int(re.search(r"label_(\d+)\.pt", f).group(1))
        )
        if len(self.sample_files) != len(self.label_files):
            raise ValueError("Mismatch between number of sample files and label files.")
        if not self.sample_files:
            raise FileNotFoundError(f"No data files (sample_*.pt) found in {data_path}")

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, idx):
        hiddens = torch.load(self.sample_files[idx])
        label = torch.load(self.label_files[idx])
        return hiddens, label

def collate_fn_pad(batch):
    """
    Collate function that pads hidden state sequences to the max length in the batch.
    """
    # Separate hidden states and labels
    hiddens = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # Pad the hidden states
    padded_hiddens = torch.nn.utils.rnn.pad_sequence(
        hiddens, batch_first=True, padding_value=0.0
    )

    # Stack the labels
    labels_tensor = torch.stack(labels)
    
    return padded_hiddens, labels_tensor


# =================================================================================
# 3. Setup and Data Extraction
# =================================================================================

def get_data_path(output_dir, model_name, task_type, layer, n_samples):
    """Generates a unique directory path for storing/retrieving generated data."""
    model_name_safe = model_name.replace("/", "_")
    return os.path.join(
        output_dir,
        "generated_data",
        f"{model_name_safe}__{task_type}__layer{layer}__s{n_samples}"
    )

def setup_model_and_tokenizer(model_name, cache_dir, device):
    """Set up model and tokenizer."""
    logging.info(f"Setting up model {model_name} from {cache_dir}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
    config.output_hidden_states = True

    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        config=config,
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir
    ).to(device)
    model.eval()
    
    logging.info("Model and tokenizer setup complete.")
    return model, tokenizer

def setup_dataset(
    task_type, n_samples, seed, cache_dir, 
    remove_outliers, outlier_lower_bound, outlier_upper_bound,
    outlier_lower_percentile, outlier_upper_percentile,
    run=None
):
    """Load and prepare the dataset based on the task type."""
    logging.info(f"Setting up dataset for {task_type} task from {cache_dir}")
    if task_type == "classification":
        dataset = GPQADataset(split="train", config_name="gpqa_main", seed=seed, cache_dir=cache_dir)
        # GPQADataset is not a datasets.Dataset object, so we can't slice it before loading
        # We'll just rely on n_samples later in the main loop.
        return dataset
    elif task_type == "regression":
        logging.info(f"Loading NuminaMath dataset from {cache_dir}")
        full_dataset = datasets.load_dataset(
            'PrimeIntellect/NuminaMath-QwQ-CoT-5M',
            cache_dir=cache_dir,
            streaming=False,
            split='train' 
        )
        
        lower_bound, upper_bound = outlier_lower_bound, outlier_upper_bound
        if remove_outliers and (lower_bound is None or upper_bound is None):
            logging.warning(
                "No outlier bounds provided. Calculating percentiles by scanning the entire dataset. "
                "This will be very slow. Provide `outlier_lower_bound` and `outlier_upper_bound` to avoid this."
            )
            all_numbers = [
                num for example in tqdm(full_dataset, desc="Scanning dataset for outlier calculation")
                if (num := extract_final_number(example['ground_truth'])) is not None
            ]
            if not all_numbers:
                raise ValueError("No numerical data found in the dataset to calculate outlier bounds.")
            
            lower_bound = np.percentile(all_numbers, outlier_lower_percentile)
            upper_bound = np.percentile(all_numbers, outlier_upper_percentile)
            
            log_message = (
                "\n" + "="*80 + "\n"
                f"CALCULATED OUTLIER BOUNDS FOR RE-USE (percentiles: {outlier_lower_percentile}% and {outlier_upper_percentile}%):\n"
                f"  --outlier_lower_bound {lower_bound}\n"
                f"  --outlier_upper_bound {upper_bound}\n"
                "To speed up future runs, pass these arguments to the script.\n"
                + "="*80
            )
            logging.info(log_message)
            
            if run:
                run["parameters/calculated_outlier_lower_bound"] = lower_bound
                run["parameters/calculated_outlier_upper_bound"] = upper_bound
        
        logging.info(f"Using outlier bounds: [{lower_bound}, {upper_bound}]")
        
        logging.info(f"Filtering to find {n_samples} numerical examples (this might take a while)...")

        # Shuffle the dataset first to get a random sample, then iterate to find enough valid samples.
        # This is much faster than filtering the entire 5M+ dataset.
        shuffled_dataset = full_dataset.shuffle(seed=seed)

        filtered_examples = []
        with tqdm(total=n_samples, desc="Finding numerical examples") as pbar:
            for example in shuffled_dataset:
                number = extract_final_number(example['ground_truth'])
                if number is not None:
                    if remove_outliers and (number < lower_bound or number > upper_bound):
                        continue
                    
                    filtered_examples.append(example)
                    pbar.update(1)
                    if len(filtered_examples) >= n_samples:
                        break
        
        if not filtered_examples:
            raise ValueError("No numerical samples found in the dataset within the specified bounds.")

        if len(filtered_examples) < n_samples:
            logging.warning(
                f"Found only {len(filtered_examples)} numerical examples after scanning the entire dataset. "
                f"Using all found samples."
            )
        
        # Convert the list of dicts back to a Dataset object
        return datasets.Dataset.from_list(filtered_examples)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

def find_target_indices(tokenizer, full_prompt, target_text):
    """
    Find the token indices corresponding to the target text within the full prompt
    using character-level offset mapping.
    """
    target_start_char = full_prompt.find(target_text)

    if target_start_char == -1:
        error_message = (
            "\n" + "="*80 + "\n"
            "ERROR: `find_target_indices` failed. Could not find `target_text` via string matching.\n"
            "This can happen if the tokenizer's chat template adds characters (e.g., spaces) that break a simple substring search.\n"
            f"--- Target Text (len={len(target_text)}) ---\n{target_text}\n"
            f"--- Full Prompt (len={len(full_prompt)}) ---\n{full_prompt}\n"
            "="*80
        )
        raise ValueError(error_message)

    target_end_char = target_start_char + len(target_text)

    # Tokenize the full prompt to get offset mappings
    inputs = tokenizer(full_prompt, return_tensors="pt", return_offsets_mapping=True)
    offset_mapping = inputs.offset_mapping[0]

    # Find tokens that fall within the target's character range
    target_token_indices = [
        i for i, (start, end) in enumerate(offset_mapping)
        if start >= target_start_char and end <= target_end_char and start < end
    ]
    
    return target_token_indices, inputs.input_ids


def get_hidden_states_for_layer(model, tokenizer, sample, model_name, reasoning_stub, revealing_answer_prompt, target_text, layer, task_type):
    """Get the hidden states for a target text from a specific layer."""
    
    user_prompt = sample["prompt"]
    
    if task_type == "classification":
        answer = sample["answer_letter"]
    elif task_type == "regression":
        answer = sample["ground_truth"]
    else:
        raise ValueError(f"Invalid task_type: {task_type}")

    if 'QwQ' in model_name:
        assistant_content = f"{reasoning_stub} {revealing_answer_prompt} {answer}"
        messages = [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_content},
        ]
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        
    elif ('Qwen3' in model_name):
        messages = [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": ''},
        ]

        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        full_prompt = full_prompt.replace(
            '<think>\n\n</think>', f'<think>\n{reasoning_stub} {revealing_answer_prompt} {answer}'
        )

    else:
        # Fallback for other models or custom logic might be needed
        raise NotImplementedError(f"Prompt construction for model '{model_name}' is not implemented.")

    target_indices, input_ids = find_target_indices(tokenizer, full_prompt, target_text)
    
    if not target_indices:
        return None, None, None
        
    with torch.no_grad():
        outputs = model(input_ids.to(model.device))
    
    # hidden_states is a tuple, one element per layer
    # The first element is the embedding layer, so hidden_states[i] is the output of the i-th layer
    layer_hidden_states = outputs.hidden_states[layer]
    
    # (batch_size, seq_len, hidden_size) -> (seq_len, hidden_size)
    layer_hidden_states = layer_hidden_states.squeeze(0) 

    # get target hidden states
    target_hidden_states = layer_hidden_states[target_indices, :] # (num_target_tokens, hidden_size)
    
    target_tokens = tokenizer.convert_ids_to_tokens(input_ids[0, target_indices])
    
    return layer_hidden_states, target_tokens, target_indices

# =================================================================================
# 3. Training and Evaluation Loop
# =================================================================================
def main(
    # General parameters
    action: str, # "generate_data" or "train_probe"
    probe_type: str = "mlp", # "mlp" or "attention"
    task_type: str = "classification", # "classification" or "regression"
    model_name: str = "Qwen/Qwen3-8B",
    layer: int = 40,
    reasoning_stub: str = "Okay, I have finished thinking.",
    revealing_answer_prompt: str = "The final answer is",
    n_samples: int = 1000,
    seed: int = 42,
    output_dir: str = "results_fixed_prompt_token_probing",
    cache_dir: Optional[str] =  None,
    device: str = "cuda",
    # Probe parameters
    probe_hidden_dim: int = 16, # Hidden dim for MLP or attention_dim for Attention
    num_classes: int = 4, # A, B, C, D for classification
    # Training parameters
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    epochs: int = 50,
    val_split_size: float = 0.20,
    weight_decay: float = 0.01,
    neptune_project: str = "fixed-prompt-token-probing",

    # Regression-specific parameters
    normalize_targets: bool = True,
    log_transform_targets: bool = True,
    remove_outliers: bool = True,
    outlier_lower_bound: Optional[float] = None,
    outlier_upper_bound: Optional[float] = None,
    outlier_lower_percentile: float = 1.0,
    outlier_upper_percentile: float = 95.0,

    cache_dir_model: Optional[str] = None,
    cache_dir_dataset: Optional[str] = None,
):
    """Main function to run the experiment."""
  
    if cache_dir is None:
        cache_dir = "/mnt/nfs_share/tikhonov/hf_cache"

    if cache_dir_model is None:
        cache_dir_model = cache_dir
    if cache_dir_dataset is None:
        cache_dir_dataset = cache_dir
    
    data_path = get_data_path(output_dir, model_name, task_type, layer, n_samples)

    if action == "generate_data":
        run_generate_data(
            probe_type=probe_type, task_type=task_type, model_name=model_name, layer=layer,
            reasoning_stub=reasoning_stub, revealing_answer_prompt=revealing_answer_prompt,
            n_samples=n_samples, seed=seed, output_dir=output_dir, cache_dir=cache_dir,
            device=device, num_classes=num_classes, batch_size=batch_size,
            learning_rate=learning_rate, epochs=epochs, val_split_size=val_split_size,
            weight_decay=weight_decay, neptune_project=neptune_project,
            normalize_targets=normalize_targets, log_transform_targets=log_transform_targets,
            remove_outliers=remove_outliers, outlier_lower_bound=outlier_lower_bound,
            outlier_upper_bound=outlier_upper_bound, outlier_lower_percentile=outlier_lower_percentile,
            outlier_upper_percentile=outlier_upper_percentile, cache_dir_model=cache_dir_model,
            cache_dir_dataset=cache_dir_dataset, data_path=data_path
        )
    elif action == "train_probe":
        run_train_probe(
            probe_type=probe_type, task_type=task_type, model_name=model_name, layer=layer,
            reasoning_stub=reasoning_stub, revealing_answer_prompt=revealing_answer_prompt,
            n_samples=n_samples, seed=seed, output_dir=output_dir, cache_dir=cache_dir,
            device=device, probe_hidden_dim=probe_hidden_dim, num_classes=num_classes,
            batch_size=batch_size, learning_rate=learning_rate, epochs=epochs,
            val_split_size=val_split_size, weight_decay=weight_decay,
            neptune_project=neptune_project, normalize_targets=normalize_targets,
            log_transform_targets=log_transform_targets, remove_outliers=remove_outliers,
            outlier_lower_bound=outlier_lower_bound, outlier_upper_bound=outlier_upper_bound,
            outlier_lower_percentile=outlier_lower_percentile,
            outlier_upper_percentile=outlier_upper_percentile, cache_dir_model=cache_dir_model,
            cache_dir_dataset=cache_dir_dataset, data_path=data_path
        )
    else:
        raise ValueError(f"Unknown action: {action}. Must be 'generate_data' or 'train_probe'.")


def run_generate_data(**kwargs):
    """Generates and saves hidden states to disk."""
    # Unpack kwargs
    data_path = kwargs['data_path']
    model_name = kwargs['model_name']
    task_type = kwargs['task_type']
    layer = kwargs['layer']
    n_samples = kwargs['n_samples']
    seed = kwargs['seed']
    cache_dir_model = kwargs['cache_dir_model']
    cache_dir_dataset = kwargs['cache_dir_dataset']
    device = kwargs['device']
    reasoning_stub = kwargs['reasoning_stub']
    revealing_answer_prompt = kwargs['revealing_answer_prompt']
    remove_outliers = kwargs['remove_outliers']
    outlier_lower_bound = kwargs['outlier_lower_bound']
    outlier_upper_bound = kwargs['outlier_upper_bound']
    outlier_lower_percentile = kwargs['outlier_lower_percentile']
    outlier_upper_percentile = kwargs['outlier_upper_percentile']
    probe_type = kwargs['probe_type']

    logging.info(f"Action: generate_data. Data will be saved to: {data_path}")

    if os.path.exists(data_path) and os.listdir(data_path):
        logging.warning(
            f"Data directory {data_path} already exists and is not empty. "
            "Skipping data generation. If you want to regenerate, delete the directory."
        )
        return

    os.makedirs(data_path, exist_ok=True)

    # --- Setup ---
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # --- Model, Tokenizer, Dataset ---
    model, tokenizer = setup_model_and_tokenizer(model_name, cache_dir_model, device)
    dataset = setup_dataset(
        task_type, n_samples, seed, cache_dir_dataset,
        remove_outliers, outlier_lower_bound, outlier_upper_bound,
        outlier_lower_percentile, outlier_upper_percentile,
        run=None # No neptune run during generation
    )
    hidden_size = model.config.hidden_size

    # --- Generate Data ---
    logging.info(f"Generating hidden states for {n_samples} samples from layer {layer}...")
    
    target_prompt_text = f"{reasoning_stub} {revealing_answer_prompt}"
    
    # Get metadata from the first sample
    dummy_sample = dataset[0]
    _ , target_tokens, target_indices_in_sequence = get_hidden_states_for_layer(
        model, tokenizer, dummy_sample, model_name, 
        reasoning_stub, revealing_answer_prompt, 
        target_prompt_text, layer, task_type
    )
    if target_tokens is None:
        raise ValueError("Could not determine target tokens from the first sample.")

    num_tokens_to_probe = len(target_tokens)
    logging.info(f"Probing {num_tokens_to_probe} tokens: {target_tokens}")
    
    metadata = {
        "model_name": model_name, "task_type": task_type, "probe_type": probe_type,
        "layer": layer, "n_samples_requested": n_samples,
        "num_tokens_to_probe": num_tokens_to_probe, "target_tokens": target_tokens,
        "target_indices_in_sequence": target_indices_in_sequence,
        "hidden_size": hidden_size
    }

    num_samples_to_process = min(n_samples, len(dataset))
    samples_saved = 0
    
    for i in tqdm(range(num_samples_to_process), desc="Generating and saving data"):
        sample = dataset[i]
        
        full_sequence_hiddens, tokens, _ = get_hidden_states_for_layer(
            model, tokenizer, sample, model_name, 
            reasoning_stub, revealing_answer_prompt,
            target_prompt_text, layer, task_type
        )
        
        if full_sequence_hiddens is None or len(tokens) != num_tokens_to_probe:
            logging.warning(f"Skipping sample {i} due to tokenization mismatch. Expected {num_tokens_to_probe}, got {len(tokens) if tokens else 'None'}.")
            continue
        
        if probe_type == "mlp":
            hiddens = full_sequence_hiddens[target_indices_in_sequence, :]
        elif probe_type == "attention":
            hiddens = full_sequence_hiddens[:target_indices_in_sequence[-1] + 1, :]
        else:
            raise ValueError(f"Unknown probe_type: {probe_type}")

        if task_type == "classification":
            label_val = ord(sample['answer_letter']) - ord('A')
            label = torch.tensor(label_val, dtype=torch.long)
        elif task_type == "regression":
            label_val = extract_final_number(sample['ground_truth'])
            if label_val is None:
                logging.warning(f"Could not extract number for sample {i}. Skipping.")
                continue
            label = torch.tensor(label_val, dtype=torch.bfloat16) # Store with precision
        
        # Move to CPU and save individually
        hiddens_cpu = hiddens.to('cpu')
        torch.save(hiddens_cpu, os.path.join(data_path, f"sample_{samples_saved}.pt"))
        torch.save(label, os.path.join(data_path, f"label_{samples_saved}.pt"))
        samples_saved += 1
    
    metadata["n_samples_saved"] = samples_saved
    with open(os.path.join(data_path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    logging.info(f"Finished generating data. Saved {samples_saved} samples to {data_path}")
    # Free up VRAM
    del model
    del tokenizer
    torch.cuda.empty_cache()


def run_train_probe(**kwargs):
    """Loads precomputed data and runs the training loop."""
    # Unpack kwargs
    data_path = kwargs['data_path']
    probe_type = kwargs['probe_type']
    task_type = kwargs['task_type']
    model_name = kwargs['model_name']
    layer = kwargs['layer']
    n_samples = kwargs['n_samples']
    seed = kwargs['seed']
    output_dir = kwargs['output_dir']
    device = kwargs['device']
    probe_hidden_dim = kwargs['probe_hidden_dim']
    num_classes = kwargs['num_classes']
    batch_size = kwargs['batch_size']
    learning_rate = kwargs['learning_rate']
    epochs = kwargs['epochs']
    val_split_size = kwargs['val_split_size']
    weight_decay = kwargs['weight_decay']
    neptune_project = kwargs['neptune_project']
    normalize_targets = kwargs['normalize_targets']
    log_transform_targets = kwargs['log_transform_targets']
    
    logging.info(f"Action: train_probe. Loading data from: {data_path}")

    if not os.path.exists(data_path) or not os.listdir(data_path):
        raise FileNotFoundError(
            f"Data directory {data_path} is empty or does not exist. "
            f"Please run the 'generate_data' action first for these parameters."
        )
    
    with open(os.path.join(data_path, "metadata.json"), "r") as f:
        metadata = json.load(f)

    # Extract metadata needed for training
    hidden_size = metadata['hidden_size']
    num_tokens_to_probe = metadata['num_tokens_to_probe']
    target_tokens = metadata['target_tokens']
    target_indices_in_sequence = metadata['target_indices_in_sequence']
    
    # --- Setup ---
    run_timestamp = f"{model_name.replace('/', '_')}_layer{layer}_{n_samples}s_{probe_type}"
    run_output_dir = os.path.join(output_dir, run_timestamp)
    os.makedirs(run_output_dir, exist_ok=True)
    logging.info(f"Results will be saved to: {run_output_dir}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # --- Neptune ---
    run = neptune.init_run(
        project=neptune_project,
        tags=[f"task_{task_type}", f"probe_{probe_type}", model_name, f"layer_{layer}"]
    )
    # Log all kwargs passed to the function
    params = {k: v for k, v in kwargs.items() if v is not None and isinstance(v, (str, int, float, bool, list, dict))}
    run["parameters"] = params
    
    save_dir = os.path.join(f"saved_probes_{task_type}", run_timestamp, run["sys/id"].fetch())
    os.makedirs(save_dir, exist_ok=True)
    logging.info(f"Probe weights will be saved to: {save_dir}")

    # --- Model and Data ---
    model_dtype = torch.bfloat16 if "cuda" in device else torch.float32

    # --- Create Probes and Datasets ---
    if task_type == "classification":
        if probe_type == "mlp":
            probes = [
                ProbeMlpClassifier(hidden_size, probe_hidden_dim, num_classes, device, model_dtype)
                for _ in range(num_tokens_to_probe)
            ]
        elif probe_type == "attention":
            probes = [
                ProbeAttentionClassifier(hidden_size, probe_hidden_dim, num_classes, device, model_dtype)
                for _ in range(num_tokens_to_probe)
            ]
        criterion = nn.CrossEntropyLoss()
    elif task_type == "regression":
        if probe_type == "mlp":
            probes = [
                ProbeMlpRegressor(hidden_size, probe_hidden_dim, device, model_dtype)
                for _ in range(num_tokens_to_probe)
            ]
        elif probe_type == "attention":
            probes = [
                ProbeAttentionRegressor(hidden_size, probe_hidden_dim, device, model_dtype)
                for _ in range(num_tokens_to_probe)
            ]
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Invalid task_type: {task_type}")

    optimizers = [
        torch.optim.AdamW(probe.parameters(), lr=learning_rate, weight_decay=weight_decay)
        for probe in probes
    ]
    
    # Create dataset from precomputed files
    full_dataset = PrecomputedHiddenStateDataset(data_path)
    
    # Split dataset
    val_size = int(len(full_dataset) * val_split_size)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # --- Normalize Regression Targets ---
    scaler = None
    log_shift = None
    if task_type == "regression" and normalize_targets:
        logging.info("Normalizing regression targets...")
        
        # To get all labels, we need to iterate through the indices
        train_labels_list = [full_dataset[i][1].item() for i in train_dataset.indices]
        val_labels_list = [full_dataset[i][1].item() for i in val_dataset.indices]
        train_labels_np = np.array(train_labels_list).reshape(-1, 1)
        val_labels_np = np.array(val_labels_list).reshape(-1, 1)

        # Optional: Log transform
        if log_transform_targets:
            min_val = train_labels_np.min()
            log_shift = 1 - min_val if min_val <= 0 else 0
            logging.info(f"Applying log transform with shift C={log_shift:.4f}")
            run["parameters/log_transform_shift"] = log_shift
            
            train_labels_np = np.log(train_labels_np + log_shift)
            val_labels_np = np.log(val_labels_np + log_shift)

        scaler = StandardScaler()
        train_labels_scaled = scaler.fit_transform(train_labels_np)
        val_labels_scaled = scaler.transform(val_labels_np)
        
        # We can't easily create a new TensorDataset, so we'll store scaled labels
        # and apply them inside the training loop. A bit less clean but avoids
        # loading all hidden states into memory. We'll create a mapping from original index.
        scaled_train_labels = {idx: torch.tensor(val, dtype=model_dtype) for idx, val in zip(train_dataset.indices, train_labels_scaled)}
        scaled_val_labels = {idx: torch.tensor(val, dtype=model_dtype) for idx, val in zip(val_dataset.indices, val_labels_scaled)}

        logging.info(f"Targets scaled. Mean: {scaler.mean_[0]:.4f}, Scale: {scaler.scale_[0]:.4f}")
        run["parameters/normalization_mean"] = scaler.mean_[0]
        run["parameters/normalization_scale"] = scaler.scale_[0]

    # Use a collate function only if needed (for attention probe)
    current_collate_fn = collate_fn_pad if probe_type == 'attention' else None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=current_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=current_collate_fn)

    # --- Training Loop ---
    logging.info("Starting training...")
    best_avg_val_metric = float('inf') if task_type == "regression" else 0.0
    
    for epoch in range(epochs):
        # --- Training ---
        for probe in probes:
            probe.train()
        
        # The dataloader gives us original indices if we wrap it.
        # But we can get them from the dataset subset. This is complex.
        # Let's simplify the normalization logic.
        # Let's create new datasets for normalization, accepting the memory hit for labels.
        train_hiddens = [full_dataset[i][0] for i in train_dataset.indices]
        val_hiddens = [full_dataset[i][0] for i in val_dataset.indices]
        
        # This will be slow. Let's rethink the normalization.
        # The previous approach of a mapping is better. Let's refine it.
        # The custom dataset should return the index.
        # Let's modify PrecomputedHiddenStateDataset to return index.
        # No, that complicates the dataloader. The mapping is fine, just need to use it.
        # A simpler way: create a new dataset that WRAPS the original and applies the transform.
        class ScaledDataset(Dataset):
            def __init__(self, subset, scaled_labels_map, model_dtype):
                self.subset = subset
                self.scaled_labels_map = scaled_labels_map
                self.model_dtype = model_dtype
                self.indices = self.subset.indices

            def __len__(self):
                return len(self.subset)

            def __getitem__(self, idx):
                # The idx here is relative to the subset, so we map it back
                original_idx = self.indices[idx]
                # Access the subset directly, which will call the underlying dataset's getitem
                hiddens, _ = self.subset[idx]
                scaled_label = self.scaled_labels_map[original_idx]
                return hiddens, scaled_label

        if task_type == "regression" and normalize_targets:
             train_dataset = ScaledDataset(train_dataset, scaled_train_labels, model_dtype)
             val_dataset = ScaledDataset(val_dataset, scaled_val_labels, model_dtype)

        # Re-create loaders if datasets were replaced
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=current_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=current_collate_fn)

        for batch_hiddens, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            batch_hiddens, batch_labels = batch_hiddens.to(device), batch_labels.to(device)
            
            for token_idx in range(num_tokens_to_probe):
                optimizers[token_idx].zero_grad()
                
                if probe_type == "mlp":
                    # For mlp, hiddens are already (batch, num_tokens, hidden)
                    # We need to load them as such. Let's adjust the collate func.
                    # The loader now returns (batch, seq, hidden) for attention
                    # or a list of (num_tokens, hidden) for mlp.
                    # The collate_fn needs to handle both.
                    token_hiddens = batch_hiddens[:, token_idx, :]
                elif probe_type == "attention":
                    current_token_absolute_index = target_indices_in_sequence[token_idx]
                    # Ensure we don't slice beyond the padded length
                    max_len_in_batch = batch_hiddens.size(1)
                    slice_end = min(current_token_absolute_index + 1, max_len_in_batch)
                    token_hiddens = batch_hiddens[:, :slice_end, :]
                
                if task_type == "classification":
                    logits = probes[token_idx](token_hiddens)
                    loss = criterion(logits, batch_labels)
                elif task_type == "regression":
                    preds = probes[token_idx](token_hiddens)
                    loss = criterion(preds, batch_labels.view(-1, 1))
                
                loss.backward()
                optimizers[token_idx].step()
                
                run[f"train/token_{token_idx}/loss"].log(loss.item())

        # --- Validation ---
        for probe in probes:
            probe.eval()
            
        if task_type == "classification":
            total_correct = [0] * num_tokens_to_probe
        else: # regression
            total_mse = [0.0] * num_tokens_to_probe

        total_samples = 0
        
        with torch.no_grad():
            for batch_hiddens, batch_labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                batch_hiddens, batch_labels = batch_hiddens.to(device), batch_labels.to(device)
                total_samples += batch_hiddens.size(0)

                for token_idx in range(num_tokens_to_probe):
                    if probe_type == "mlp":
                        token_hiddens = batch_hiddens[:, token_idx, :]
                    elif probe_type == "attention":
                        current_token_absolute_index = target_indices_in_sequence[token_idx]
                        max_len_in_batch = batch_hiddens.size(1)
                        slice_end = min(current_token_absolute_index + 1, max_len_in_batch)
                        token_hiddens = batch_hiddens[:, :slice_end, :]
                    
                    if task_type == "classification":
                        logits = probes[token_idx](token_hiddens)
                        preds = torch.argmax(logits, dim=1)
                        total_correct[token_idx] += (preds == batch_labels).sum().item()
                    elif task_type == "regression":
                        preds = probes[token_idx](token_hiddens)
                        
                        # For logging, calculate MSE on the original, un-normalized scale
                        # Cast to float32 before converting to numpy, as numpy doesn't support bfloat16
                        preds_denorm = preds.cpu().detach().to(torch.float32).numpy()
                        labels_denorm = batch_labels.cpu().detach().to(torch.float32).numpy().reshape(-1, 1)
                        
                        if scaler:
                            preds_denorm = scaler.inverse_transform(preds_denorm)
                            labels_denorm = scaler.inverse_transform(labels_denorm)

                        if log_transform_targets:
                            preds_denorm = np.exp(preds_denorm) - log_shift
                            labels_denorm = np.exp(labels_denorm) - log_shift
                        
                        mse = F.mse_loss(torch.tensor(preds_denorm), torch.tensor(labels_denorm), reduction='sum')

                        total_mse[token_idx] += mse.item()

        if task_type == "classification":
            val_accuracies = [corr / total_samples for corr in total_correct]
            avg_val_metric = np.mean(val_accuracies)
            metric_name = "Accuracy"
            logging.info(f"Epoch {epoch+1}: Avg Val Acc: {avg_val_metric:.4f}")
            run["val/avg_accuracy"].log(avg_val_metric, step=epoch+1)
            for token_idx, acc in enumerate(val_accuracies):
                run[f"val/token_{token_idx}/accuracy"].log(acc, step=epoch+1)
                logging.info(f"  - Token '{target_tokens[token_idx]}' ({token_idx}): {acc:.4f}")
            
            is_better = avg_val_metric > best_avg_val_metric
        
        elif task_type == "regression":
            val_mses = [mse / total_samples for mse in total_mse]
            val_rmses = [np.sqrt(mse) for mse in val_mses]
            avg_val_metric = np.mean(val_mses) # We still use MSE to determine the best model
            avg_val_rmse = np.mean(val_rmses)
            metric_name = "MSE"
            
            logging.info(f"Epoch {epoch+1}: Avg Val MSE: {avg_val_metric:.4f}, Avg Val RMSE: {avg_val_rmse:.4f}")
            run["val/avg_mse"].log(avg_val_metric, step=epoch+1)
            run["val/avg_rmse"].log(avg_val_rmse, step=epoch+1)

            for token_idx, (mse, rmse) in enumerate(zip(val_mses, val_rmses)):
                run[f"val/token_{token_idx}/mse"].log(mse, step=epoch+1)
                run[f"val/token_{token_idx}/rmse"].log(rmse, step=epoch+1)
                logging.info(f"  - Token '{target_tokens[token_idx]}' ({token_idx}): MSE={mse:.4f}, RMSE={rmse:.4f}")
            
            is_better = avg_val_metric < best_avg_val_metric


        if is_better:
            best_avg_val_metric = avg_val_metric
            logging.info(f"New best average validation {metric_name.lower()}: {best_avg_val_metric:.4f}. Saving probes to {save_dir}.")
            for token_idx, probe in enumerate(probes):
                save_path = os.path.join(save_dir, f"best_probe_token_{token_idx}.pt")
                torch.save(probe.state_dict(), save_path)
    
    # --- Final saving and plotting ---
    for token_idx, probe in enumerate(probes):
        final_save_path = os.path.join(save_dir, f"final_probe_token_{token_idx}.pt")
        torch.save(probe.state_dict(), final_save_path)
    
    # Plot final metrics
    if task_type == "classification":
        final_metrics = val_accuracies
    else: # regression
        final_metrics = val_rmses # Plot RMSE for better interpretability
        metric_name = "RMSE"

    fig, ax = plt.subplots(figsize=(max(10, num_tokens_to_probe), 6))
    sns.barplot(x=target_tokens, y=final_metrics, ax=ax)
    ax.set_xlabel("Token")
    ax.set_ylabel(f"Final Validation {metric_name}")
    ax.set_title(f"Probe {metric_name} per Token - Layer {layer}")
    if task_type == "classification":
        ax.set_ylim(0, 1.0)
    for i, metric in enumerate(final_metrics):
        ax.text(i, metric + 0.01, f"{metric:.3f}", ha='center')
    plt.tight_layout()
    run[f"val/final_{metric_name.lower()}_plot"].upload(File.as_image(fig))
    plt.close(fig)

    run.stop()
    logging.info("Training finished.")


if __name__ == "__main__":
    fire.Fire(main)
