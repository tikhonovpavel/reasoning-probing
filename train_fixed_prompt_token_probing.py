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

# =================================================================================
# 2. Setup and Data Extraction
# =================================================================================

def setup(model_name, seed, cache_dir, device):
    """Set up model, tokenizer, and dataset."""
    logging.info("Setting up model, tokenizer, and dataset...")
    
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

    dataset = GPQADataset(split="train", config_name="gpqa_main", seed=seed, cache_dir=cache_dir)
    
    logging.info("Setup complete.")
    return model, tokenizer, dataset

def find_target_indices(tokenizer, full_prompt, target_text):
    """Find the token indices corresponding to the target text within the full prompt."""
    
    # Find the character start and end positions of the target text in the full prompt
    target_start_char = full_prompt.find(target_text)
    if target_start_char == -1:
        # Fallback for when tokenization adds prefixes, making exact string match fail
        # This is a bit of a heuristic.
        encoded_target = tokenizer.encode(target_text, add_special_tokens=False)
        encoded_full = tokenizer.encode(full_prompt, add_special_tokens=False)
        
        # Naive subsequence search
        for i in range(len(encoded_full) - len(encoded_target) + 1):
            if encoded_full[i:i+len(encoded_target)] == encoded_target:
                return list(range(i, i + len(encoded_target))), torch.tensor([encoded_full], dtype=torch.long)
        
        error_message = (
            "\n" + "="*80 + "\n"
            "ERROR: `find_target_indices` failed. Could not find `target_text` via string matching or token matching.\n"
            f"--- Target Text (len={len(target_text)}) ---\n{target_text}\n"
            f"--- Full Prompt (len={len(full_prompt)}) ---\n{full_prompt}\n"
            "--- Tokenization Debug ---\n"
            f"Target tokens: {tokenizer.convert_ids_to_tokens(encoded_target)}\n"
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


def get_hidden_states_for_layer(model, tokenizer, sample, model_name, reasoning_stub, revealing_answer_prompt, target_text, layer):
    """Get the hidden states for a target text from a specific layer."""
    
    user_prompt = sample["prompt"]
    answer_letter = sample["answer_letter"]
    
    if 'QwQ' in model_name:
        assistant_content = f"{reasoning_stub} {revealing_answer_prompt} {answer_letter}"
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
            '<think>\n\n</think>', f'<think>\n{reasoning_stub} {revealing_answer_prompt} {answer_letter}'
        )

    else:
        # Fallback for other models or custom logic might be needed
        raise NotImplementedError(f"Prompt construction for model '{model_name}' is not implemented.")

    target_indices, input_ids = find_target_indices(tokenizer, full_prompt, target_text)
    
    if not target_indices:
        return None, None
        
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
    
    return target_hidden_states, target_tokens

# =================================================================================
# 3. Training and Evaluation Loop
# =================================================================================
def main(
    model_name: str = "Qwen/Qwen3-8B",
    layer: int = 40,
    reasoning_stub: str = "Okay, I have finished thinking.",
    revealing_answer_prompt: str = "The final answer is",
    n_samples: int = 1000,
    seed: int = 42,
    output_dir: str = "results_fixed_prompt_token_probing",
    cache_dir: str =  '/home/tikhonov/.cache/huggingface/hub/', #"/mnt/nfs_share/tikhonov/hf_cache",
    device: str = "cuda",
    # Probe parameters
    mlp_hidden_dim: int = 16,
    num_classes: int = 4, # A, B, C, D
    # Training parameters
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    epochs: int = 50,
    val_split_size: float = 0.20,
    weight_decay: float = 0.01,
    neptune_project: str = "fixed-prompt-token-probing",
):
    """Main function to run the experiment."""
    # --- Setup ---
    run_timestamp = f"{model_name.split('/')[-1]}_layer{layer}_{n_samples}s"
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
        tags=["train_classification", model_name, f"layer_{layer}"]
    )
    params = {k: v for k, v in locals().items() if isinstance(v, (str, int, float, bool))}
    run["parameters"] = params
    
    save_dir = os.path.join("saved_probes_classification", run_timestamp, run["sys/id"].fetch())
    os.makedirs(save_dir, exist_ok=True)
    logging.info(f"Probe weights will be saved to: {save_dir}")

    # --- Model, Tokenizer, Dataset ---
    model, tokenizer, dataset = setup(model_name, seed, cache_dir, device)
    model_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    hidden_size = model.config.hidden_size

    # --- Sanity Check: Assert tokenization consistency ---
    logging.info("Running tokenization sanity check...")
    base_prompt_text = f"{reasoning_stub} {revealing_answer_prompt}"
    prompts_to_check = [f"{base_prompt_text} {letter}" for letter in ["A", "B", "C", "D"]]
    tokenized_prompts = [tokenizer.encode(p) for p in prompts_to_check]

    base_length = len(tokenized_prompts[0])
    for i, tokens in enumerate(tokenized_prompts):
        assert len(tokens) == base_length, \
            f"Tokenization length mismatch! Prompt A has {base_length} tokens, but prompt {chr(65+i)} has {len(tokens)} tokens."
        # Check that the shared part is identical
        assert tokenized_prompts[0][:-1] == tokens[:-1], \
            f"Shared prompt part tokenization mismatch between prompt A and prompt {chr(65+i)}."
            
    logging.info("Tokenization sanity check passed. Assistant prompts are consistent.")

    # --- Generate Data ---
    logging.info(f"Generating hidden states for {n_samples} samples from layer {layer}...")
    
    target_prompt_text = f"{reasoning_stub} {revealing_answer_prompt}"
    
    all_hiddens = []
    all_labels = []
    
    # This is to get the tokenization of the target prompt once
    _ , target_tokens = get_hidden_states_for_layer(
        model, tokenizer, dataset[0], model_name, 
        reasoning_stub, revealing_answer_prompt, 
        target_prompt_text,
        layer
    )
    num_tokens_to_probe = len(target_tokens)
    logging.info(f"Probing {num_tokens_to_probe} tokens: {target_tokens}")
    run["parameters/target_tokens"] = ", ".join(target_tokens)

    num_samples_to_process = min(n_samples, len(dataset))
    if num_samples_to_process < n_samples:
        logging.warning(f"Requested {n_samples} samples, but dataset only has {len(dataset)}. Processing {num_samples_to_process} samples.")

    for i in tqdm(range(num_samples_to_process), desc="Generating data"):
        sample = dataset[i]
        # We need to add the answer letter to the prompt to get the hidden states in context
        
        hiddens, tokens = get_hidden_states_for_layer(
            model, tokenizer, sample, model_name, 
            reasoning_stub, revealing_answer_prompt,
            target_prompt_text, layer
        )
        
        if hiddens is None or len(tokens) != num_tokens_to_probe:
            logging.warning(f"Skipping sample {i} due to tokenization mismatch or error. Expected {num_tokens_to_probe} tokens, got {len(tokens) if tokens else 'None'}.")
            continue
            
        all_hiddens.append(hiddens)
        # Convert 'A', 'B', 'C', 'D' to 0, 1, 2, 3
        label = ord(sample['answer_letter']) - ord('A')
        all_labels.append(label)

    if not all_hiddens:
        logging.error("No data was generated. Exiting.")
        run.stop()
        return

    # Stack all hidden states: (n_samples, num_tokens, hidden_size)
    all_hiddens_tensor = torch.stack(all_hiddens, dim=0).to(dtype=model_dtype)
    all_labels_tensor = torch.tensor(all_labels, dtype=torch.long)

    # --- Create Probes and Datasets ---
    probes = [
        ProbeMlpClassifier(hidden_size, mlp_hidden_dim, num_classes, device, model_dtype)
        for _ in range(num_tokens_to_probe)
    ]
    optimizers = [
        torch.optim.AdamW(probe.parameters(), lr=learning_rate, weight_decay=weight_decay)
        for probe in probes
    ]
    criterion = nn.CrossEntropyLoss()

    # Create one dataset for all probes, we'll slice it during training
    full_dataset = TensorDataset(all_hiddens_tensor, all_labels_tensor)
    
    # Split dataset
    val_size = int(len(full_dataset) * val_split_size)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # --- Training Loop ---
    logging.info("Starting training...")
    best_avg_val_acc = 0.0
    
    for epoch in range(epochs):
        # --- Training ---
        for probe in probes:
            probe.train()
        
        for batch_hiddens, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            batch_hiddens, batch_labels = batch_hiddens.to(device), batch_labels.to(device)
            
            for token_idx in range(num_tokens_to_probe):
                optimizers[token_idx].zero_grad()
                
                # Get hidden states for the current token: (batch_size, hidden_size)
                token_hiddens = batch_hiddens[:, token_idx, :]
                
                logits = probes[token_idx](token_hiddens)
                loss = criterion(logits, batch_labels)
                
                loss.backward()
                optimizers[token_idx].step()
                
                run[f"train/token_{token_idx}/loss"].log(loss.item())

        # --- Validation ---
        for probe in probes:
            probe.eval()
            
        total_correct = [0] * num_tokens_to_probe
        total_samples = 0
        
        with torch.no_grad():
            for batch_hiddens, batch_labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                batch_hiddens, batch_labels = batch_hiddens.to(device), batch_labels.to(device)
                total_samples += batch_hiddens.size(0)

                for token_idx in range(num_tokens_to_probe):
                    token_hiddens = batch_hiddens[:, token_idx, :]
                    logits = probes[token_idx](token_hiddens)
                    preds = torch.argmax(logits, dim=1)
                    total_correct[token_idx] += (preds == batch_labels).sum().item()

        val_accuracies = [corr / total_samples for corr in total_correct]
        avg_val_acc = np.mean(val_accuracies)
        
        logging.info(f"Epoch {epoch+1}: Avg Val Acc: {avg_val_acc:.4f}")
        run["val/avg_accuracy"].log(avg_val_acc, step=epoch+1)
        for token_idx, acc in enumerate(val_accuracies):
            run[f"val/token_{token_idx}/accuracy"].log(acc, step=epoch+1)
            logging.info(f"  - Token '{target_tokens[token_idx]}' ({token_idx}): {acc:.4f}")

        if avg_val_acc > best_avg_val_acc:
            best_avg_val_acc = avg_val_acc
            logging.info(f"New best average validation accuracy: {best_avg_val_acc:.4f}. Saving probes.")
            for token_idx, probe in enumerate(probes):
                save_path = os.path.join(save_dir, f"best_probe_token_{token_idx}.pt")
                torch.save(probe.state_dict(), save_path)
    
    # --- Final saving and plotting ---
    for token_idx, probe in enumerate(probes):
        final_save_path = os.path.join(save_dir, f"final_probe_token_{token_idx}.pt")
        torch.save(probe.state_dict(), final_save_path)
    
    # Plot final accuracies
    fig, ax = plt.subplots(figsize=(max(10, num_tokens_to_probe), 6))
    sns.barplot(x=target_tokens, y=val_accuracies, ax=ax)
    ax.set_xlabel("Token")
    ax.set_ylabel("Final Validation Accuracy")
    ax.set_title(f"Probe Accuracy per Token - Layer {layer}")
    ax.set_ylim(0, 1.0)
    for i, acc in enumerate(val_accuracies):
        ax.text(i, acc + 0.01, f"{acc:.3f}", ha='center')
    plt.tight_layout()
    run["val/final_accuracy_plot"].upload(File.as_image(fig))
    plt.close(fig)

    run.stop()
    logging.info("Training finished.")


if __name__ == "__main__":
    fire.Fire(main)
