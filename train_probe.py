import sqlite3
import os
import json
import torch
import fire
import neptune
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, Dataset, Sampler
from collections import defaultdict
import random
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from neptune.types import File

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ReasoningTracesDataset(Dataset):
    """Dataset for loading reasoning traces from the SQLite database."""
    def __init__(self, db_path: str, tokenizer: AutoTokenizer, model_name: str, split: str = 'train', val_split_size: float = 0.1):
        self.tokenizer = tokenizer
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

        self.cursor.execute("SELECT DISTINCT question_id FROM reasoning_traces_qpqa WHERE model_path=?", (model_name,))
        all_ids = [row[0] for row in self.cursor.fetchall()]
        
        # Deterministic split
        random.Random(42).shuffle(all_ids)
        split_idx = int(len(all_ids) * (1 - val_split_size))
        
        if split == 'train':
            self.question_ids = all_ids[:split_idx]
        else:
            self.question_ids = all_ids[split_idx:]
            
        logging.info(f"Loaded {len(self.question_ids)} samples for {split} split.")

        self.data = []
        for q_id in tqdm(self.question_ids, desc=f"Loading {split} data"):
            self.cursor.execute(
                "SELECT question_id, question_text, choices, correct_answer_letter, full_prompt_text FROM reasoning_traces_qpqa WHERE question_id=?", (q_id,)
            )
            row = self.cursor.fetchone()
            if row:
                token_ids = tokenizer.encode(row[4])
                if len(token_ids) > 12_500:
                    continue
                
                self.data.append({
                    "question_id": row[0],
                    "question_text": row[1],
                    "choices": json.loads(row[2]),
                    "correct_answer_letter": row[3],
                    "token_ids": tokenizer.encode(row[4]),
                })
        
        self.answer_to_indices = defaultdict(list)
        for i, item in enumerate(self.data):
            self.answer_to_indices[item['correct_answer_letter']].append(i)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        correct_answer_token_id = self.tokenizer.encode(item['correct_answer_letter'], add_special_tokens=False)[0]
        
        return {
            "token_ids": torch.tensor(item['token_ids'], dtype=torch.long),
            "target_token_id": correct_answer_token_id,
            "correct_answer_letter": item['correct_answer_letter'],
        }
        
class BalancedSampler(Sampler):
    """
    Samples elements to ensure each batch contains a diverse set of answer letters.
    """
    def __init__(self, dataset: ReasoningTracesDataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_samples = len(self.dataset)
        self.answer_to_indices = dataset.answer_to_indices
        self.labels = list(self.answer_to_indices.keys())
    
    def __iter__(self):
        # Create a copy to modify and shuffle indices for randomness
        available_indices = {label: random.sample(idx_list, len(idx_list)) for label, idx_list in self.answer_to_indices.items()}
        
        num_batches = len(self)
        for _ in range(num_batches):
            batch_indices = []
            
            # Ensure we have enough samples for a full batch
            if sum(len(v) for v in available_indices.values()) < self.batch_size:
                break
                
            for i in range(self.batch_size):
                # Cycle through labels to create a balanced batch
                label = self.labels[i % len(self.labels)]
                
                # If a label is exhausted, pick from another non-empty one
                if not available_indices[label]:
                    non_empty_labels = [l for l, idxs in available_indices.items() if idxs]
                    if not non_empty_labels:
                        break  # Should not happen due to the check above
                    label = random.choice(non_empty_labels)
                
                chosen_idx = available_indices[label].pop()
                batch_indices.append(chosen_idx)
            
            if len(batch_indices) == self.batch_size:
                yield batch_indices

    def __len__(self):
        # For a batch sampler, __len__ should return the number of batches.
        return self.num_samples // self.batch_size


class ProbeClassifier(nn.Module):
    """
    A probe classifier that operates on hidden states.
    It learns a linear transformation (A, b) and uses a frozen unembedding
    matrix (W_u) to project the result into the vocabulary space.
    
    The operation is: logits = W_u @ LayerNorm(A @ h_t + b)
    """
    def __init__(self, hidden_size: int, device: str, dtype: torch.dtype):
        super().__init__()
        self.hidden_size = hidden_size
        self.device = device

        # A and b are the learnable parameters
        self.transformer = nn.Linear(hidden_size, hidden_size).to(device=device, dtype=dtype)
        self.layer_norm = nn.LayerNorm(hidden_size).to(device=device, dtype=dtype)

    def forward(self, hidden_states: torch.Tensor, unembedding_matrix: torch.Tensor):
        """
        Args:
            hidden_states (torch.Tensor): A tensor of shape (batch_size, hidden_size).
            unembedding_matrix (torch.Tensor): The frozen unembedding matrix W_u of shape (vocab_size, hidden_size).

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, vocab_size).
        """
        # (batch_size, hidden_size) -> (batch_size, hidden_size)
        transformed_states = self.transformer(hidden_states)
        
        # (batch_size, hidden_size) -> (batch_size, hidden_size)
        norm_states = self.layer_norm(transformed_states)

        # (batch_size, hidden_size) @ (hidden_size, vocab_size) -> (batch_size, vocab_size)
        # Note: W_u is (vocab_size, hidden_size), so we transpose it.
        logits = torch.matmul(norm_states, unembedding_matrix.t())
        
        return logits

def validate_and_log(probe, dataloader, model, hidden_states_container, think_token_id, layer_index, device, unembedding_matrix, answer_token_ids, answer_map, run, epoch):
    probe.eval()
    all_targets = []
    all_preds = []
    total_val_loss = 0
    
    # For confidence dynamics plot
    # Bins for relative token position
    num_bins = 20 
    prob_correct_binned = defaultdict(list)
    prob_incorrect_binned = defaultdict(list)
    prob_other_binned = defaultdict(list)
    

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running validation"):
            for sample in batch:
                input_ids = sample["token_ids"].to(device)
                target_token_id = sample["target_token_id"]
                target_letter = sample["correct_answer_letter"]
                
                hidden_states_container.clear()
                model(input_ids.unsqueeze(0))
                
                seq_hiddens = hidden_states_container[0].squeeze(0)
                
                think_token_positions = (input_ids == think_token_id).nonzero(as_tuple=True)[0]
                if think_token_positions.numel() > 0:
                    start_pos = think_token_positions[0]
                    relevant_hiddens = seq_hiddens[start_pos:]
                    num_relevant_tokens = len(relevant_hiddens)

                    if num_relevant_tokens == 0:
                        continue
                        
                    logits = probe(relevant_hiddens.to(device), unembedding_matrix)
                    
                    # Calculate probabilities for confidence dynamics
                    probs = torch.softmax(logits, dim=-1) # (seq_len, vocab_size)
                    
                    p_a, p_b, p_c, p_d = [probs[:, token_id] for token_id in answer_token_ids]
                    
                    p_correct = probs[:, target_token_id]
                    p_all_answers = p_a + p_b + p_c + p_d
                    p_incorrect = p_all_answers - p_correct
                    p_other = 1.0 - p_all_answers

                    for i in range(num_relevant_tokens):
                        relative_position = i / num_relevant_tokens
                        bin_index = int(relative_position * num_bins)
                        
                        prob_correct_binned[bin_index].append(p_correct[i].item())
                        prob_incorrect_binned[bin_index].append(p_incorrect[i].item())
                        prob_other_binned[bin_index].append(p_other[i].item())

                    # For loss and accuracy, we can just use the last token's prediction
                    last_token_logits = logits[-1, answer_token_ids]
                    target_idx = torch.tensor([answer_map[target_letter]], device=device)
                    
                    loss = nn.CrossEntropyLoss()(last_token_logits.unsqueeze(0), target_idx)
                    total_val_loss += loss.item()
                    
                    pred_idx = torch.argmax(last_token_logits)
                    all_preds.append(pred_idx.item())
                    all_targets.append(target_idx.item())

    # Log metrics to Neptune
    avg_val_loss = total_val_loss / len(dataloader.dataset)
    accuracy = np.mean([p == t for p, t in zip(all_preds, all_targets)])

    run["val/loss"].log(avg_val_loss, step=epoch)
    run["val/accuracy"].log(accuracy, step=epoch)

    # Plot and log confidence dynamics
    fig_dyn, ax_dyn = plt.subplots(figsize=(10, 6))
    
    bins = range(num_bins)
    x_axis = [i/num_bins for i in bins]
    
    mean_p_correct = [np.mean(prob_correct_binned[i]) if prob_correct_binned[i] else np.nan for i in bins]
    mean_p_incorrect = [np.mean(prob_incorrect_binned[i]) if prob_incorrect_binned[i] else np.nan for i in bins]
    mean_p_other = [np.mean(prob_other_binned[i]) if prob_other_binned[i] else np.nan for i in bins]

    ax_dyn.plot(x_axis, mean_p_correct, label='P(correct)', marker='o')
    ax_dyn.plot(x_axis, mean_p_incorrect, label='P(incorrect)', marker='o')
    ax_dyn.plot(x_axis, mean_p_other, label='P(other)', marker='o')
    
    ax_dyn.set_xlabel("Relative Position in Reasoning Trace")
    ax_dyn.set_ylabel("Average Probability")
    ax_dyn.set_title(f"Confidence Dynamics - Epoch {epoch}")
    ax_dyn.legend()
    ax_dyn.grid(True)
    plt.tight_layout()
    
    run["val/confidence_dynamics"].log(File.as_image(fig_dyn), step=epoch)
    plt.close(fig_dyn)

    # Confusion Matrix
    cm = np.zeros((4, 4))
    for true, pred in zip(all_targets, all_preds):
        cm[true, pred] += 1
    
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt=".0f", cmap='Blues', xticklabels=answer_map.keys(), yticklabels=answer_map.keys())
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - Epoch {epoch}")
    run["val/confusion_matrix"].upload(File.as_image(fig), step=epoch)
    plt.close(fig)


def main(
    model_name: str = "Qwen/Qwen3-32B", #"Qwen/Qwen3-8B",
    db_path: str = "reasoning_traces.sqlite",
    layer_index: int = 16,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    epochs: int = 50,
    neptune_project: str = "probing-reasoning-classifier",
):
    """
    Trains a probe classifier on the hidden states of a language model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # Initialize Neptune
    logging.info(f"Initializing Neptune for project: {neptune_project}")
    run = neptune.init_run(
        project=neptune_project,
    )

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Setup datasets and dataloaders
    train_dataset = ReasoningTracesDataset(db_path, tokenizer, model_name, split='train')
    val_dataset = ReasoningTracesDataset(db_path, tokenizer, model_name, split='val')

    train_sampler = BalancedSampler(train_dataset, batch_size)

    train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=lambda x: x)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=lambda x: x)


    # 1. Model Loading
    logging.info(f"Loading base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
    ).to(device)
    model.eval() # We are not training the base model
    
    # Freeze the base model
    for param in model.parameters():
        param.requires_grad = False
        
    unembedding_matrix = model.get_output_embeddings().weight

    # 2. Probe and Optimizer initialization
    probe = ProbeClassifier(hidden_size=model.config.hidden_size, device=device, dtype=model.dtype)
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Get the token IDs for A, B, C, D
    answer_token_ids = [tokenizer.encode(letter, add_special_tokens=False)[0] for letter in ['A', 'B', 'C', 'D']]
    answer_map = {letter: i for i, letter in enumerate(['A', 'B', 'C', 'D'])}
    
    think_token_id = tokenizer.encode("<think>", add_special_tokens=False)[-1]

    # 3. Forward hook setup
    hidden_states_container = []
    def hook_fn(module, input, output):
        # output is a tuple for decoder layers, we need the first element
        hidden_states_container.append(output[0].cpu())

    hook = model.model.layers[layer_index].register_forward_hook(hook_fn)
    
    # 4. Training loop
    logging.info("Starting training loop...")
    global_step = 0
    for epoch in range(epochs):
        probe.train()
        
        for i, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
            optimizer.zero_grad()
            
            # --- Step 1: Get Hidden States and batch-level stats ---
            all_hiddens = []
            all_targets = []
            batch_prompt_lengths = []
            batch_last_hidden_preds = []
            batch_true_targets = []

            for sample in batch:
                input_ids = sample["token_ids"].to(device)
                batch_prompt_lengths.append(len(input_ids))
                
                # Clear container for each sample
                hidden_states_container.clear()
                
                # Forward pass to trigger the hook
                with torch.no_grad():
                    model(input_ids.unsqueeze(0))

                # Find the start of the reasoning part
                # The container will have one item of shape (1, seq_len, hidden_size)
                seq_hiddens = hidden_states_container[0].squeeze(0)
                
                # Find the position of <think> token
                think_token_positions = (input_ids == think_token_id).nonzero(as_tuple=True)[0]
                if think_token_positions.numel() > 0:
                    start_pos = think_token_positions[0]
                    
                    # Collect all hidden states from <think> to the end
                    relevant_hiddens = seq_hiddens[start_pos:]
                    
                    if relevant_hiddens.numel() > 0:
                        # For training on all hidden states
                        all_hiddens.append(relevant_hiddens)
                        target_letter = sample['correct_answer_letter']
                        target_idx = answer_map[target_letter]
                        all_targets.extend([target_idx] * len(relevant_hiddens))

                        # For batch-level accuracy
                        with torch.no_grad():
                            probe.eval() # Use eval mode for prediction
                            last_hidden = relevant_hiddens[-1].unsqueeze(0).to(device)
                            logits = probe(last_hidden, unembedding_matrix)
                            answer_logits = logits[:, answer_token_ids]
                            pred_idx = torch.argmax(answer_logits, dim=1).item()
                            batch_last_hidden_preds.append(pred_idx)
                            batch_true_targets.append(target_idx)
                            probe.train() # Back to train mode
            
            if not all_hiddens:
                continue

            avg_prompt_len = np.mean(batch_prompt_lengths) if batch_prompt_lengths else 0
            
            batch_accuracy = 0
            if batch_true_targets:
                batch_accuracy = np.mean([pred == true for pred, true in zip(batch_last_hidden_preds, batch_true_targets)])

            # --- Step 2: Train the Probe ---
            hiddens_tensor = torch.cat(all_hiddens, dim=0).to(device)
            targets_tensor = torch.tensor(all_targets, dtype=torch.long).to(device)
            
            # Shuffle the collected hidden states
            perm = torch.randperm(hiddens_tensor.size(0))
            hiddens_tensor = hiddens_tensor[perm]
            targets_tensor = targets_tensor[perm]

            # Process in sub-batches to avoid memory issues if hiddens_tensor is too large
            for sub_batch_idx, sub_batch_start in enumerate(range(0, hiddens_tensor.size(0), 16)): 
                sub_batch_hiddens = hiddens_tensor[sub_batch_start:sub_batch_start+16]
                sub_batch_targets = targets_tensor[sub_batch_start:sub_batch_start+16]
                
                # Get full logits
                logits = probe(sub_batch_hiddens, unembedding_matrix)
                
                # Filter logits to only A, B, C, D
                
                answer_logits = logits[:, answer_token_ids]
                
                loss = criterion(answer_logits, sub_batch_targets)
                loss.backward()

                # Log all metrics with the same global_step, which corresponds to a sub-batch
                run["train/loss"].log(loss.item(), step=global_step)

                preds = torch.argmax(answer_logits, dim=1)
                accuracy = (preds == sub_batch_targets).float().mean()
                run["train/accuracy_on_hiddens"].log(accuracy.item(), step=global_step)

                # Log batch-level stats (will be the same for all sub-batches of a batch)
                run["train/avg_prompt_length"].log(avg_prompt_len, step=global_step)
                run["train/batch_accuracy"].log(batch_accuracy, step=global_step)

                # Log sub-batch index
                run["train/sub_batch_index"].log(sub_batch_idx, step=global_step)

                # Log step-epoch relationship and increment step
                run["train/epoch"].log(epoch, step=global_step)
                global_step += 1
            
            optimizer.step()

        # --- 5. Validation Loop ---
        logging.info(f"Running validation for epoch {epoch+1}...")
        validate_and_log(probe, val_dataloader, model, hidden_states_container, think_token_id, layer_index, device, unembedding_matrix, answer_token_ids, answer_map, run, epoch+1)

    
    hook.remove()
    run.stop()
    logging.info("Training finished.")


if __name__ == "__main__":
    fire.Fire(main) 