import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import fire
import neptune
import logging
import numpy as np
import random
from collections import defaultdict, deque
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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


class ProbeLstmClassifier(nn.Module):
    """
    An LSTM-based probe classifier that operates on sequences of hidden states.
    It learns to classify each token's hidden state into one of N classes.
    The architecture is inspired by "Answer Convergence as a Signal for Early Stopping in Reasoning".
    
    The operation is: logits = Classifier(Dropout(LSTM(h_1, ..., h_t)))
    """
    def __init__(
        self,
        input_size: int,
        lstm_hidden_size: int,
        num_classes: int,
        dropout_rate: float,
        device: str,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True,
        ).to(device=device, dtype=dtype)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self.classifier = nn.Linear(lstm_hidden_size, num_classes).to(device=device, dtype=dtype)

    def forward(self, hidden_states: torch.Tensor):
        """
        Args:
            hidden_states (torch.Tensor): A tensor of shape (batch_size, seq_len, input_size)
                                          or (seq_len, input_size) for a single sample.

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, seq_len, num_classes) or (seq_len, num_classes).
        """
        is_batched = hidden_states.dim() == 3
        if not is_batched:
            hidden_states = hidden_states.unsqueeze(0)

        lstm_out, _ = self.lstm(hidden_states)
        lstm_out = self.dropout(lstm_out)
        logits = self.classifier(lstm_out)
        
        if not is_batched:
            logits = logits.squeeze(0)
            
        return logits


import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 15000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # x is batch-first, pe is seq-len first.
        x = x + self.pe.permute(1, 0, 2)[:, :x.size(1), :]
        return self.dropout(x)


class ProbeAttentionClassifier(nn.Module):
    """
    An Attention-based probe classifier that first projects hidden states to a smaller dimension.
    The operation is: 
    h_proj = Projection(h)
    logits = Classifier(Dropout(LayerNorm(h_proj + Attention(PositionalEncoding(h_proj)))))
    """
    def __init__(
        self,
        input_size: int,
        num_heads: int,
        num_classes: int,
        dropout_rate: float,
        device: str,
        dtype: torch.dtype,
        projection_dim: Optional[int] = None,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        
        # If no projection_dim is provided, it defaults to input_size
        self.projection_dim = projection_dim if projection_dim is not None else input_size
        
        # This is our new projection layer (the non-square matrix)
        if projection_dim:
            self.projection = nn.Linear(input_size, self.projection_dim).to(device=device, dtype=dtype)
        else:
            self.projection = nn.Identity()

        assert self.projection_dim % num_heads == 0, "projection_dim must be divisible by num_heads"

        self.pos_encoder = PositionalEncoding(d_model=self.projection_dim, dropout=dropout_rate).to(device=device, dtype=dtype)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.projection_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        ).to(device=device, dtype=dtype)

        self.layer_norm = nn.LayerNorm(self.projection_dim).to(device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.projection_dim, num_classes).to(device=device, dtype=dtype)

    def forward(self, hidden_states: torch.Tensor, key_padding_mask: torch.Tensor = None):
        """
        Args:
            hidden_states (torch.Tensor): A tensor of shape (batch_size, seq_len, input_size)
                                          or (seq_len, input_size) for a single sample.
            key_padding_mask (torch.Tensor, optional): Mask for padding tokens. Shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, seq_len, num_classes) or (seq_len, num_classes).
        """
        is_batched = hidden_states.dim() == 3
        if not is_batched:
            hidden_states = hidden_states.unsqueeze(0)

        # 1. Project to smaller dimension
        projected_hiddens = self.projection(hidden_states)

        # 2. Add positional encoding
        hiddens_with_pos = self.pos_encoder(projected_hiddens)

        # Create a causal mask to prevent the probe from looking ahead in the sequence.
        seq_len = hidden_states.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device=self.device, dtype=self.dtype)

        # 3. Self-attention with residual connection and layer norm
        attn_output, _ = self.attention(
            query=hiddens_with_pos, 
            key=hiddens_with_pos, 
            value=hiddens_with_pos,
            key_padding_mask=key_padding_mask,
            attn_mask=causal_mask,
            need_weights=False
        )
        
        # Add & Norm, but the residual connection is from the *projected* states
        norm_output = self.layer_norm(projected_hiddens + self.dropout(attn_output))
        
        logits = self.classifier(norm_output)
        
        if not is_batched:
            logits = logits.squeeze(0)
            
        return logits



import matplotlib.pyplot as plt
import seaborn as sns
from neptune.types import File
import os, json, csv


def validate_and_log_detailed(
    probe,
    val_q_ids,
    data_by_question,
    target_key,
    criterion,
    device,
    unembedding_matrix, # Can be None for LSTM probe
    answer_token_ids,
    answer_map_rev,
    tokenizer,
    run,
    epoch,
    probe_type: str,
):
    """
    Runs detailed validation, calculating loss, accuracy, a confusion matrix,
    and confidence dynamics.
    """
    probe.eval()
    all_targets = []
    all_preds = []
    total_val_loss = 0
    total_tokens = 0

    # For confidence dynamics plot
    num_bins = 20 
    prob_correct_binned = defaultdict(list)
    prob_incorrect_binned = defaultdict(list)
    prob_other_binned = defaultdict(list)
    prob_correct_abcd_binned = defaultdict(list)  # For the second plot
    accuracy_binned = defaultdict(list)  # For argmax accuracy dynamics
    
    # For quintile accuracy plots
    quintile_accuracies_binned = [[] for _ in range(5)]
    
    # Collect per-token detailed records
    detailed_records = []
    
    answer_token_ids_tensor = torch.tensor(answer_token_ids, device=device)
    
    with torch.no_grad():
        for q_id in tqdm(val_q_ids, desc="Running detailed validation"):
            for sample in data_by_question[q_id]:
                hiddens = sample['hiddens'].to(device)
                target_idx = sample[target_key]
                
                if hiddens.shape[0] == 0:
                    continue
                    
                seq_len = hiddens.shape[0]

                if probe_type == 'linear':
                    logits = probe(hiddens, unembedding_matrix) # (seq_len, vocab_size)
                    answer_logits_seq = logits[:, answer_token_ids]  # (seq_len, 4)
                elif probe_type == 'lstm':
                    answer_logits_seq = probe(hiddens) # (seq_len, 4)
                elif probe_type == 'attention':
                    # For validation, we process one sequence at a time, so no padding mask is needed.
                    answer_logits_seq = probe(hiddens, key_padding_mask=None)

                
                target_tensor_seq = torch.tensor([target_idx] * seq_len, device=device)
                
                loss = criterion(answer_logits_seq, target_tensor_seq)
                total_val_loss += loss.item() * seq_len
                total_tokens += seq_len
                
                preds_seq = torch.argmax(answer_logits_seq, dim=1).tolist()
                all_preds.extend(preds_seq)
                all_targets.extend([target_idx] * seq_len)
                
                # Save detailed per-token information
                answer_probs_seq = torch.softmax(answer_logits_seq, dim=-1)  # (seq_len, 4)
                for t_idx in range(seq_len):
                    probs = answer_probs_seq[t_idx].tolist()
                    token_id_int = int(sample['token_ids'][t_idx]) if 'token_ids' in sample else -1
                    token_text = tokenizer.convert_ids_to_tokens(token_id_int) if token_id_int >= 0 else "?"
                    record = {
                        "epoch": int(epoch),
                        "question_id": int(q_id),
                        "token_idx": int(t_idx),
                        "token_id": token_id_int,
                        "token_text": token_text,
                        "prob_A": probs[0],
                        "prob_B": probs[1],
                        "prob_C": probs[2],
                        "prob_D": probs[3],
                        "pred_letter": answer_map_rev[preds_seq[t_idx]],
                        "target_letter": answer_map_rev[target_idx],
                    }
                    detailed_records.append(record)
                
                # --- Confidence Dynamics Calculations ---
                
                # 1. Probs over the entire vocabulary (for the main plot)
                if probe_type == 'linear':
                    probs = torch.softmax(logits, dim=-1)
                    p_answers = probs[:, answer_token_ids_tensor]
                    p_all_answers = torch.sum(p_answers, dim=-1)
                    p_correct = p_answers[:, target_idx]
                    p_incorrect = p_all_answers - p_correct
                    p_other = 1.0 - p_all_answers
                
                # 2. Probs over only A,B,C,D (for the second plot)
                abcd_probs = torch.softmax(answer_logits_seq, dim=-1)
                p_correct_abcd = abcd_probs[:, target_idx]
                
                num_tokens = hiddens.shape[0]
                for i in range(num_tokens):
                    relative_position = i / num_tokens if num_tokens > 1 else 0
                    bin_index = int(relative_position * num_bins)
                    
                    # Quintile calculation
                    quintile_index = int(relative_position * 5)
                    if quintile_index == 5: quintile_index = 4 # Ensure it stays within 0-4 range
                    
                    is_correct = 1 if preds_seq[i] == target_idx else 0
                    quintile_accuracies_binned[quintile_index].append(is_correct)

                    if probe_type == 'linear':
                        prob_correct_binned[bin_index].append(p_correct[i].item())
                        prob_incorrect_binned[bin_index].append(p_incorrect[i].item())
                        prob_other_binned[bin_index].append(p_other[i].item())

                    prob_correct_abcd_binned[bin_index].append(p_correct_abcd[i].item())
                    accuracy_binned[bin_index].append(is_correct)

    # --- Log Main Metrics ---
    avg_val_loss = total_val_loss / total_tokens if total_tokens > 0 else 0
    accuracy = np.mean([p == t for p, t in zip(all_preds, all_targets)]) if all_targets else 0

    logging.info(f"Epoch {epoch+1}: Val Loss: {avg_val_loss:.4f}, Val Accuracy: {accuracy:.4f}")
    run['val/loss'].log(avg_val_loss, step=epoch + 1)
    run['val/accuracy'].log(accuracy, step=epoch + 1)

    # --- Log Quintile Accuracies ---
    for i in range(5):
        quintile_acc = np.mean(quintile_accuracies_binned[i]) if quintile_accuracies_binned[i] else 0
        run[f'val/accuracy_quintile_{i+1}'].log(quintile_acc, step=epoch + 1)
        logging.info(f"  Accuracy for quintile {i+1} ({(i)*20}%-{(i+1)*20}%): {quintile_acc:.4f}")

    # --- Save detailed per-token predictions ---
    if detailed_records:
        os.makedirs("probe_predictions", exist_ok=True)
        epoch_label = "pre" if epoch < 0 else f"epoch{epoch+1}"
        jsonl_path = os.path.join("probe_predictions", f"val_details_{epoch_label}_{probe_type}.jsonl")
        csv_path = os.path.join("probe_predictions", f"val_details_{epoch_label}_{probe_type}.csv")
        with open(jsonl_path, 'w') as jf:
            for rec in detailed_records:
                jf.write(json.dumps(rec) + '\n')
        fieldnames = detailed_records[0].keys()
        with open(csv_path, 'w', newline='') as cf:
            writer = csv.DictWriter(cf, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(detailed_records)
        logging.info(f"Saved detailed validation predictions to {jsonl_path} and {csv_path}")
        # Upload to Neptune
        # run[f"val/detailed_jsonl/{epoch_label}"].upload(jsonl_path)
        # run[f"val/detailed_csv/{epoch_label}"].upload(csv_path)

    # --- Confusion Matrix ---
    if all_targets:
        cm = np.zeros((len(answer_map_rev), len(answer_map_rev)))
        for true, pred in zip(all_targets, all_preds):
            cm[true, pred] += 1
        
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(cm, annot=True, fmt=".0f", cmap='Blues', 
                    xticklabels=answer_map_rev.values(), yticklabels=answer_map_rev.values(), ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix - Epoch {epoch+1}")
        run["val/confusion_matrix"].log(File.as_image(fig), step=epoch + 1)
        plt.close(fig)

    # --- Confidence Dynamics ---
    if probe_type == 'linear':
        fig_dyn, ax_dyn = plt.subplots(figsize=(10, 6))
        bins = range(num_bins)
        x_axis = [(i + 0.5) / num_bins for i in bins]
        
        mean_p_correct = [np.mean(prob_correct_binned[i]) if prob_correct_binned[i] else np.nan for i in bins]
        mean_p_incorrect = [np.mean(prob_incorrect_binned[i]) if prob_incorrect_binned[i] else np.nan for i in bins]
        mean_p_other = [np.mean(prob_other_binned[i]) if prob_other_binned[i] else np.nan for i in bins]
        
        ax_dyn.plot(x_axis, mean_p_correct, label='P(correct)', marker='o')
        ax_dyn.plot(x_axis, mean_p_incorrect, label='P(incorrect)', marker='o')
        ax_dyn.plot(x_axis, mean_p_other, label='P(other)', marker='o')
        
        ax_dyn.set_xlabel("Relative Position in Reasoning Trace")
        ax_dyn.set_ylabel("Average Probability")
        ax_dyn.set_title(f"Confidence Dynamics - Epoch {epoch+1}")
        ax_dyn.legend()
        ax_dyn.grid(True)
        plt.tight_layout()
        
        run["val/confidence_dynamics"].log(File.as_image(fig_dyn), step=epoch + 1)
        plt.close(fig_dyn)
    
    # --- Confidence Dynamics (Correct Only, normalized over A,B,C,D) ---
    fig_correct, ax_correct = plt.subplots(figsize=(10, 6))
    bins = range(num_bins)
    x_axis = [(i + 0.5) / num_bins for i in bins]
    
    mean_p_correct_abcd = [np.mean(prob_correct_abcd_binned[i]) if prob_correct_abcd_binned[i] else np.nan for i in bins]
    ax_correct.plot(x_axis, mean_p_correct_abcd, label='P(correct|A,B,C,D)', marker='o', color='C0')
    
    ax_correct.set_xlabel("Relative Position in Reasoning Trace")
    ax_correct.set_ylabel("Average Probability")
    ax_correct.set_title(f"P(correct | A,B,C,D) Dynamics - Epoch {epoch+1}")
    ax_correct.legend()
    ax_correct.grid(True)
    plt.tight_layout()
    
    run["val/p_correct_dynamics"].log(File.as_image(fig_correct), step=epoch + 1)
    plt.close(fig_correct)

    # --- Accuracy Dynamics (argmax) ---
    fig_acc, ax_acc = plt.subplots(figsize=(10, 6))
    mean_accuracy_dyn = [np.mean(accuracy_binned[i]) if accuracy_binned[i] else np.nan for i in bins]
    ax_acc.plot(x_axis, mean_accuracy_dyn, label='Accuracy (argmax)', marker='o', color='C3')
    ax_acc.set_xlabel("Relative Position in Reasoning Trace")
    ax_acc.set_ylabel("Average Accuracy")
    ax_acc.set_title(f"Accuracy Dynamics - Epoch {epoch+1}")
    ax_acc.set_ylim(0, 1)
    ax_acc.grid(True)
    plt.tight_layout()
    run["val/accuracy_dynamics"].log(File.as_image(fig_acc), step=epoch + 1)
    plt.close(fig_acc)

def main(
    probe_data_path: str = "probe_data/probe_data_45.pt",
    model_name: str = "Qwen/Qwen3-32B",
    target_type: str = "model_answer",
    probe_type: str = "linear",
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    epochs: int = 5,
    val_split_size: float = 0.2,
    neptune_project: str = "probing-reasoning-classifier",
    # LSTM-specific parameters
    lstm_hidden_size: int = 128,
    dropout_rate: float = 0.1,
    # Attention-specific parameters
    attention_heads: int = 1,
    attention_projection_dim: Optional[int] = 64,
    # Optimizer parameters
    weight_decay: float = 0.5,
):
    """
    Trains a probe classifier from a pre-generated dataset of hidden states.

    Args:
        probe_data_path (str): Path to the .pt file containing the probe data.
        model_name (str): The name of the Hugging Face model (for unembedding matrix and config).
        target_type (str): What to predict: 'ground_truth' or 'model_answer'.
        probe_type (str): Type of probe to train: 'linear', 'lstm', or 'attention'.
        batch_size (int): Training batch size. For LSTM/Attention, this is per-sequence.
        learning_rate (float): Adam optimizer learning rate.
        epochs (int): Number of training epochs.
        val_split_size (float): Fraction of data to use for validation.
        neptune_project (str): Neptune project name.
        lstm_hidden_size (int): Hidden size for the LSTM probe.
        dropout_rate (float): Dropout rate for the LSTM/Attention probe.
        attention_heads (int): Number of heads for the Attention probe.
        attention_projection_dim (Optional[int]): Project hidden states to this dimension before attention.
        weight_decay (float): Weight decay (L2 penalty) for the Adam optimizer.
    """
    assert target_type in ['ground_truth', 'model_answer'], "target_type must be 'ground_truth' or 'model_answer'"
    assert probe_type in ['linear', 'lstm', 'attention'], "probe_type must be 'linear', 'lstm', or 'attention'"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # --- 1. Initialize Neptune ---
    logging.info(f"Initializing Neptune for project: {neptune_project}")
    run = neptune.init_run(
        project=neptune_project,
        tags=["train_from_file", target_type, probe_type, probe_data_path]
    )
    run["parameters"] = {
        "probe_data_path": probe_data_path,
        "model_name": model_name,
        "target_type": target_type,
        "probe_type": probe_type,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "val_split_size": val_split_size,
        "weight_decay": weight_decay,
    }
    if probe_type == 'lstm':
        run["parameters/lstm"] = {
            "lstm_hidden_size": lstm_hidden_size,
            "dropout_rate": dropout_rate,
        }
    elif probe_type == 'attention':
        run["parameters/attention"] = {
            "attention_heads": attention_heads,
            "attention_projection_dim": attention_projection_dim,
            "dropout_rate": dropout_rate,
        }
    
    # --- 2. Load Model components (for config and unembedding) ---
    model_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    
    cache_dir = "unembedding_cache"
    os.makedirs(cache_dir, exist_ok=True)
    safe_model_name = model_name.replace("/", "_")
    cache_path = os.path.join(cache_dir, f"{safe_model_name}_unembedding.pt")

    if os.path.exists(cache_path):
        logging.info(f"Loading cached unembedding matrix from {cache_path}...")
        cached_data = torch.load(cache_path, map_location=device)
        hidden_size = cached_data['hidden_size']
        unembedding_matrix = cached_data['unembedding_matrix'].to(dtype=model_dtype)
        logging.info("Cached components loaded.")
    else:
        logging.info(f"Loading {model_name} for config and unembedding matrix...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=model_dtype
        ).to(device)
        
        hidden_size = model.config.hidden_size
        unembedding_matrix = model.get_output_embeddings().weight.clone().to(device=device, dtype=model_dtype)
        
        logging.info(f"Saving unembedding matrix to {cache_path}")
        torch.save({
            'hidden_size': hidden_size,
            'unembedding_matrix': unembedding_matrix.cpu() # Save on CPU
        }, cache_path)
        
        del model
        torch.cuda.empty_cache()
        logging.info("Model components loaded and cached, base model deleted from memory.")

    # --- 3. Load and prepare data ---
    logging.info(f"Loading probe data from {probe_data_path}")
    probe_data = torch.load(probe_data_path)
    
    if target_type == 'model_answer':
        original_count = len(probe_data)
        probe_data = [d for d in probe_data if d['model_answer_idx'] != -1]
        logging.info(f"Filtered out {original_count - len(probe_data)} samples with invalid model answers.")

    # Group by question_id for splitting
    data_by_question = defaultdict(list)
    for sample in probe_data:
        data_by_question[sample['question_id']].append(sample)
    
    question_ids = list(data_by_question.keys())
    random.Random(42).shuffle(question_ids) # Deterministic shuffle
    
    split_idx = int(len(question_ids) * (1 - val_split_size))
    train_q_ids = question_ids[:split_idx]
    val_q_ids = question_ids[split_idx:]
    
    def create_tensor_dataset(q_ids, data_map, target_key):
        all_hiddens = []
        all_targets = []
        for q_id in q_ids:
            for sample in data_map[q_id]:
                hiddens = sample['hiddens']
                target = sample[target_key]
                all_hiddens.append(hiddens)
                all_targets.extend([target] * hiddens.shape[0])
        
        if not all_hiddens:
            return None
            
        hiddens_tensor = torch.cat(all_hiddens, dim=0)
        targets_tensor = torch.tensor(all_targets, dtype=torch.long)
        return TensorDataset(hiddens_tensor, targets_tensor)

    # For LSTM/Attention, we need a different dataset structure that keeps sequences separate
    class SequenceDataset(torch.utils.data.Dataset):
        def __init__(self, q_ids, data_map, target_key):
            self.sequences = []
            for q_id in q_ids:
                for sample in data_map[q_id]:
                    if sample['hiddens'].shape[0] > 0:
                        self.sequences.append({
                            'hiddens': sample['hiddens'],
                            'target': sample[target_key]
                        })

        def __len__(self):
            return len(self.sequences)

        def __getitem__(self, idx):
            return self.sequences[idx]

    def collate_fn_for_sequences(batch):
        hiddens_list = [item['hiddens'] for item in batch]
        targets_list = [item['target'] for item in batch]
        lengths = torch.tensor([len(h) for h in hiddens_list])

        # Pad sequences
        hiddens_padded = pad_sequence(hiddens_list, batch_first=True, padding_value=0.0)
        
        # Create a targets tensor that matches the padded sequence length
        # We'll repeat the target for each token in the original sequence
        targets_padded = torch.full_like(hiddens_padded[:, :, 0], fill_value=-100, dtype=torch.long) # -100 is ignored by CrossEntropyLoss
        for i, target in enumerate(targets_list):
            targets_padded[i, :lengths[i]] = target
            
        return hiddens_padded, targets_padded, lengths

    logging.info("Creating train and validation datasets...")
    target_key = 'ground_truth_idx' if target_type == 'ground_truth' else 'model_answer_idx'
    
    if probe_type == 'linear':
        train_dataset = create_tensor_dataset(train_q_ids, data_by_question, target_key)
        
        if not train_dataset:
            logging.error("Train dataset is empty. Exiting.")
            run.stop()
            return
            
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        logging.info(f"Created train dataset with {len(train_dataset)} hidden states for Linear Probe.")

    else: # lstm or attention
        train_dataset = SequenceDataset(train_q_ids, data_by_question, target_key)
        if len(train_dataset) == 0:
            logging.error("Train dataset is empty. Exiting.")
            run.stop()
            return
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn_for_sequences
        )
        logging.info(f"Created train dataset with {len(train_dataset)} sequences for {probe_type.upper()} Probe.")

    if val_q_ids:
        logging.info(f"Created validation set with {len(val_q_ids)} questions.")

    # --- 4. Initialize Probe and Optimizer ---
    answer_letters = ['A', 'B', 'C', 'D']
    if probe_type == 'linear':
        probe = ProbeClassifier(hidden_size=hidden_size, device=device, dtype=model_dtype)
    elif probe_type == 'lstm':
        probe = ProbeLstmClassifier(
            input_size=hidden_size,
            lstm_hidden_size=lstm_hidden_size,
            num_classes=len(answer_letters),
            dropout_rate=dropout_rate,
            device=device,
            dtype=model_dtype,
        )
    elif probe_type == 'attention':
        probe = ProbeAttentionClassifier(
            input_size=hidden_size,
            num_heads=attention_heads,
            num_classes=len(answer_letters),
            dropout_rate=dropout_rate,
            device=device,
            dtype=model_dtype,
            projection_dim=attention_projection_dim,
        )

    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    answer_token_ids = [tokenizer.convert_tokens_to_ids(letter) for letter in answer_letters]
    assert all(token_id is not None for token_id in answer_token_ids), "Some answers are not valid tokens"

    answer_map_rev = {i: letter for i, letter in enumerate(answer_letters)}


    # --- 5. Pre-training Validation ---
    if val_q_ids:
        logging.info("Running pre-training validation...")
        validate_and_log_detailed(
            probe, val_q_ids, data_by_question, target_key, criterion,
            device, unembedding_matrix if probe_type == 'linear' else None, 
            answer_token_ids, answer_map_rev, tokenizer, run, epoch=-1,
            probe_type=probe_type
        )

    # --- 6. Training Loop ---
    logging.info("Starting training...")
    global_step = 0
    for epoch in range(epochs):
        probe.train()
        
        # Keep track of recent training accuracies
        recent_accuracies = deque(maxlen=10)
        recent_quintile_accuracies = [deque(maxlen=10) for _ in range(5)]
        
        # Shuffle question IDs for each epoch for sequence-based training
        if probe_type in ['lstm', 'attention']:
            random.shuffle(train_q_ids)

        # The training loop differs for the two probe types
        if probe_type == 'linear':
            for hiddens, targets in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                optimizer.zero_grad()
                
                hiddens, targets = hiddens.to(device), targets.to(device)
                
                logits = probe(hiddens, unembedding_matrix)
                
                answer_logits = logits[:, answer_token_ids]
                loss = criterion(answer_logits, targets)
                
                loss.backward()
                optimizer.step()
                
                run['train/loss'].log(loss.item(), step=global_step)
                
                preds = torch.argmax(answer_logits, dim=1)
                accuracy = (preds == targets).float().mean()
                run['train/accuracy'].log(accuracy.item(), step=global_step)
                recent_accuracies.append(accuracy.item())
                
                run['misc/epoch'].log(epoch + 1, step=global_step)
                global_step += 1
        
        else: # lstm or attention
            for hiddens, targets, lengths in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} ({probe_type.upper()})"):
                optimizer.zero_grad()
                
                hiddens, targets = hiddens.to(device), targets.to(device)
                
                # hiddens: (batch_size, seq_len, hidden_size)
                # targets: (batch_size, seq_len), padded with -100
                
                if probe_type == 'attention':
                    # Create a key_padding_mask
                    mask = (targets == -100) # True for padded tokens
                    answer_logits = probe(hiddens, key_padding_mask=mask)
                else: # lstm
                    answer_logits = probe(hiddens) # (batch_size, seq_len, 4)
                
                # Reshape for CrossEntropyLoss, which ignores -100
                loss = criterion(answer_logits.view(-1, 4), targets.view(-1))
                
                loss.backward()
                optimizer.step()
                
                run['train/loss'].log(loss.item(), step=global_step)
                
                # Calculate accuracy and quintile accuracies
                with torch.no_grad():
                    preds = torch.argmax(answer_logits, dim=2) # (batch_size, seq_len)
                    mask = targets != -100
                    
                    # Overall accuracy
                    if mask.any():
                        accuracy = (preds[mask] == targets[mask]).float().mean()
                        recent_accuracies.append(accuracy.item())
                        run['train/accuracy'].log(accuracy.item(), step=global_step)

                    # --- Log Quintile Accuracies for Training Batch ---
                    seq_indices = torch.arange(targets.shape[1], device=device).unsqueeze(0).expand_as(targets)
                    safe_lengths = torch.clamp(lengths.unsqueeze(1), min=1).to(device)
                    relative_positions = seq_indices / safe_lengths
                    
                    quintile_indices = (relative_positions * 5).long()
                    quintile_indices = torch.clamp(quintile_indices, 0, 4)

                    is_correct_matrix = (preds == targets)

                    for i in range(5):
                        quintile_mask = (quintile_indices == i) & mask
                        if quintile_mask.any():
                            quintile_acc = is_correct_matrix[quintile_mask].float().mean().item()
                        else:
                            quintile_acc = 0.0
                        
                        run[f'train/accuracy_quintile_{i+1}'].log(quintile_acc, step=global_step)
                        recent_quintile_accuracies[i].append(quintile_acc)

                run['misc/epoch'].log(epoch + 1, step=global_step)
                global_step += 1
            
        # --- 7. Validation ---
        if val_q_ids:
            if recent_accuracies:
                avg_recent_acc = np.mean(list(recent_accuracies))
                logging.info(f"Epoch {epoch+1}: Avg Train Accuracy (last {len(recent_accuracies)} batches): {avg_recent_acc:.4f}")
                
                if probe_type in ['lstm', 'attention']:
                    for i in range(5):
                        if recent_quintile_accuracies[i]:
                             quintile_acc = np.mean(list(recent_quintile_accuracies[i]))
                             logging.info(f"  Avg Train Accuracy Quintile {i+1} ({(i)*20}%-{(i+1)*20}%): {quintile_acc:.4f}")

            validate_and_log_detailed(
                probe, val_q_ids, data_by_question, target_key, criterion,
                device, unembedding_matrix if probe_type == 'linear' else None, 
                answer_token_ids, answer_map_rev, tokenizer, run, epoch,
                probe_type=probe_type
            )
            
    run.stop()
    logging.info("Training finished.")

if __name__ == "__main__":
    fire.Fire(main)
