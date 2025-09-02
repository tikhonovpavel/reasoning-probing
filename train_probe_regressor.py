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


class ProbeRegressor(nn.Module):
    """A linear probe regressor that operates on hidden states."""
    def __init__(self, hidden_size: int, device: str, dtype: torch.dtype):
        super().__init__()
        self.hidden_size = hidden_size
        self.device = device
        # A single output for regression
        self.regressor = nn.Linear(hidden_size, 1).to(device=device, dtype=dtype)

    def forward(self, hidden_states: torch.Tensor):
        """
        Args:
            hidden_states (torch.Tensor): A tensor of shape (batch_size, hidden_size).
        Returns:
            torch.Tensor: Predictions tensor of shape (batch_size, 1).
        """
        return self.regressor(hidden_states)


class ProbeLstmRegressor(nn.Module):
    """An LSTM-based probe regressor."""
    def __init__(
        self,
        input_size: int,
        lstm_hidden_size: int,
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
        self.regressor = nn.Linear(lstm_hidden_size, 1).to(device=device, dtype=dtype)

    def forward(self, hidden_states: torch.Tensor):
        """
        Args:
            hidden_states (torch.Tensor): Shape (batch_size, seq_len, input_size) or (seq_len, input_size).
        Returns:
            torch.Tensor: Predictions of shape (batch_size, seq_len, 1) or (seq_len, 1).
        """
        is_batched = hidden_states.dim() == 3
        if not is_batched:
            hidden_states = hidden_states.unsqueeze(0)

        lstm_out, _ = self.lstm(hidden_states)
        lstm_out = self.dropout(lstm_out)
        predictions = self.regressor(lstm_out)
        
        if not is_batched:
            predictions = predictions.squeeze(0)
            
        return predictions


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


class ProbeAttentionRegressor(nn.Module):
    """An Attention-based probe regressor."""
    def __init__(
        self,
        input_size: int,
        num_heads: int,
        dropout_rate: float,
        device: str,
        dtype: torch.dtype,
        projection_dim: Optional[int] = None,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        
        self.projection_dim = projection_dim if projection_dim is not None else input_size
        
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
        self.regressor = nn.Linear(self.projection_dim, 1).to(device=device, dtype=dtype)

    def forward(self, hidden_states: torch.Tensor, key_padding_mask: torch.Tensor = None):
        """
        Args:
            hidden_states (torch.Tensor): Shape (batch_size, seq_len, input_size) or (seq_len, input_size).
            key_padding_mask (torch.Tensor, optional): Mask for padding. Shape (batch_size, seq_len).
        Returns:
            torch.Tensor: Predictions of shape (batch_size, seq_len, 1) or (seq_len, 1).
        """
        is_batched = hidden_states.dim() == 3
        if not is_batched:
            hidden_states = hidden_states.unsqueeze(0)

        projected_hiddens = self.projection(hidden_states)
        hiddens_with_pos = self.pos_encoder(projected_hiddens)

        seq_len = hidden_states.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device=self.device, dtype=self.dtype)

        attn_output, _ = self.attention(
            query=hiddens_with_pos, 
            key=hiddens_with_pos, 
            value=hiddens_with_pos,
            key_padding_mask=key_padding_mask,
            attn_mask=causal_mask,
            need_weights=False
        )
        
        norm_output = self.layer_norm(projected_hiddens + self.dropout(attn_output))
        predictions = self.regressor(norm_output)
        
        if not is_batched:
            predictions = predictions.squeeze(0)
            
        return predictions



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
    tokenizer,
    run,
    epoch,
    probe_type: str,
):
    """
    Runs detailed validation for the regressor, calculating loss, MAE,
    a scatter plot, and MAE dynamics.
    """
    probe.eval()
    all_targets = []
    all_preds = []
    total_val_loss = 0
    total_tokens = 0

    # For dynamics plot
    num_bins = 20
    mae_binned = defaultdict(list)
    quintile_maes_binned = [[] for _ in range(5)]

    # Collect per-token detailed records
    detailed_records = []
    
    with torch.no_grad():
        for q_id in tqdm(val_q_ids, desc="Running detailed validation"):
            for sample in data_by_question[q_id]:
                hiddens = sample['hiddens'].to(device)
                target_value = sample[target_key]
                
                if hiddens.shape[0] == 0:
                    continue
                    
                seq_len = hiddens.shape[0]

                if probe_type == 'linear':
                    outputs_seq = probe(hiddens) # (seq_len, 1)
                elif probe_type in ['lstm', 'attention']:
                    # For validation, we process one sequence at a time, so no padding mask is needed.
                    outputs_seq = probe(hiddens, key_padding_mask=None) if probe_type == 'attention' else probe(hiddens)

                target_tensor_seq = torch.tensor([target_value] * seq_len, device=device, dtype=hiddens.dtype).unsqueeze(1)

                loss = criterion(outputs_seq, target_tensor_seq).sum() # Sum loss over sequence
                total_val_loss += loss.item()
                total_tokens += seq_len
                
                # --- Per-token data collection ---
                preds_seq = outputs_seq.squeeze(1).tolist()
                all_preds.extend(preds_seq)
                all_targets.extend([target_value] * seq_len)

                # Save detailed per-token information
                for t_idx in range(seq_len):
                    token_id_int = int(sample['token_ids'][t_idx]) if 'token_ids' in sample else -1
                    token_text = tokenizer.decode(token_id_int) if token_id_int >= 0 else "?"
                    
                    record = {
                        "epoch": int(epoch),
                        "question_id": int(q_id),
                        "token_idx": int(t_idx),
                        "token_id": token_id_int,
                        "token_text": token_text,
                        "prediction": preds_seq[t_idx],
                        "target": target_value,
                        "abs_error": abs(preds_seq[t_idx] - target_value)
                    }
                    detailed_records.append(record)
                
                # --- Binned Metrics Calculations ---
                num_tokens = hiddens.shape[0]
                for i in range(num_tokens):
                    relative_position = i / num_tokens if num_tokens > 1 else 0
                    bin_index = int(relative_position * num_bins)
                    quintile_index = int(relative_position * 5)
                    if quintile_index == 5: quintile_index = 4
                    
                    abs_error = abs(preds_seq[i] - target_value)
                    quintile_maes_binned[quintile_index].append(abs_error)
                    mae_binned[bin_index].append(abs_error)


    # --- Log Main Metrics ---
    avg_val_loss = total_val_loss / total_tokens if total_tokens > 0 else 0
    run['val/loss'].log(avg_val_loss, step=epoch + 1)

    mae = np.mean([abs(p - t) for p, t in zip(all_preds, all_targets)]) if all_targets else 0
    logging.info(f"Epoch {epoch+1}: Val Loss: {avg_val_loss:.4f}, Val MAE: {mae:.4f}")
    run['val/mae'].log(mae, step=epoch + 1)
    
    # --- Log Quintile MAEs ---
    for i in range(5):
        quintile_mae = np.mean(quintile_maes_binned[i]) if quintile_maes_binned[i] else 0
        run[f'val/mae_quintile_{i+1}'].log(quintile_mae, step=epoch + 1)
        logging.info(f"  MAE for quintile {i+1} ({(i)*20}%-{(i+1)*20}%): {quintile_mae:.4f}")

    # --- Save detailed per-token predictions ---
    if detailed_records:
        os.makedirs("probe_predictions", exist_ok=True)
        epoch_label = "pre" if epoch < 0 else f"epoch{epoch+1}"
        safe_probe_type = probe_type.replace("/", "_")
        jsonl_path = os.path.join("probe_predictions", f"val_details_{epoch_label}_{safe_probe_type}_reg.jsonl")
        csv_path = os.path.join("probe_predictions", f"val_details_{epoch_label}_{safe_probe_type}_reg.csv")
        with open(jsonl_path, 'w') as jf:
            for rec in detailed_records:
                jf.write(json.dumps(rec) + '\n')
        fieldnames = detailed_records[0].keys()
        with open(csv_path, 'w', newline='') as cf:
            writer = csv.DictWriter(cf, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(detailed_records)
        logging.info(f"Saved detailed validation predictions to {jsonl_path} and {csv_path}")

    # --- PLOTTING ---
    if not all_targets:
        return # No data to plot

    # --- Scatter Plot (Predicted vs. True) ---
    fig_scatter, ax_scatter = plt.subplots(figsize=(8, 8))
    # Subsample for performance if there are too many points
    plot_points = 10000
    if len(all_targets) > plot_points:
        indices = np.random.choice(len(all_targets), plot_points, replace=False)
        targets_sample = np.array(all_targets)[indices]
        preds_sample = np.array(all_preds)[indices]
    else:
        targets_sample, preds_sample = all_targets, all_preds
        
    ax_scatter.scatter(targets_sample, preds_sample, alpha=0.1, s=5)
    # Add a y=x line for reference
    lims = [
        np.min([ax_scatter.get_xlim(), ax_scatter.get_ylim()]),
        np.max([ax_scatter.get_xlim(), ax_scatter.get_ylim()]),
    ]
    ax_scatter.plot(lims, lims, 'r-', alpha=0.75, zorder=0)
    ax_scatter.set_xlabel("True Values")
    ax_scatter.set_ylabel("Predicted Values")
    ax_scatter.set_title(f"Predicted vs. True Values - Epoch {epoch+1}")
    ax_scatter.grid(True)
    plt.tight_layout()
    run["val/scatter_plot"].log(File.as_image(fig_scatter), step=epoch + 1)
    plt.close(fig_scatter)

    # --- MAE Dynamics ---
    fig_mae, ax_mae = plt.subplots(figsize=(10, 6))
    bins = range(num_bins)
    x_axis = [(i + 0.5) / num_bins for i in bins]
    mean_mae_dyn = [np.mean(mae_binned[i]) if mae_binned[i] else np.nan for i in bins]
    ax_mae.plot(x_axis, mean_mae_dyn, label='Mean Absolute Error', marker='o', color='C1')
    ax_mae.set_xlabel("Relative Position in Reasoning Trace")
    ax_mae.set_ylabel("Average MAE")
    ax_mae.set_title(f"MAE Dynamics - Epoch {epoch+1}")
    ax_mae.grid(True)
    plt.tight_layout()
    run["val/mae_dynamics"].log(File.as_image(fig_mae), step=epoch + 1)
    plt.close(fig_mae)


def main(
    probe_data_path: str,
    model_name: str = "Qwen/QwQ-32B",
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
    Trains a probe regressor from a pre-generated dataset of hidden states.

    Args:
        probe_data_path (str): Path to the .pt file containing the probe data.
        model_name (str): The name of the Hugging Face model (for config).
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
    assert probe_type in ['linear', 'lstm', 'attention'], "probe_type must be 'linear', 'lstm', or 'attention'"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # --- 1. Initialize Neptune ---
    logging.info(f"Initializing Neptune for project: {neptune_project}")
    run = neptune.init_run(
        project=neptune_project,
        tags=["train_regressor", probe_type, probe_data_path]
    )
    run["parameters"] = {
        "probe_data_path": probe_data_path,
        "model_name": model_name,
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
    
    # --- 2. Load Model components (for config) ---
    model_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    
    # We only need the config for the hidden size, not the full model
    from transformers import AutoConfig
    logging.info(f"Loading config for {model_name}...")
    config = AutoConfig.from_pretrained(model_name)
    hidden_size = config.hidden_size
    logging.info(f"Model hidden size: {hidden_size}")


    # --- 3. Load and prepare data ---
    logging.info(f"Loading probe data from {probe_data_path}")
    probe_data = torch.load(probe_data_path)
    
    # Group by question_id for splitting
    data_by_question = defaultdict(list)
    for sample in probe_data:
        data_by_question[sample['question_id']].append(sample)
    
    question_ids = list(data_by_question.keys())
    random.Random(42).shuffle(question_ids) # Deterministic shuffle
    
    split_idx = int(len(question_ids) * (1 - val_split_size))
    train_q_ids = question_ids[:split_idx]
    val_q_ids = question_ids[split_idx:]
    
    target_key = 'ground_truth_value'

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
        
        targets_tensor = torch.tensor(all_targets, dtype=torch.float32)
        targets_tensor = targets_tensor.unsqueeze(1) # Reshape for MSELoss

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
        
        # For regression, pad with a value like 0.0, the loss function will use a mask.
        targets_padded = torch.full_like(hiddens_padded[:, :, :1], fill_value=0.0, dtype=torch.float32)
        for i, target in enumerate(targets_list):
            targets_padded[i, :lengths[i], 0] = target
            
        return hiddens_padded, targets_padded, lengths

    logging.info("Creating train and validation datasets...")
    
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
    if probe_type == 'linear':
        probe = ProbeRegressor(hidden_size=hidden_size, device=device, dtype=model_dtype)
    elif probe_type == 'lstm':
        probe = ProbeLstmRegressor(
            input_size=hidden_size,
            lstm_hidden_size=lstm_hidden_size,
            dropout_rate=dropout_rate,
            device=device,
            dtype=model_dtype,
        )
    elif probe_type == 'attention':
        probe = ProbeAttentionRegressor(
            input_size=hidden_size,
            num_heads=attention_heads,
            dropout_rate=dropout_rate,
            device=device,
            dtype=model_dtype,
            projection_dim=attention_projection_dim,
        )

    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss(reduction='none') # Use 'none' for sequence masking
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)


    # --- 5. Pre-training Validation ---
    if val_q_ids:
        logging.info("Running pre-training validation...")
        validate_and_log_detailed(
            probe, val_q_ids, data_by_question, target_key, criterion,
            device, tokenizer, run, epoch=-1,
            probe_type=probe_type,
        )

    # --- 6. Training Loop ---
    logging.info("Starting training...")
    global_step = 0
    for epoch in range(epochs):
        probe.train()
        
        # The training loop differs for the two probe types
        if probe_type == 'linear':
            for hiddens, targets in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                optimizer.zero_grad()
                
                hiddens, targets = hiddens.to(device), targets.to(device)
                
                predictions = probe(hiddens)
                loss = criterion(predictions, targets).mean()
                
                loss.backward()
                optimizer.step()
                
                run['train/loss'].log(loss.item(), step=global_step)
                
                with torch.no_grad():
                    mae = torch.abs(predictions - targets).mean()
                    run['train/mae'].log(mae.item(), step=global_step)
                
                run['misc/epoch'].log(epoch + 1, step=global_step)
                global_step += 1
        
        else: # lstm or attention
            for hiddens, targets, lengths in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} ({probe_type.upper()})"):
                optimizer.zero_grad()
                
                hiddens, targets = hiddens.to(device), targets.to(device)
                
                # hiddens: (batch_size, seq_len, hidden_size)
                # targets: (batch, seq, 1)
                
                if probe_type == 'attention':
                    # Create a key_padding_mask
                    max_len = targets.shape[1]
                    mask = torch.arange(max_len, device=device)[None, :] >= lengths[:, None]
                    outputs = probe(hiddens, key_padding_mask=mask)
                else: # lstm
                    outputs = probe(hiddens) # (batch_size, seq_len, 1)
                
                # Create mask to exclude padded tokens from loss
                max_len = targets.shape[1]
                mask = torch.arange(max_len, device=device)[None, :] < lengths[:, None]
                
                masked_loss = criterion(outputs.squeeze(-1), targets.squeeze(-1))
                loss = torch.sum(masked_loss * mask) / torch.sum(mask)

                loss.backward()
                optimizer.step()
                
                run['train/loss'].log(loss.item(), step=global_step)
                
                with torch.no_grad():
                    if torch.sum(mask) > 0:
                        mae = torch.sum(torch.abs(outputs.squeeze(-1) - targets.squeeze(-1)) * mask) / torch.sum(mask)
                        run['train/mae'].log(mae.item(), step=global_step)

                run['misc/epoch'].log(epoch + 1, step=global_step)
                global_step += 1
            
        # --- 7. Validation ---
        if val_q_ids:
            validate_and_log_detailed(
                probe, val_q_ids, data_by_question, target_key, criterion,
                device, tokenizer, run, epoch,
                probe_type=probe_type,
            )
            
    run.stop()
    logging.info("Training finished.")

if __name__ == "__main__":
    fire.Fire(main)
