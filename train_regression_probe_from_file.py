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
from transformers import AutoTokenizer
from tqdm import tqdm
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ProbeLinearRegressor(nn.Module):
    """A linear probe for regression, predicting a single value from hidden states."""
    def __init__(self, hidden_size: int, device: str, dtype: torch.dtype):
        super().__init__()
        self.regressor = nn.Linear(hidden_size, 1).to(device=device, dtype=dtype)

    def forward(self, hidden_states: torch.Tensor):
        """
        Args:
            hidden_states (torch.Tensor): Shape (batch_size, hidden_size) or (seq_len, hidden_size).
        Returns:
            torch.Tensor: Predicted values, shape (batch_size, 1) or (seq_len, 1).
        """
        return self.regressor(hidden_states)

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

class ProbeLstmRegressor(nn.Module):
    """An LSTM-based probe for regression on sequences of hidden states."""
    def __init__(
        self,
        input_size: int,
        lstm_hidden_size: int,
        dropout_rate: float,
        device: str,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True,
        ).to(device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout_rate)
        self.regressor = nn.Linear(lstm_hidden_size, 1).to(device=device, dtype=dtype)

    def forward(self, hidden_states: torch.Tensor):
        is_batched = hidden_states.dim() == 3
        if not is_batched:
            hidden_states = hidden_states.unsqueeze(0)

        lstm_out, _ = self.lstm(hidden_states)
        lstm_out = self.dropout(lstm_out)
        predictions = self.regressor(lstm_out)
        
        return predictions if is_batched else predictions.squeeze(0)

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
        x = x + self.pe.permute(1, 0, 2)[:, :x.size(1), :]
        return self.dropout(x)

class ProbeAttentionRegressor(nn.Module):
    """An Attention-based probe for regression."""
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
        
        self.projection = nn.Linear(input_size, self.projection_dim).to(device=device, dtype=dtype) if projection_dim else nn.Identity()
        assert self.projection_dim % num_heads == 0, "projection_dim must be divisible by num_heads"

        self.pos_encoder = PositionalEncoding(d_model=self.projection_dim, dropout=dropout_rate).to(device=device, dtype=dtype)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.projection_dim, num_heads=num_heads, dropout=dropout_rate, batch_first=True
        ).to(device=device, dtype=dtype)
        self.layer_norm = nn.LayerNorm(self.projection_dim).to(device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout_rate)
        self.regressor = nn.Linear(self.projection_dim, 1).to(device=device, dtype=dtype)

    def forward(self, hidden_states: torch.Tensor, key_padding_mask: torch.Tensor = None):
        is_batched = hidden_states.dim() == 3
        if not is_batched:
            hidden_states = hidden_states.unsqueeze(0)

        projected_hiddens = self.projection(hidden_states)
        hiddens_with_pos = self.pos_encoder(projected_hiddens)

        seq_len = hidden_states.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device=self.device, dtype=self.dtype)

        attn_output, _ = self.attention(
            query=hiddens_with_pos, key=hiddens_with_pos, value=hiddens_with_pos,
            key_padding_mask=key_padding_mask, attn_mask=causal_mask, need_weights=False
        )
        norm_output = self.layer_norm(projected_hiddens + self.dropout(attn_output))
        predictions = self.regressor(norm_output)
        
        return predictions if is_batched else predictions.squeeze(0)

import matplotlib.pyplot as plt
import seaborn as sns
from neptune.types import File
import glob

def validate_and_log_detailed(
    probe,
    val_q_ids,
    data_by_question,
    target_key,
    criterion,
    device,
    run,
    epoch,
    probe_type: str,
    target_mean: Optional[float] = None,
    target_std: Optional[float] = None,
):
    """
    Runs detailed validation for the regression probe, calculating loss,
    and logging scatter plots and prediction dynamics.
    """
    probe.eval()
    standardize = target_mean is not None and target_std is not None
    all_targets = []
    all_preds = []
    total_val_loss = 0
    total_tokens = 0

    # For prediction dynamics plot
    num_bins = 20 
    predictions_binned = defaultdict(list)
    targets_binned = defaultdict(list)
    
    with torch.no_grad():
        for q_id in tqdm(val_q_ids, desc="Running detailed validation"):
            for sample in data_by_question[q_id]:
                hiddens = sample['hiddens'].to(device)
                target_value = sample[target_key]
                
                if hiddens.shape[0] == 0:
                    continue
                    
                seq_len = hiddens.shape[0]

                if probe_type in ['linear', 'mlp']:
                    preds_seq = probe(hiddens).squeeze(-1) # (seq_len)
                else: # LSTM or Attention
                    preds_seq = probe(hiddens, key_padding_mask=None).squeeze(-1)

                if standardize:
                    target_tensor_seq = torch.tensor([(target_value - target_mean) / target_std] * seq_len, device=device, dtype=preds_seq.dtype)
                else:
                    target_tensor_seq = torch.tensor([target_value] * seq_len, device=device, dtype=preds_seq.dtype)
                
                loss = criterion(preds_seq, target_tensor_seq)
                total_val_loss += loss.sum().item()
                total_tokens += seq_len
                
                if standardize:
                    preds_list = (preds_seq * target_std + target_mean).tolist()
                else:
                    preds_list = preds_seq.tolist()

                all_preds.extend(preds_list)
                all_targets.extend([target_value] * seq_len)
                
                # For dynamics plot
                for i in range(seq_len):
                    relative_position = i / seq_len if seq_len > 1 else 0
                    bin_index = int(relative_position * num_bins)
                    predictions_binned[bin_index].append(preds_list[i])
                    targets_binned[bin_index].append(target_value)

    # --- Log Main Metrics ---
    avg_val_loss = total_val_loss / total_tokens if total_tokens > 0 else 0
    mae = np.mean(np.abs(np.array(all_preds) - np.array(all_targets))) if all_targets else 0
    
    logging.info(f"Epoch {epoch+1}: Val MSE Loss: {avg_val_loss:.4f}, Val MAE: {mae:.4f}")
    run['val/mse_loss'].log(avg_val_loss, step=epoch + 1)
    run['val/mae'].log(mae, step=epoch + 1)

    # --- Scatter Plot of Predictions vs. Targets ---
    if all_targets:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(all_targets, all_preds, alpha=0.3)
        ax.set_xlabel("True Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title(f"Predictions vs. True Values - Epoch {epoch+1}")
        # Add a y=x line for reference
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0)
        ax.set_aspect('equal', adjustable='box')
        run["val/predictions_scatter"].log(File.as_image(fig), step=epoch + 1)
        plt.close(fig)

    # --- Prediction Dynamics Plot ---
    fig_dyn, ax_dyn = plt.subplots(figsize=(12, 6))
    bins = range(num_bins)
    x_axis = [(i + 0.5) / num_bins for i in bins]
    
    mean_preds = [np.mean(predictions_binned[i]) if predictions_binned[i] else np.nan for i in bins]
    mean_targets = [np.mean(targets_binned[i]) if targets_binned[i] else np.nan for i in bins]
    
    ax_dyn.plot(x_axis, mean_preds, label='Avg. Prediction', marker='o')
    ax_dyn.plot(x_axis, mean_targets, label='Avg. True Value', marker='x', linestyle='--')
    
    ax_dyn.set_xlabel("Relative Position in Reasoning Trace")
    ax_dyn.set_ylabel("Value")
    ax_dyn.set_title(f"Prediction Dynamics - Epoch {epoch+1}")
    ax_dyn.legend()
    ax_dyn.grid(True)
    plt.tight_layout()
    
    run["val/prediction_dynamics"].log(File.as_image(fig_dyn), step=epoch + 1)
    plt.close(fig_dyn)

    return mae

def main(
    probe_data_path: str = "probe_data/layer_45",
    model_name: str = "Qwen/QwQ-32B",
    target_type: str = "ground_truth",
    probe_type: str = "linear",
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    epochs: int = 100,
    val_split_size: float = 0.2,
    neptune_project: str = "probing-reasoning-regressor",
    # MLP/LSTM/Attention-specific parameters
    mlp_hidden_dim: int = 16,
    lstm_hidden_size: int = 256,
    attention_heads: int = 4,
    attention_projection_dim: Optional[int] = 128,
    dropout_rate: float = 0.1,
    # Optimizer parameters
    weight_decay: float = 0.01,
    # New filtering parameters
    filter_percentile_lower: float = 5.0,
    filter_percentile_upper: float = 95.0,
    standardize_targets: bool = True,
):
    """
    Trains a regression probe from a pre-generated dataset of hidden states.

    Args:
        probe_data_path (str): Path to the DIRECTORY containing probe data chunks.
        model_name (str): Name of the HF model (for config).
        target_type (str): What to predict: 'ground_truth' or 'model_answer'.
        probe_type (str): Type of probe: 'linear', 'mlp', 'lstm', or 'attention'.
        ...
        filter_percentile_lower (float): Lower percentile to filter target values.
        filter_percentile_upper (float): Upper percentile to filter target values.
        standardize_targets (bool): Whether to standardize the target values after filtering.
    """
    assert target_type in ['ground_truth', 'model_answer'], "target_type must be 'ground_truth' or 'model_answer'"
    assert probe_type in ['linear', 'mlp', 'lstm', 'attention'], "probe_type must be 'linear', 'mlp', 'lstm', or 'attention'"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # --- 1. Initialize Neptune ---
    run = neptune.init_run(
        project=neptune_project,
        tags=["train_regression", target_type, probe_type, os.path.basename(probe_data_path)]
    )
    run["parameters"] = {k: v for k, v in locals().items() if isinstance(v, (str, int, float, bool))}

    # --- Create directory for saving probes ---
    run_id = run["sys/id"].fetch()
    save_dir = os.path.join("saved_probes", os.path.basename(probe_data_path).replace('/', '_'), probe_type, run_id)
    os.makedirs(save_dir, exist_ok=True)
    logging.info(f"Probe weights will be saved to: {save_dir}")

    # --- 2. Load and prepare data ---
    model_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    target_key = 'ground_truth_value' if target_type == 'ground_truth' else 'model_answer_value'

    logging.info(f"Searching for probe data chunks in {probe_data_path}...")
    chunk_files = sorted(glob.glob(os.path.join(probe_data_path, "chunk_*.pt")))
    if not chunk_files:
        raise FileNotFoundError(f"No chunk files found in {probe_data_path}")
    
    logging.info(f"Found {len(chunk_files)} chunk files.")

    # --- Calculate Percentiles on the ENTIRE dataset ---
    logging.info("Calculating target value percentiles across all data...")
    all_target_values = []
    for chunk_file in tqdm(chunk_files, desc="Loading all targets for percentile calculation"):
        chunk_data = torch.load(chunk_file)
        for sample in chunk_data:
            all_target_values.append(sample[target_key])
            
    lower_bound = np.percentile(all_target_values, filter_percentile_lower)
    upper_bound = np.percentile(all_target_values, filter_percentile_upper)
    logging.info(f"Filtering targets between P{filter_percentile_lower} ({lower_bound:.2f}) and P{filter_percentile_upper} ({upper_bound:.2f}).")

    # Split chunk files for train and validation
    random.Random(42).shuffle(chunk_files)
    split_idx = int(len(chunk_files) * (1 - val_split_size))
    train_chunk_files, val_chunk_files = chunk_files[:split_idx], chunk_files[split_idx:]
    
    # Calculate mean and std on the filtered TRAINING data for standardization
    target_mean, target_std = None, None
    if standardize_targets:
        logging.info("Calculating mean and std for standardization on filtered training data...")
        filtered_train_targets = []
        for chunk_file in tqdm(train_chunk_files, desc="Loading filtered train targets for standardization"):
            chunk_data = torch.load(chunk_file)
            for sample in chunk_data:
                target_val = sample[target_key]
                if lower_bound <= target_val <= upper_bound:
                    filtered_train_targets.append(target_val)
        
        if filtered_train_targets:
            target_mean = np.mean(filtered_train_targets)
            target_std = np.std(filtered_train_targets)
            if target_std < 1e-6: # Avoid division by zero or near-zero
                target_std = 1.0
                logging.warning("Target standard deviation is close to 0. Standardization will only subtract the mean.")
            logging.info(f"Standardizing targets with mean={target_mean:.4f} and std={target_std:.4f}")
            run["parameters/target_mean"] = target_mean
            run["parameters/target_std"] = target_std
        else:
            logging.warning("No training data left after filtering. Cannot compute mean/std.")
            standardize_targets = False # Disable if no data

    # Infer hidden_size from the first chunk
    try:
        first_chunk = torch.load(chunk_files[0])
        hidden_size = first_chunk[0]['hiddens'].shape[-1]
        logging.info(f"Inferred hidden size: {hidden_size}")
    except (IndexError, KeyError) as e:
        raise ValueError(f"Could not infer hidden size from the first chunk '{chunk_files[0]}': {e}")
    
    # We need all val data in memory for detailed logging
    val_data_by_question = defaultdict(list)
    if val_chunk_files:
        for chunk_file in tqdm(val_chunk_files, desc="Loading and filtering validation data"):
            chunk_data = torch.load(chunk_file)
            for sample in chunk_data:
                # Filter validation data based on the calculated bounds
                if lower_bound <= sample[target_key] <= upper_bound:
                    val_data_by_question[sample['question_id']].append(sample)
    val_q_ids = list(val_data_by_question.keys())

    class IterableChunkedDataset(torch.utils.data.IterableDataset):
        """An iterable dataset that loads data chunk by chunk and filters on the fly."""
        def __init__(self, chunk_files, target_key, is_sequence: bool, lower_bound: float, upper_bound: float, shuffle_chunks: bool = True, target_mean: Optional[float] = None, target_std: Optional[float] = None):
            super().__init__()
            self.chunk_files = chunk_files
            self.target_key = target_key
            self.is_sequence = is_sequence
            self.lower_bound = lower_bound
            self.upper_bound = upper_bound
            self.shuffle_chunks = shuffle_chunks
            self.target_mean = target_mean
            self.target_std = target_std
            self.standardize = self.target_mean is not None and self.target_std is not None

        def __iter__(self):
            # Shuffle chunks at the beginning of each epoch
            if self.shuffle_chunks:
                random.shuffle(self.chunk_files)
            
            for chunk_file in self.chunk_files:
                chunk_data = torch.load(chunk_file)
                for sample in chunk_data:
                    hiddens = sample['hiddens']
                    target = sample[self.target_key]

                    if not (self.lower_bound <= target <= self.upper_bound):
                        continue

                    if self.standardize:
                        target = (target - self.target_mean) / self.target_std

                    if hiddens.shape[0] == 0:
                        continue
                    
                    if self.is_sequence:
                        yield {'hiddens': hiddens, 'target': target}
                    else: # Flat tensor for linear probe
                        for i in range(hiddens.shape[0]):
                            yield hiddens[i], target

    def collate_fn_for_linear(batch, model_dtype):
        # batch is a list of (hidden_tensor, target_float)
        hiddens = torch.stack([item[0] for item in batch], dim=0)
        targets = torch.tensor([item[1] for item in batch], dtype=model_dtype)
        return hiddens, targets

    def collate_fn_for_sequences(batch):
        hiddens_list = [item['hiddens'] for item in batch]
        targets_list = [item['target'] for item in batch]
        lengths = torch.tensor([len(h) for h in hiddens_list])
        hiddens_padded = pad_sequence(hiddens_list, batch_first=True, padding_value=0.0)
        targets_padded = torch.full_like(hiddens_padded[:, :, 0], fill_value=-100.0, dtype=model_dtype)
        for i, target in enumerate(targets_list):
            targets_padded[i, :lengths[i]] = target
        return hiddens_padded, targets_padded

    logging.info("Creating train and validation datasets...")
    is_sequence_probe = probe_type in ['lstm', 'attention']
    
    train_dataset = IterableChunkedDataset(
        train_chunk_files, target_key, is_sequence=is_sequence_probe, 
        lower_bound=lower_bound, upper_bound=upper_bound,
        target_mean=target_mean if standardize_targets else None,
        target_std=target_std if standardize_targets else None,
    )

    if is_sequence_probe:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn_for_sequences)
    else:
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            collate_fn=lambda batch: collate_fn_for_linear(batch, model_dtype)
        )

    # --- 4. Initialize Probe and Optimizer ---
    if probe_type == 'linear':
        probe = ProbeLinearRegressor(hidden_size=hidden_size, device=device, dtype=model_dtype)
    elif probe_type == 'mlp':
        probe = ProbeMlpRegressor(hidden_size=hidden_size, mlp_hidden_dim=mlp_hidden_dim, device=device, dtype=model_dtype)
    elif probe_type == 'lstm':
        probe = ProbeLstmRegressor(input_size=hidden_size, lstm_hidden_size=lstm_hidden_size, dropout_rate=dropout_rate, device=device, dtype=model_dtype)
    elif probe_type == 'attention':
        probe = ProbeAttentionRegressor(input_size=hidden_size, num_heads=attention_heads, dropout_rate=dropout_rate, device=device, dtype=model_dtype, projection_dim=attention_projection_dim)

    optimizer = torch.optim.AdamW(probe.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss(reduction='none') # Use 'none' for sequence masking

    # --- 5. Pre-training Validation ---
    if val_q_ids:
        logging.info("Running pre-training validation...")
        validate_and_log_detailed(
            probe, val_q_ids, val_data_by_question, target_key, criterion, device, run, epoch=-1, probe_type=probe_type,
            target_mean=target_mean if standardize_targets else None,
            target_std=target_std if standardize_targets else None,
        )

    # --- 6. Training Loop ---
    logging.info("Starting training...")
    global_step = 0
    best_val_mae = float('inf')
    for epoch in range(epochs):
        probe.train()
        total_train_loss = 0
        total_tokens = 0
        
        if not is_sequence_probe:
            for hiddens, targets in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                optimizer.zero_grad()
                hiddens, targets = hiddens.to(device), targets.to(device)
                preds = probe(hiddens).squeeze(-1)
                loss = criterion(preds, targets).mean()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    mae = torch.abs(preds - targets).mean().item()
                run['train/mse_loss'].log(loss.item(), step=global_step)
                run['train/mae'].log(mae, step=global_step)
                global_step += 1
        else: # lstm or attention
            for hiddens, targets in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} ({probe_type.upper()})"):
                optimizer.zero_grad()
                hiddens, targets = hiddens.to(device), targets.to(device)
                
                mask = targets != -100.0
                if probe_type == 'attention':
                    padding_mask = (targets == -100.0)[:, :, 0] if targets.dim() == 3 else (targets == -100.0)
                    preds = probe(hiddens, key_padding_mask=padding_mask).squeeze(-1)
                else: # lstm
                    preds = probe(hiddens).squeeze(-1)
                
                loss_tensor = criterion(preds, targets)
                loss = torch.sum(loss_tensor * mask) / torch.sum(mask)
                
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    mae = (torch.sum(torch.abs(preds - targets) * mask) / torch.sum(mask)).item()
                
                run['train/mse_loss'].log(loss.item(), step=global_step)
                run['train/mae'].log(mae, step=global_step)
                global_step += 1
            
        # --- 7. Validation ---
        if val_q_ids:
            val_mae = validate_and_log_detailed(
                probe, val_q_ids, val_data_by_question, target_key, criterion, device, run, epoch, probe_type=probe_type,
                target_mean=target_mean if standardize_targets else None,
                target_std=target_std if standardize_targets else None,
            )
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                save_path = os.path.join(save_dir, "best_probe.pt")
                torch.save(probe.state_dict(), save_path)
                logging.info(f"New best probe saved with Val MAE: {best_val_mae:.4f} at {save_path}")

    # --- 8. Save final model and upload artifacts ---
    final_save_path = os.path.join(save_dir, "last_epoch_probe.pt")
    torch.save(probe.state_dict(), final_save_path)
    logging.info(f"Saved final epoch probe to {final_save_path}")
    
    run.stop()
    logging.info("Training finished.")

if __name__ == "__main__":
    fire.Fire(main)
