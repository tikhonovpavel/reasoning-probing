import logging
import sqlite3
import json
from typing import Optional, List, Dict
import numpy as np

import fire
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_json_safe(raw: str, default_val=None) -> Optional[List | Dict]:
    if not raw:
        return default_val
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return default_val


def find_first_change_index(predictions: List[Optional[str]]) -> Optional[int]:
    last_pred = None
    for i, pred in enumerate(predictions):
        if pred and pred in "ABCD":
            if last_pred is None:
                last_pred = pred
            elif pred != last_pred:
                return i
    return None


def run(
    sqlite_db: str,
    model_name: str = "Qwen/Qwen3-32B",
    where_model_path: str = "Qwen/Qwen3-32B",
    system_prompt: str = "Answer only with a letter of a correct choice.",
    head_limit: Optional[int] = None,
    out_dir: str = "analysis_results/critical_chunk_prob_change",
):
    """
    Analyzes the change in probability of the correct answer around the "critical chunk"
    where the model changes its mind for the first and only time (num_changes = 1).
    """
    conn = sqlite3.connect(sqlite_db)
    
    # We need the ground truth correct_answer_letter, so we JOIN with reasoning_traces_qpqa
    # This assumes `reasoning_traces_qpqa` has columns `id` and `correct_answer_letter`.
    sql = """
    SELECT
        m.predictions_json,
        m.letter_probs_json,
        m.num_chunks,
        t.correct_answer_letter
    FROM
        reasoning_trace_forced_solution_metrics AS m
    JOIN
        reasoning_traces_qpqa AS t ON m.trace_id = t.id
    WHERE
        m.num_changes = 1
        AND m.model_name = ?
        AND m.model_path = ?
        AND m.system_prompt = ?
    """
    params = [model_name, where_model_path, system_prompt]

    if head_limit:
        sql += f" LIMIT {int(head_limit)}"

    try:
        df = pd.read_sql_query(sql, conn, params=params)
    except sqlite3.OperationalError as e:
        logger.error(f"Failed to execute SQL query. Did you mean to join with a different table or column? Error: {e}")
        conn.close()
        return
    finally:
        conn.close()

    if df.empty:
        logger.info("No examples with exactly one change found for the specified configuration.")
        return

    analysis_data = []
    for _, row in df.iterrows():
        predictions = parse_json_safe(row['predictions_json'], [])
        letter_probs = parse_json_safe(row['letter_probs_json'], [])
        correct_letter = row['correct_answer_letter']
        num_chunks = int(row['num_chunks']) if pd.notna(row['num_chunks']) else None

        if not predictions or not letter_probs or not correct_letter or not num_chunks:
            continue
        
        k = find_first_change_index(predictions)
        
        if k is None or k == 0:
            continue

        # Find the previous valid prediction and its value
        prev_pred = None
        for i in range(k - 1, -1, -1):
            if predictions[i] and predictions[i] in "ABCD":
                prev_pred = predictions[i]
                break
        
        if prev_pred is None:
            continue # Should not happen if num_changes = 1
            
        crit_pred = predictions[k]
        
        # Ensure probability data is available for both chunks
        if k >= len(letter_probs) or not isinstance(letter_probs[k-1], dict) or not isinstance(letter_probs[k], dict):
            continue

        prob_before_correct = letter_probs[k-1].get(correct_letter, 0.0) or 0.0
        prob_at_correct = letter_probs[k].get(correct_letter, 0.0) or 0.0
        
        # Determine the type of switch
        switch_type = "Other"
        if crit_pred == correct_letter and prev_pred != correct_letter:
            switch_type = "Switch TO Correct"
        elif crit_pred != correct_letter and prev_pred == correct_letter:
            switch_type = "Switch FROM Correct"
        elif crit_pred != correct_letter and prev_pred != correct_letter and crit_pred != prev_pred:
            switch_type = "Switch Incorrect to Incorrect"
            
        analysis_data.append({
            'prob_before': prob_before_correct,
            'prob_at': prob_at_correct,
            'change': prob_at_correct - prob_before_correct,
            'switch_type': switch_type,
            'k': k,
            'num_chunks': num_chunks,
            'k_relative': (k / num_chunks) if num_chunks > 0 else 0,
        })
        
    if not analysis_data:
        logger.info("Could not extract any valid change events from the data.")
        return

    analysis_df = pd.DataFrame(analysis_data)

    analysis_df['chunks_remaining'] = analysis_df['num_chunks'] - 1 - analysis_df['k']

    print("\n--- Analysis of Probability Change for CORRECT Answer (num_changes = 1) ---")
    print(analysis_df.groupby('switch_type')['change'].describe().to_string())
    print("-" * 70)

    print("\n--- Distribution of Critical Chunk Index (Absolute) ---")
    print(analysis_df['k'].describe().to_string())
    print("\n--- Distribution of Critical Chunk Index (Relative) ---")
    print(analysis_df['k_relative'].describe().to_string())
    print("-" * 70)
    
    # --- Plotting ---
    os.makedirs(out_dir, exist_ok=True)

    # Prepare consistent ordering and palette across plots
    preferred_order = [
        'Switch TO Correct',
        'Switch FROM Correct',
        'Switch Incorrect to Incorrect',
        'Other',
    ]
    present_types = analysis_df['switch_type'].dropna().unique().tolist()
    hue_order = [t for t in preferred_order if t in present_types]
    palette_colors = sns.color_palette('colorblind', n_colors=len(hue_order))
    palette_dict = {t: palette_colors[i] for i, t in enumerate(hue_order)}
    
    # Scatter Plot of Before vs. At
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        data=analysis_df,
        x='prob_before',
        y='prob_at',
        hue='switch_type',
        style='switch_type',
        hue_order=hue_order,
        style_order=hue_order,
        palette=palette_dict,
        alpha=0.85,
        s=80,
    )
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='No Change (y=x)')
    plt.title('Probability of Correct Answer: Before vs. At Critical Switch', fontsize=16)
    plt.xlabel('P(Correct Answer) at Chunk k-1 (Before Switch)', fontsize=12)
    plt.ylabel('P(Correct Answer) at Chunk k (At Switch)', fontsize=12)
    plt.grid(True)
    plt.legend(title='Switch Type', fontsize=11)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    
    plot_filename = f"prob_change_scatter_{model_name.replace('/', '_')}.png"
    plot_path = os.path.join(out_dir, plot_filename)
    plt.savefig(plot_path)
    logger.info(f"Scatter plot saved to {plot_path}")
    plt.show()

    # Histogram of the Change
    plt.figure(figsize=(12, 7))
    ax = sns.histplot(
        data=analysis_df,
        x='change',
        hue='switch_type',
        kde=True,
        multiple="layer",
        bins=30,
        hue_order=hue_order,
        palette=palette_dict,
        legend=False,  # We'll build a custom legend with patches
    )
    ax.axvline(0, color='k', linestyle='--', alpha=0.7, label='No Change')
    ax.set_title('Distribution of Probability Change for Correct Answer', fontsize=16)
    ax.set_xlabel('Change in Probability (P_at - P_before)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.grid(True)

    # Build a combined legend: patches for hist groups + line for No Change
    patch_handles = [
        Patch(facecolor=palette_dict[t], edgecolor='black', alpha=0.6, label=t)
        for t in hue_order
    ]
    line_handle = Line2D([0], [0], color='k', linestyle='--', alpha=0.7, label='No Change')
    ax.legend(handles=[*patch_handles, line_handle], title='Switch Type', fontsize=11)

    hist_filename = f"prob_change_hist_{model_name.replace('/', '_')}.png"
    hist_path = os.path.join(out_dir, hist_filename)
    plt.savefig(hist_path)
    logger.info(f"Histogram saved to {hist_path}")
    plt.show()

    # --- New: Histograms of critical index (absolute and relative) ---
    # Absolute index histogram (0-based index k)
    plt.figure(figsize=(12, 6))
    sns.histplot(data=analysis_df, x='k', bins=30, color=palette_colors[0], alpha=0.7)
    plt.title('Distribution of Critical Chunk Index (Absolute, 0-based)', fontsize=16)
    plt.xlabel('Critical Index k', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.grid(True)
    plt.xlim(-0.5, 30.5)
    
    abs_hist_filename = f"critical_index_abs_hist_{model_name.replace('/', '_')}.png"
    abs_hist_path = os.path.join(out_dir, abs_hist_filename)
    plt.savefig(abs_hist_path)
    logger.info(f"Absolute index histogram saved to {abs_hist_path}")
    plt.show()

    # Relative index histogram (k / num_chunks)
    plt.figure(figsize=(12, 6))
    sns.histplot(data=analysis_df, x='k_relative', bins=20, color=palette_colors[1], alpha=0.7)
    plt.title('Distribution of Critical Chunk Index (Relative)', fontsize=16)
    plt.xlabel('Relative Critical Position (k / num_chunks)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xlim(-0.02, 1.02)
    plt.grid(True)
    
    rel_hist_filename = f"critical_index_rel_hist_{model_name.replace('/', '_')}.png"
    rel_hist_path = os.path.join(out_dir, rel_hist_filename)
    plt.savefig(rel_hist_path)
    logger.info(f"Relative index histogram saved to {rel_hist_path}")
    plt.show()

    # --- New: Scatter plot of probability change vs. critical index ---
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
    fig.suptitle('Probability Change vs. Critical Chunk Index', fontsize=20)

    # Subplot 1: Absolute Index
    sns.scatterplot(
        data=analysis_df,
        x='k',
        y='change',
        hue='switch_type',
        style='switch_type',
        hue_order=hue_order,
        style_order=hue_order,
        palette=palette_dict,
        alpha=0.8,
        s=70,
        ax=axes[0]
    )
    axes[0].set_title('vs. Absolute Index (k)', fontsize=16)
    axes[0].set_xlabel('Critical Index k', fontsize=12)
    axes[0].set_ylabel('Change in Probability (P_at - P_before)', fontsize=12)
    axes[0].grid(True)
    axes[0].axhline(0, color='k', linestyle='--', alpha=0.7)
    axes[0].set_xlim(-0.5, 30.5)  # Limit x-axis for absolute index
    axes[0].set_xticks(range(31))
    axes[0].tick_params(axis='x', rotation=90, labelsize=8)
    
    # Grab handles and labels from the first plot, then remove its legend
    handles, labels = axes[0].get_legend_handles_labels()
    if axes[0].get_legend():
        axes[0].get_legend().remove()

    # Subplot 2: Relative Index
    sns.scatterplot(
        data=analysis_df,
        x='k_relative',
        y='change',
        hue='switch_type',
        style='switch_type',
        hue_order=hue_order,
        style_order=hue_order,
        palette=palette_dict,
        alpha=0.8,
        s=70,
        ax=axes[1],
        legend=False  # Don't create a legend for the second plot
    )
    axes[1].set_title('vs. Relative Index (k / num_chunks)', fontsize=16)
    axes[1].set_xlabel('Relative Critical Position', fontsize=12)
    axes[1].set_ylabel('')  # Y-axis is shared
    axes[1].grid(True)
    axes[1].axhline(0, color='k', linestyle='--', alpha=0.7)

    # Create a single legend for the figure, anchored to the right of the second plot
    axes[1].legend(handles, labels, title='Switch Type', bbox_to_anchor=(1.02, 0.5), loc="center left", fontsize=11)

    plt.tight_layout()  # Adjust layout to make space for the legend

    combo_plot_filename = f"prob_change_vs_index_{model_name.replace('/', '_')}.png"
    combo_plot_path = os.path.join(out_dir, combo_plot_filename)
    plt.savefig(combo_plot_path)
    logger.info(f"Probability change vs. index plot saved to {combo_plot_path}")
    plt.show()

    # --- New: Histogram of chunks remaining after critical switch ---
    print("\n--- Distribution of Chunks Remaining After Critical Switch ---")
    print(analysis_df.groupby('switch_type')['chunks_remaining'].describe().to_string())
    print("-" * 70)

    plt.figure(figsize=(14, 8))

    bin_width = 5
    max_val = analysis_df['chunks_remaining'].max()

    ax = sns.histplot(
        data=analysis_df,
        x='chunks_remaining',
        hue='switch_type',
        hue_order=hue_order,
        palette=palette_dict,
        multiple="layer",
        kde=True,
        binwidth=bin_width,
        legend=False,
    )
    ax.set_title('Distribution of Chunks Remaining After Critical Switch', fontsize=16)
    ax.set_xlabel('Number of Chunks Remaining After Critical Switch', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.grid(True)
    
    # Set clear ticks for each bin
    xticks = np.arange(0, max_val + bin_width, bin_width)
    ax.set_xticks(xticks)
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.set_xlim(left=-1, right=max_val + bin_width)

    # Re-use the custom legend logic
    patch_handles = [
        Patch(facecolor=palette_dict.get(t, '#808080'), edgecolor='black', alpha=0.6, label=t)
        for t in hue_order
    ]
    ax.legend(handles=patch_handles, title='Switch Type', fontsize=11)

    remaining_hist_filename = f"chunks_remaining_hist_{model_name.replace('/', '_')}.png"
    remaining_hist_path = os.path.join(out_dir, remaining_hist_filename)
    plt.savefig(remaining_hist_path)
    logger.info(f"Chunks remaining histogram saved to {remaining_hist_path}")
    plt.show()


if __name__ == "__main__":
    fire.Fire(run)
