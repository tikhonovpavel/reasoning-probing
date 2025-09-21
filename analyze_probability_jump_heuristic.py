import logging
import sqlite3
import json
from typing import Optional, List, Tuple, Dict

import fire
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import MaxNLocator
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_json_safe(raw: str, default_val=None) -> Optional[List | Dict]:
    if not raw:
        return default_val
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return default_val


def simulate_jump_heuristic(
    letter_probs: List[Optional[Dict[str, float]]],
    threshold: float
) -> Optional[Tuple[int, str]]:
    """
    Simulates an early stopping heuristic based on a sudden jump in a letter's probability.

    Triggers if P(letter)_k - P(letter)_(k-1) > threshold for any letter.

    Returns (stop_index, predicted_letter) if the rule triggers, otherwise None.
    """
    if not letter_probs or len(letter_probs) < 2:
        return None

    for k in range(1, len(letter_probs)):
        probs_k_minus_1 = letter_probs[k-1]
        probs_k = letter_probs[k]

        if not isinstance(probs_k_minus_1, dict) or not isinstance(probs_k, dict):
            continue

        # Find all letters that jumped above the threshold
        jumped_letters = []
        for letter in "ABCD":
            prob_before = probs_k_minus_1.get(letter, 0.0) or 0.0
            prob_after = probs_k.get(letter, 0.0) or 0.0
            if prob_after - prob_before > threshold:
                jumped_letters.append((letter, prob_after))

        if jumped_letters:
            # If multiple letters jump, pick the one with the highest final probability
            best_letter, _ = max(jumped_letters, key=lambda item: item[1])
            return k, best_letter

    return None


def run(
    sqlite_db: str = "reasoning_traces.sqlite",
    source_table_name: str = "reasoning_traces_gpqa",
    model_name: str = "Qwen/Qwen3-32B",
    where_model_path: str = "Qwen/Qwen3-32B",
    system_prompt: str = "Answer only with a letter of a correct choice.",
    head_limit: Optional[int] = None,
    out_dir: str = "analysis_results/jump_heuristic",
    threshold_step: float = 0.1,
):
    """
    Analyzes an early stopping strategy based on a jump in answer probability.
    """
    logger.info(f"Loading tokenizer for model: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        logger.error(f"Failed to load tokenizer for model '{model_name}'. Please ensure you are logged in if it's a private model. Error: {e}")
        return

    conn = sqlite3.connect(sqlite_db)

    conds = [
        "m.source_table = ?",
        "m.stabilized_index IS NOT NULL",
        "m.model_name = ?",
        "m.model_path = ?",
        "m.system_prompt = ?",
    ]
    params = [source_table_name, model_name, where_model_path, system_prompt]

    where_sql = " AND ".join(conds)
    sql = f"""
        SELECT 
            m.letter_probs_json, 
            m.stabilized_value, 
            m.num_chunks, 
            m.overall_correct, 
            m.predictions_json,
            t.full_prompt_text
        FROM reasoning_trace_forced_solution_metrics as m
        JOIN {source_table_name} AS t ON m.trace_id = t.id AND m.source_table = ?
        WHERE {where_sql}
    """
    # The first parameter in `params` is for the JOIN condition, the rest are for the WHERE clause
    sql_params = [source_table_name] + params
    if head_limit:
        sql += f" LIMIT {int(head_limit)}"

    try:
        df = pd.read_sql_query(sql, conn, params=sql_params)
    finally:
        conn.close()

    if df.empty:
        logger.info("No stabilized examples found for the specified configuration.")
        return

    df['letter_probs'] = df['letter_probs_json'].apply(lambda x: parse_json_safe(x, []))
    df['predictions'] = df['predictions_json'].apply(lambda x: parse_json_safe(x, []))
    df = df.dropna(subset=['stabilized_value'])

    thresholds = np.arange(threshold_step, 1.0, threshold_step)
    results = []

    total_examples = len(df)

    for threshold in thresholds:
        heuristic_results = df['letter_probs'].apply(
            lambda p: simulate_jump_heuristic(p, threshold)
        )
        
        df['k_stop'] = heuristic_results.apply(lambda x: x[0] if x else None)
        df['pred_letter'] = heuristic_results.apply(lambda x: x[1] if x else None)
        
        triggered_df = df.dropna(subset=['k_stop'])
        num_triggered = len(triggered_df)

        if num_triggered == 0:
            logger.info(f"Coverage dropped to 0% at threshold={threshold:.2f}, stopping.")
            break
        
        coverage = (num_triggered / total_examples) * 100 if total_examples > 0 else 0
        
        # This part now only runs if num_triggered > 0
        correct_stops = (triggered_df['pred_letter'] == triggered_df['stabilized_value']).sum()
        precision = (correct_stops / num_triggered) * 100
        
        triggered_df = triggered_df.copy()
        # Calculate savings in chunks
        triggered_df['chunk_saving'] = triggered_df.apply(
            lambda row: row['num_chunks'] - 1 - row['k_stop'],
            axis=1
        )
        avg_chunk_saving = triggered_df['chunk_saving'].mean()

        # --- Calculate Overall Accuracy ---
        not_triggered_df = df[df['k_stop'].isna()]
        correct_fallbacks = not_triggered_df['overall_correct'].sum()
        overall_correct_count = correct_stops + correct_fallbacks
        overall_accuracy = (overall_correct_count / total_examples) * 100

        # Calculate savings in tokens
        def calculate_token_saving(row):
            # This is the "ideal world" calculation as requested.
            # We fetch the original reasoning text, split it into chunks,
            # and then tokenize the chunks that were skipped by the heuristic.
            full_text = row.get('full_prompt_text')
            if not full_text:
                return 0

            # Import helper functions locally to avoid top-level dependency
            from rc_utils import extract_think_content, split_reasoning_chain

            think_content = extract_think_content(full_text)
            if not think_content:
                return 0

            chunks = split_reasoning_chain(think_content)
            if not chunks:
                return 0
            
            k_stop = int(row['k_stop'])
            if k_stop + 1 >= len(chunks):
                return 0 # Heuristic triggered on the last chunk, no savings

            saved_chunks = chunks[k_stop + 1:]
            
            if not saved_chunks:
                return 0
            
            # We tokenize the actual text of the reasoning chunks that were skipped.
            return len(tokenizer.encode("\n\n".join(saved_chunks)))

        triggered_df['token_saving'] = triggered_df.apply(calculate_token_saving, axis=1)
        avg_token_saving = triggered_df['token_saving'].mean()

        # --- New: Calculate Overall Average Savings (the "honest" metric) ---
        overall_avg_chunk_saving = triggered_df['chunk_saving'].sum() / total_examples
        overall_avg_token_saving = triggered_df['token_saving'].sum() / total_examples

        logger.info(
            f"Threshold={threshold:.2f} | "
            f"Coverage={coverage:.2f}% ({num_triggered}/{total_examples}) | "
            f"Precision={precision:.2f}% ({correct_stops}/{num_triggered}) | "
            f"Overall Acc={overall_accuracy:.2f}% | "
            f"Avg Chunk Savings={avg_chunk_saving:.2f} (Overall: {overall_avg_chunk_saving:.2f}) | "
            f"Avg Token Savings={avg_token_saving:.2f} (Overall: {overall_avg_token_saving:.2f})"
        )

        results.append({
            'threshold': threshold,
            'coverage': coverage,
            'precision': precision,
            'overall_accuracy': overall_accuracy,
            'avg_chunk_saving': avg_chunk_saving,
            'avg_token_saving': avg_token_saving,
            'overall_avg_chunk_saving': overall_avg_chunk_saving,
            'overall_avg_token_saving': overall_avg_token_saving,
            'num_triggered': num_triggered,
            'correct_stops': correct_stops
        })

    if not results:
        logger.info("No results to plot.")
        return

    results_df = pd.DataFrame(results)
    
    # --- Find and log the best performing threshold ---
    best_idx = results_df['overall_accuracy'].idxmax()
    best_row = results_df.loc[best_idx]

    logger.info("--- Best Heuristic Performance ---")
    logger.info(
        f"Max Overall Accuracy: {best_row['overall_accuracy']:.2f}% "
        f"achieved at Threshold={best_row['threshold']:.2f}"
    )
    if best_row['num_triggered'] > 0:
        precision_str = f"{best_row['precision']:.2f}% ({int(best_row['correct_stops'])}/{int(best_row['num_triggered'])})"
    else:
        precision_str = "N/A (0 triggered)"
        
    logger.info(f"  - Coverage: {best_row['coverage']:.2f}% ({int(best_row['num_triggered'])}/{total_examples})")
    logger.info(f"  - Precision: {precision_str}")
    logger.info(f"  - Avg Chunk Savings (per trigger): {best_row['avg_chunk_saving']:.2f}")
    logger.info(f"  - Overall Avg Chunk Savings: {best_row['overall_avg_chunk_saving']:.2f}")
    logger.info(f"  - Avg Token Savings (per trigger): {best_row['avg_token_saving']:.2f}")
    logger.info(f"  - Overall Avg Token Savings: {best_row['overall_avg_token_saving']:.2f}")
    logger.info("------------------------------------")
    
    # --- Plotting ---
    os.makedirs(out_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    fig.suptitle('Probability Jump Heuristic: Precision/Coverage/Savings Trade-off', fontsize=20)

    # --- Smartly select points for annotations and markers to avoid clutter ---
    locator = MaxNLocator(nbins=10, prune='both')
    t_min, t_max = results_df['threshold'].min(), results_df['threshold'].max()
    nice_ticks = locator.tick_values(0, 1) if 0 <= t_min and t_max <= 1 else locator.tick_values(t_min, t_max)
    
    ticks_df_indices = sorted(list(set(
        [np.abs(results_df['threshold'] - tick).idxmin() for tick in nice_ticks if t_min <= tick <= t_max]
    )))
    ticks_df = results_df.loc[ticks_df_indices]

    # --- PLOT 1: CHUNKS ---
    ax1 = axes[0]
    ax1.set_title("Trade-off vs. Chunks Saved", fontsize=16)

    # Ensure consistent colors for this axis
    ax1.set_prop_cycle(color=['b', 'g', 'darkorange'])
    
    # Plot smooth lines and markers for selected points
    # ax1.plot(results_df['threshold'], results_df['precision'], color='b', marker='o', markevery=ticks_df_indices, label='Precision (%)')
    # ax1.plot(results_df['threshold'], results_df['coverage'], color='g', marker='o', markevery=ticks_df_indices, label='Coverage (%)')
    ax1.plot(results_df['threshold'], results_df['overall_accuracy'], color='darkorange', linestyle='--', marker='X', markevery=ticks_df_indices, label='Overall Accuracy (%)')

    ax1.set_xlabel('Probability Jump Threshold', fontsize=12)
    ax1.set_ylabel('Percentage (%) [Precision / Coverage / Accuracy]', fontsize=12)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Second Y-Axis for Chunk Savings
    ax1_twin = ax1.twinx()
    # ax1_twin.set_prop_cycle(color=['r', 'sandybrown'])
    ax1_twin.plot(results_df['threshold'], results_df['avg_chunk_saving'], color='r', linestyle='-', marker='s', markevery=ticks_df_indices, label='Avg. Chunks Saved (per trigger)')
    ax1_twin.plot(results_df['threshold'], results_df['overall_avg_chunk_saving'], color='sandybrown', linestyle='--', marker='D', markevery=ticks_df_indices, label='Overall Avg. Chunks Saved')
    ax1_twin.set_ylabel('Average Chunks Saved', color='r', fontsize=12)
    ax1_twin.tick_params(axis='y', labelcolor='r')
    ax1_twin.set_ylim(bottom=0, top=max(1, results_df['avg_chunk_saving'].max() * 1.1))

    # Highlight the best threshold
    ax1.axvline(x=best_row['threshold'], color='k', linestyle=':', linewidth=1.5, alpha=0.9)
    ax1.text(best_row['threshold'] * 1.01, ax1.get_ylim()[1] * 0.95, 'Max Acc.', rotation=90, va='top', ha='center', fontsize=10, backgroundcolor='w')

    # --- PLOT 2: TOKENS ---
    ax2 = axes[1]
    ax2.set_title("Trade-off vs. Tokens Saved", fontsize=16)

    # Ensure consistent colors for this axis
    ax2.set_prop_cycle(color=['b', 'g', 'darkorange'])

    # Plot smooth lines and markers for selected points
    # ax2.plot(results_df['threshold'], results_df['precision'], color='b', marker='o', markevery=ticks_df_indices, label='Precision (%)')
    # ax2.plot(results_df['threshold'], results_df['coverage'], color='g', marker='o', markevery=ticks_df_indices, label='Coverage (%)')
    ax2.plot(results_df['threshold'], results_df['overall_accuracy'], color='darkorange', linestyle='--', marker='X', markevery=ticks_df_indices, label='Overall Accuracy (%)')

    ax2.set_xlabel('Probability Jump Threshold', fontsize=12)
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Second Y-Axis for Token Savings
    ax2_twin = ax2.twinx()
    # ax2_twin.set_prop_cycle(color=['purple', 'plum'])
    ax2_twin.plot(results_df['threshold'], results_df['avg_token_saving'], color='purple', linestyle='-', marker='s', markevery=ticks_df_indices, label='Avg. Tokens Saved (per trigger)')
    ax2_twin.plot(results_df['threshold'], results_df['overall_avg_token_saving'], color='plum', linestyle='--', marker='D', markevery=ticks_df_indices, label='Overall Avg. Tokens Saved')
    ax2_twin.set_ylabel('Average Tokens Saved', color='purple', fontsize=12)
    ax2_twin.tick_params(axis='y', labelcolor='purple')
    ax2_twin.set_ylim(bottom=0, top=max(1, results_df['avg_token_saving'].max() * 1.1))
    
    # Highlight the best threshold
    ax2.axvline(x=best_row['threshold'], color='k', linestyle=':', linewidth=1.5, alpha=0.9)
    ax2.text(best_row['threshold'] * 1.01, ax2.get_ylim()[1] * 0.95, 'Max Acc.', rotation=90, va='top', ha='center', fontsize=10, backgroundcolor='w')
    
    # --- Common settings for both plots ---
    for ax in [ax1, ax2]:
        ax.set_xticks(nice_ticks)
        ax.set_xticklabels([f'{t:.2f}' for t in nice_ticks], rotation=45, ha='right')
        ax.set_xlim(left=results_df['threshold'].min() * 0.95, right=results_df['threshold'].max() * 1.05)

    # --- Create separate legends for each subplot ---
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles1_twin, labels1_twin = ax1_twin.get_legend_handles_labels()
    ax1.legend(handles1 + handles1_twin, labels1 + labels1_twin, loc='lower left')

    handles2, labels2 = ax2.get_legend_handles_labels()
    handles2_twin, labels2_twin = ax2_twin.get_legend_handles_labels()
    ax2.legend(handles2 + handles2_twin, labels2 + labels2_twin, loc='lower left')
    
    fig.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust for suptitle

    plot_filename = f"jump_heuristic_tradeoff_{source_table_name}_{model_name.replace('/', '_')}.png"
    plot_path = os.path.join(out_dir, plot_filename)
    plt.savefig(plot_path)
    logger.info(f"Trade-off plot saved to {plot_path}")
    plt.show()


if __name__ == "__main__":
    fire.Fire(run)
