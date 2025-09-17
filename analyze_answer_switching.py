import logging
import sqlite3
import json
from typing import Optional, List

import fire
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_predictions(raw: str) -> List[Optional[str]]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [str(item).strip().upper() if item else None for item in data]
    except (json.JSONDecodeError, TypeError):
        pass
    return []


def find_first_prediction(predictions: List[Optional[str]]) -> Optional[str]:
    for pred in predictions:
        if pred and pred in "ABCD":
            return pred
    return None


def find_first_change_index(predictions: List[Optional[str]]) -> Optional[int]:
    last_pred = None
    for i, pred in enumerate(predictions):
        if pred and pred in "ABCD":
            if last_pred is None:
                last_pred = pred
            elif pred != last_pred:
                return i
    return None


def simulate_cooldown_heuristic(
    predictions: List[Optional[str]],
    cooldown_window: int
) -> Optional[tuple[int, str]]:
    """
    Simulates an early stopping heuristic based on a cooldown period after a change.
    Returns (stop_index, predicted_letter) if the rule triggers, otherwise None.
    """
    if not predictions:
        return None

    has_changed = False
    last_pred = None
    streak = 0
    
    for i, pred in enumerate(predictions):
        if not pred or pred not in "ABCD":
            # Reset streak on nulls but don't reset last_pred
            streak = 0
            continue

        if last_pred is None:
            last_pred = pred
            streak = 1
            continue

        if pred == last_pred:
            streak += 1
        else:
            # A change occurred
            has_changed = True
            last_pred = pred
            streak = 1
        
        if has_changed and streak >= cooldown_window:
            return i, last_pred
            
    return None


def run(
    sqlite_db: str,
    model_name: str = "Qwen/Qwen3-32B",
    where_model_path: str = "Qwen/Qwen3-32B",
    system_prompt: str = "Answer only with a letter of a correct choice.",
    head_limit: Optional[int] = None,
    out_dir: str = "analysis_results/answer_switching",
    cooldown_window: int = 3,
):
    """
    Analyzes answer switching behavior for examples with num_changes > 0.
    """
    conn = sqlite3.connect(sqlite_db)
    
    conds = [
        "num_changes > 0",
        "stabilized_value IS NOT NULL",
        "model_name = ?",
        "model_path = ?",
        "system_prompt = ?",
    ]
    params = [model_name, where_model_path, system_prompt]

    where_sql = " AND ".join(conds)
    sql = f"""
        SELECT predictions_json, stabilized_value, num_chunks, num_changes, overall_correct
        FROM reasoning_trace_forced_solution_metrics
        WHERE {where_sql}
    """
    if head_limit:
        sql += f" LIMIT {int(head_limit)}"

    try:
        df = pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()

    if df.empty:
        logger.info("No examples with answer changes found for the specified configuration.")
        return

    df['predictions'] = df['predictions_json'].apply(parse_predictions)
    df['initial_pred'] = df['predictions'].apply(find_first_prediction)
    df = df.dropna(subset=['initial_pred', 'stabilized_value'])

    # --- Analysis 1: Transition Matrix ---
    transition_matrix = pd.crosstab(df['initial_pred'], df['stabilized_value'], dropna=False)
    
    print("\n--- Answer Switching Transition Matrix (Initial -> Final Stabilized) ---")
    print("Rows: Initial Prediction, Columns: Final Stabilized Prediction")
    print(transition_matrix.to_string())
    print("-" * 70)

    # --- Analysis of accuracy based on number of changes ---
    df_one_change = df[df['num_changes'] == 1]
    df_many_changes = df[df['num_changes'] > 1]

    count_one_change = len(df_one_change)
    accuracy_one_change = (df_one_change['overall_correct'].sum() / count_one_change) * 100 if count_one_change > 0 else 0

    count_many_changes = len(df_many_changes)
    accuracy_many_changes = (df_many_changes['overall_correct'].sum() / count_many_changes) * 100 if count_many_changes > 0 else 0

    print("\n--- Accuracy by Number of Answer Changes ---")
    print(f"Total examples with changes: {len(df)}")
    print(f"  - Examples with exactly ONE change (num_changes = 1): {count_one_change}")
    print(f"    - Accuracy for ONE change cases: {df_one_change['overall_correct'].sum()} / {count_one_change} ({accuracy_one_change:.2f}%)")
    print(f"  - Examples with MULTIPLE changes (num_changes > 1): {count_many_changes}")
    print(f"    - Accuracy for MULTIPLE change cases: {df_many_changes['overall_correct'].sum()} / {count_many_changes} ({accuracy_many_changes:.2f}%)")
    print("-" * 70)

    # --- Analysis 3: Cooldown Heuristic Simulation ---
    heuristic_results = df['predictions'].apply(lambda p: simulate_cooldown_heuristic(p, cooldown_window))
    df['k_stop'] = heuristic_results.apply(lambda x: x[0] if x else None)
    df['pred_letter'] = heuristic_results.apply(lambda x: x[1] if x else None)

    triggered_df = df.dropna(subset=['k_stop'])
    
    num_triggered = len(triggered_df)
    total_eligible = len(df)
    coverage = (num_triggered / total_eligible) * 100 if total_eligible > 0 else 0

    if num_triggered > 0:
        correct_stops = (triggered_df['pred_letter'] == triggered_df['stabilized_value']).sum()
        precision = (correct_stops / num_triggered) * 100
        
        # Savings calculation
        triggered_df = triggered_df.copy()
        triggered_df['saving'] = triggered_df.apply(
            lambda row: row['num_chunks'] - 1 - row['k_stop'],
            axis=1
        )
        avg_saving = triggered_df['saving'].mean()
    else:
        precision = 0
        avg_saving = 0

    print(f"\n--- Cooldown Heuristic Simulation (window = {cooldown_window}) ---")
    print(f"Heuristic triggered on: {num_triggered} / {total_eligible} examples ({coverage:.2f}%)")
    print(f"Precision of triggered stops (vs stabilized value): {correct_stops} / {num_triggered} ({precision:.2f}%)")
    print(f"Average computation saving on triggered examples: {avg_saving:.2f} chunks")
    print("-" * 70)


    # --- Analysis 2: Position of First Change ---
    df['first_change_index'] = df['predictions'].apply(find_first_change_index)
    df = df.dropna(subset=['first_change_index'])
    
    df['relative_change_pos'] = df.apply(
        lambda row: row['first_change_index'] / row['num_chunks'] if row['num_chunks'] > 0 else 0,
        axis=1
    )

    print("\n--- Distribution of First Answer Change Position ---")
    print(df['relative_change_pos'].describe().to_string())
    print("-" * 60)

    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.histplot(df['relative_change_pos'], bins=20, kde=True)
    plt.title('Distribution of Relative Position of First Answer Change')
    plt.xlabel('Relative Position (first_change_index / num_chunks)')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    plot_filename = f"relative_change_pos_hist_{model_name.replace('/', '_')}.png"
    plot_path = os.path.join(out_dir, plot_filename)
    plt.savefig(plot_path)
    logger.info(f"Histogram saved to {plot_path}")
    plt.show()


if __name__ == "__main__":
    fire.Fire(run)
