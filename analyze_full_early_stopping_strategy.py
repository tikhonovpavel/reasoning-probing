import logging
import sqlite3
import json
from typing import Optional, List, Tuple

import fire
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_predictions(raw: str) -> List[Optional[str]]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [str(item).strip().upper() if item and str(item).strip() else None for item in data]
    except (json.JSONDecodeError, TypeError):
        pass
    return []


def simulate_unified_heuristic(
    predictions: List[Optional[str]],
    cooldown_window: int
) -> Optional[Tuple[int, str]]:
    """
    Simulates a unified early stopping heuristic for ALL traces.
    Returns (stop_index, predicted_letter) if a rule triggers.
    The type of rule is determined by the caller based on num_changes.
    Nulls do not increment the streak, but do not reset it either.
    """
    if not predictions:
        return None

    last_pred = None
    streak = 0
    
    for i, pred in enumerate(predictions):
        if not pred or pred not in "ABCD":
            # On null, we do nothing to the streak. It's a pause.
            continue

        if last_pred is None:
            # First valid prediction
            last_pred = pred
            streak = 1
        elif pred == last_pred:
            streak += 1
        else:
            # A change occurred
            last_pred = pred
            streak = 1
        
        if streak >= cooldown_window:
            return i, last_pred
            
    return None


def run(
    sqlite_db: str,
    model_name: str = "Qwen/Qwen3-32B",
    where_model_path: str = "Qwen/Qwen3-32B",
    system_prompt: str = "Answer only with a letter of a correct choice.",
    head_limit: Optional[int] = None,
    windows: str = "3,4,5,6,7",
):
    """
    Analyzes a unified early stopping strategy on ALL stabilized examples.
    """
    conn = sqlite3.connect(sqlite_db)
    
    conds = [
        "stabilized_index IS NOT NULL",
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
        logger.info("No stabilized examples found for the specified configuration.")
        return

    df['predictions'] = df['predictions_json'].apply(parse_predictions)
    
    window_list = [int(w.strip()) for w in windows.split(',') if w.strip()]

    for window in window_list:
        heuristic_results = df['predictions'].apply(lambda p: simulate_unified_heuristic(p, window))
        
        df['k_stop'] = heuristic_results.apply(lambda x: x[0] if x else None)
        df['pred_letter'] = heuristic_results.apply(lambda x: x[1] if x else None)
        
        # Determine rule based on original num_changes
        df['rule_triggered'] = df.apply(
            lambda row: 'Initial Stability' if row['num_changes'] == 0 else ('Post-Change Stability' if row['num_changes'] > 0 else None),
            axis=1
        )
        # We only care about the rule if it was actually triggered
        df.loc[df['k_stop'].isna(), 'rule_triggered'] = None


        # Overall metrics
        triggered_df = df.dropna(subset=['k_stop'])
        num_total = len(df)
        num_triggered = len(triggered_df)
        
        coverage_overall = (num_triggered / num_total) * 100 if num_total > 0 else 0
        
        if num_triggered > 0:
            correct_stops_overall = (triggered_df['pred_letter'] == triggered_df['stabilized_value']).sum()
            precision_overall = (correct_stops_overall / num_triggered) * 100
            
            triggered_df = triggered_df.copy()
            triggered_df['saving'] = triggered_df.apply(
                lambda row: row['num_chunks'] - 1 - row['k_stop'],
                axis=1
            )
            avg_saving_overall = triggered_df['saving'].mean()
        else:
            correct_stops_overall = 0
            precision_overall = 0
            avg_saving_overall = 0

        # Metrics for 'Initial Stability'
        initial_stab_df = triggered_df[triggered_df['rule_triggered'] == 'Initial Stability']
        num_initial_eligible = len(df[df['num_changes'] == 0])
        num_initial_triggered = len(initial_stab_df)
        coverage_initial = (num_initial_triggered / num_initial_eligible) * 100 if num_initial_eligible > 0 else 0
        if num_initial_triggered > 0:
            correct_initial = (initial_stab_df['pred_letter'] == initial_stab_df['stabilized_value']).sum()
            precision_initial = (correct_initial / num_initial_triggered) * 100
            avg_saving_initial = initial_stab_df['saving'].mean()
        else:
            correct_initial = 0
            precision_initial = 0
            avg_saving_initial = 0

        # Metrics for 'Post-Change Stability'
        post_change_df = triggered_df[triggered_df['rule_triggered'] == 'Post-Change Stability']
        num_change_eligible = len(df[df['num_changes'] > 0])
        num_post_change_triggered = len(post_change_df)
        coverage_post_change = (num_post_change_triggered / num_change_eligible) * 100 if num_change_eligible > 0 else 0
        if num_post_change_triggered > 0:
            correct_post_change = (post_change_df['pred_letter'] == post_change_df['stabilized_value']).sum()
            precision_post_change = (correct_post_change / num_post_change_triggered) * 100
            avg_saving_post_change = post_change_df['saving'].mean()
        else:
            correct_post_change = 0
            precision_post_change = 0
            avg_saving_post_change = 0

        # --- Print Results Table ---
        header = f"--- Heuristic Simulation (Window = {window}) on ALL {num_total} Stabilized Examples ---"
        print("\n" + header)
        
        table_data = {
            "Metric": ["Coverage", "Precision", "Avg. Savings"],
            "Overall": [
                f"{num_triggered}/{num_total} ({coverage_overall:.2f}%)",
                f"{correct_stops_overall}/{num_triggered} ({precision_overall:.2f}%)",
                f"{avg_saving_overall:.2f}"
            ],
            "Rule: Initial Stability": [
                f"{num_initial_triggered}/{num_initial_eligible} ({coverage_initial:.2f}%)",
                f"{correct_initial}/{num_initial_triggered} ({precision_initial:.2f}%)",
                f"{avg_saving_initial:.2f}"
            ],
            "Rule: Post-Change Stability": [
                f"{num_post_change_triggered}/{num_change_eligible} ({coverage_post_change:.2f}%)",
                f"{correct_post_change}/{num_post_change_triggered} ({precision_post_change:.2f}%)",
                f"{avg_saving_post_change:.2f}"
            ]
        }
        results_df = pd.DataFrame(table_data)
        print(results_df.to_string(index=False))
        print("-" * len(header))


if __name__ == "__main__":
    fire.Fire(run)
