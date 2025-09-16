import logging
import sqlite3
from typing import Optional

import fire
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run(
    sqlite_db: str,
    model_name: str = "Qwen/Qwen3-32B",
    where_model_path: str = "Qwen/Qwen3-32B",
    system_prompt: str = "Answer only with a letter of a correct choice.",
    head_limit: Optional[int] = None,
):
    """
    Analyzes stabilization stability metrics.

    Calculates the percentage of stabilized examples that stabilized without any changes,
    and among those, the percentage that were correct.
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
        SELECT num_changes, overall_correct
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

    total_stabilized = len(df)
    total_correct = int(df['overall_correct'].sum())
    overall_accuracy = (total_correct / total_stabilized) * 100 if total_stabilized > 0 else 0
    
    df_zero_changes = df[df['num_changes'] == 0]
    count_zero_changes = len(df_zero_changes)

    percent_zero_changes = (count_zero_changes / total_stabilized) * 100 if total_stabilized > 0 else 0

    if count_zero_changes > 0:
        count_correct_zero_changes = int(df_zero_changes['overall_correct'].sum())
        percent_correct_among_zero_changes = (count_correct_zero_changes / count_zero_changes) * 100
    else:
        count_correct_zero_changes = 0
        percent_correct_among_zero_changes = 0

    print("\n--- Stabilization Stability Analysis ---")
    print(f"Model: {model_name}")
    print(f"System Prompt: {system_prompt}")
    print("-" * 40)
    print(f"Total stabilized examples found: {total_stabilized}")
    print(f"Overall accuracy on stabilized: {total_correct} / {total_stabilized} ({overall_accuracy:.2f}%)")
    print("-" * 40)
    print(
        f"Examples with zero changes (stabilized immediately): {count_zero_changes} "
        f"({percent_zero_changes:.2f}%)"
    )
    print(
        f"Correct among zero-change examples: {count_correct_zero_changes} "
        f"({percent_correct_among_zero_changes:.2f}%)"
    )
    print("-" * 40)


if __name__ == "__main__":
    fire.Fire(run)
