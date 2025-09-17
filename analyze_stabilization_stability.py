import logging
import sqlite3
from typing import Optional

import fire
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run(
    sqlite_db: str,
    model_name: str = "Qwen/Qwen3-32B",
    where_model_path: str = "Qwen/Qwen3-32B",
    system_prompt: str = "Answer only with a letter of a correct choice.",
    head_limit: Optional[int] = None,
    out_dir: str = "analysis_results/stabilization_stability",
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
        SELECT trace_id, num_changes, overall_correct, stabilized_index, num_chunks,
               continuation_texts_json
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

    # --- Analysis of relative stabilization position for zero-change examples ---
    if count_zero_changes > 0:
        # Calculate relative position, avoiding division by zero
        df_zero_changes = df_zero_changes.copy()
        df_zero_changes['relative_pos'] = df_zero_changes.apply(
            lambda row: row['stabilized_index'] / row['num_chunks'] if row['num_chunks'] > 0 else 0,
            axis=1
        )

        print("\n--- Relative Stabilization Position for All Zero-Change Examples ---")
        print(df_zero_changes['relative_pos'].describe().to_string())
        print("-" * 60)

        # Plotting the histogram for all zero-change examples
        os.makedirs(out_dir, exist_ok=True)
        plt.figure(figsize=(10, 6))
        sns.histplot(df_zero_changes['relative_pos'], bins=20, kde=True)
        plt.title('Distribution of Relative Stabilization Position (num_changes = 0)')
        plt.xlabel('Relative Position (stabilized_index / num_chunks)')
        plt.ylabel('Frequency')
        plt.grid(True)
        
        plot_filename = f"relative_stabilization_pos_hist_all_{model_name.replace('/', '_')}.png"
        plot_path = os.path.join(out_dir, plot_filename)
        plt.savefig(plot_path)
        logger.info(f"Histogram for all zero-change examples saved to {plot_path}")
        plt.show()

        # --- Analysis for zero-change examples that did NOT stabilize at index 0 ---
        df_late_stab = df_zero_changes[df_zero_changes['stabilized_index'] > 0]
        if not df_late_stab.empty:
            print("\n--- Relative Stabilization Position for Zero-Change Examples (stabilized_index > 0) ---")
            print(df_late_stab['relative_pos'].describe().to_string())
            print("-" * 80)

            plt.figure(figsize=(10, 6))
            sns.histplot(df_late_stab['relative_pos'], bins=20, kde=True)
            plt.title('Distribution of Relative Stabilization Position (num_changes = 0 and stabilized_index > 0)')
            plt.xlabel('Relative Position (stabilized_index / num_chunks)')
            plt.ylabel('Frequency')
            plt.grid(True)

            plot_filename_late = f"relative_stabilization_pos_hist_late_{model_name.replace('/', '_')}.png"
            plot_path_late = os.path.join(out_dir, plot_filename_late)
            plt.savefig(plot_path_late)
            logger.info(f"Histogram for late zero-change stabilization saved to {plot_path_late}")
            plt.show()

            # # --- Print continuation texts for these specific traces ---
            # print("\n--- Continuation Texts for Late Zero-Change Stabilization Cases ---")
            # for _, row in df_late_stab.iterrows():
            #     trace_id = row['trace_id']
            #     continuations_raw = row['continuation_texts_json']
            #     try:
            #         continuations = json.loads(continuations_raw) if continuations_raw else []
            #         print(f"\nTrace ID: {trace_id} (stabilized at {row['stabilized_index']} / {row['num_chunks']})")
            #         for i, text in enumerate(continuations):
            #             print(f"  Chunk {i+1}: {repr(text)}")
            #     except Exception:
            #         print(f"\nTrace ID: {trace_id}")
            #         print(f"  Could not parse continuation_texts_json: {continuations_raw}")
            # print("-" * 80)
        else:
            logger.info("No zero-change examples found that stabilized after index 0.")


if __name__ == "__main__":
    fire.Fire(run)
