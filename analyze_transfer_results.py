import logging
import sqlite3
import pandas as pd
import fire
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_analysis_data(
    conn: sqlite3.Connection,
    transfer_model_path: str,
    original_model_name: str,
    original_model_path: str,
    original_system_prompt: str,
    head_limit: Optional[int],
) -> pd.DataFrame:
    """Loads and joins all necessary data for the analysis."""
    
    query = """
    SELECT
        r.trace_id,
        q.correct_answer_letter,
        m.stabilized_value AS qwen_final_answer,
        r.oss_prediction_at_k_minus_2_letter,
        r.oss_prediction_at_k_minus_1_letter,
        r.oss_prediction_at_k_letter,
        r.oss_prediction_after_continuation_letter,
        r.oss_prediction_after_continuation_from_k_minus_2_letter,
        r.oss_continuation_text,
        r.oss_continuation_text_from_k_minus_2
    FROM
        reasoning_transfer_results AS r
    JOIN
        reasoning_traces_qpqa AS q ON r.trace_id = q.id
    JOIN
        reasoning_trace_forced_solution_metrics AS m ON r.trace_id = m.trace_id
    WHERE
        r.transfer_model_path = ?
        AND m.model_name = ?
        AND m.model_path = ?
        AND m.system_prompt = ?
    """
    
    params = [
        transfer_model_path,
        original_model_name,
        original_model_path,
        original_system_prompt,
    ]
    
    if head_limit:
        query += f" LIMIT {int(head_limit)}"
        
    try:
        df = pd.read_sql_query(query, conn, params=params)
        logger.info(f"Successfully loaded and joined data for {len(df)} traces.")
        return df
    except Exception as e:
        logger.error(f"Failed to load analysis data: {e}")
        return pd.DataFrame()


def calculate_metrics(df: pd.DataFrame, prediction_column: str) -> dict:
    """Calculates agreement and accuracy for a given prediction column."""
    # Drop rows where the prediction is null/empty for accurate metrics
    df_filtered = df.dropna(subset=[prediction_column, 'qwen_final_answer', 'correct_answer_letter'])
    
    total_valid = len(df_filtered)
    if total_valid == 0:
        return {'agreement_with_qwen': float('nan'), 'absolute_accuracy': float('nan'), 'count': 0}

    agreement = (df_filtered[prediction_column] == df_filtered['qwen_final_answer']).sum()
    accuracy = (df_filtered[prediction_column] == df_filtered['correct_answer_letter']).sum()
    
    return {
        'agreement_with_qwen': (agreement / total_valid) * 100,
        'absolute_accuracy': (accuracy / total_valid) * 100,
        'count': total_valid,
    }


def run(
    sqlite_db: str,
    transfer_model_path: str = "openai/gpt-oss-20b",
    original_model_name: str = "Qwen/Qwen3-32B",
    original_model_path: str = "Qwen/Qwen3-32B",
    original_system_prompt: str = "Answer only with a letter of a correct choice.",
    head_limit: Optional[int] = None,
):
    """
    Analyzes the results of the reasoning transfer experiment.
    """
    logger.info(f"Analyzing transfer results for model: {transfer_model_path}")
    
    conn = sqlite3.connect(sqlite_db)
    try:
        df = load_analysis_data(
            conn,
            transfer_model_path,
            original_model_name,
            original_model_path,
            original_system_prompt,
            head_limit,
        )

        if df.empty:
            logger.warning("No data to analyze. Exiting.")
            return

        # --- Main analysis logic ---
        scenarios = {
            "At pre-pre-critical chunk (k-2)": "oss_prediction_at_k_minus_2_letter",
            "At pre-critical chunk (k-1)": "oss_prediction_at_k_minus_1_letter",
            "At critical chunk (k)": "oss_prediction_at_k_letter",
            "After continuation from (k-2)": "oss_prediction_after_continuation_from_k_minus_2_letter",
            "After continuation from (k-1)": "oss_prediction_after_continuation_letter",
        }

        results = []
        for scenario_name, column_name in scenarios.items():
            if column_name in df.columns:
                metrics = calculate_metrics(df, column_name)
                results.append({
                    "Scenario": scenario_name,
                    "Agreement with Qwen (%)": f"{metrics['agreement_with_qwen']:.2f}",
                    "Absolute Accuracy (%)": f"{metrics['absolute_accuracy']:.2f}",
                    "N": metrics['count'],
                })
        
        results_df = pd.DataFrame(results)
        
        print("\n" + "="*80)
        print(" " * 20 + "ANALYSIS OF REASONING TRANSFER RESULTS")
        print("="*80)
        print(results_df.to_string(index=False))
        print("="*80)
        
        # # --- Qualitative analysis for continuation from k-1 ---
        # if 'oss_continuation_text' in df.columns:
        #     continuation_samples = df.dropna(subset=['oss_continuation_text']).head(5)
        #     if not continuation_samples.empty:
        #         print("\n" + "-"*80)
        #         print(" " * 15 + "Examples of Generated Continuations (from k-1)")
        #         print("-"*80)
        #         for _, row in continuation_samples.iterrows():
        #             print(f"\n[Trace ID: {row['trace_id']}]")
        #             print(f"  Continuation from OSS: \"{row['oss_continuation_text'].strip()}\"")
        #             print(f"  Final Answer OSS: {row.get('oss_prediction_after_continuation_letter', 'N/A')} "
        #                   f"(Qwen: {row['qwen_final_answer']}, Correct: {row['correct_answer_letter']})")
        #         print("-"*80)

        # # --- Qualitative analysis for continuation from k-2 ---
        # if 'oss_continuation_text_from_k_minus_2' in df.columns:
        #     continuation_samples_k2 = df.dropna(subset=['oss_continuation_text_from_k_minus_2']).head(5)
        #     if not continuation_samples_k2.empty:
        #         print("\n" + "-"*80)
        #         print(" " * 15 + "Examples of Generated Continuations (from k-2)")
        #         print("-"*80)
        #         for _, row in continuation_samples_k2.iterrows():
        #             print(f"\n[Trace ID: {row['trace_id']}]")
        #             print(f"  Continuation from OSS: \"{row['oss_continuation_text_from_k_minus_2'].strip()}\"")
        #             print(f"  Final Answer OSS: {row.get('oss_prediction_after_continuation_from_k_minus_2_letter', 'N/A')} "
        #                   f"(Qwen: {row['qwen_final_answer']}, Correct: {row['correct_answer_letter']})")
        #         print("-"*80)

    finally:
        conn.close()

    # logger.info("Analysis complete.")


if __name__ == "__main__":
    fire.Fire(run)
