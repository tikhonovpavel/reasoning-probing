import sqlite3
import json
import logging
from typing import List, Optional, Dict
from tqdm.auto import tqdm
import fire
import sys

# Import the necessary function from the existing script
from precompute_forced_solution_metrics import compute_metrics

# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def backfill_metrics(
    source_table: str,
    sqlite_db: str = "reasoning_traces.sqlite",
    table_name: str = "reasoning_trace_forced_solution_metrics",
    model_name: Optional[str] = None,
):
    """
    Backfills metrics in the database by deriving predictions from `letter_probs_json`.

    This script finds rows where metrics like `first_prediction_index` are NULL
    but `letter_probs_json` is populated. It then calculates the predictions
    by taking the argmax of the letter probabilities for each step and uses
    the existing `compute_metrics` function to calculate and update the
    missing metric fields.
    """
    conn = sqlite3.connect(sqlite_db)
    conn.row_factory = sqlite3.Row  # Makes it easier to access columns by name
    cur = conn.cursor()

    logger.info(f"Connecting to database: {sqlite_db}")

    # --- Build the SELECT query with a JOIN to the source table ---
    sql = f"""
        SELECT
            m.id,
            m.letter_probs_json,
            s.correct_answer_letter
        FROM {table_name} AS m
        JOIN {source_table} AS s ON m.trace_id = s.id
        WHERE m.first_prediction_index IS NULL
          AND m.letter_probs_json IS NOT NULL
          AND m.letter_probs_json != '[]'
          AND m.letter_probs_json != '[{{}}]'
          AND m.source_table = ?
    """
    params = [source_table]
    if model_name:
        sql += " AND LOWER(m.model_name) = LOWER(?)"
        params.append(model_name)

    logger.info("Fetching rows to backfill...")
    cur.execute(sql, params)
    rows_to_update = cur.fetchall()

    if not rows_to_update:
        logger.info("No rows found that require backfilling. Exiting.")
        conn.close()
        return 0

    logger.info(f"Found {len(rows_to_update)} rows to process.")

    update_count = 0
    for row in tqdm(rows_to_update, desc="Backfilling metrics"):
        try:
            letter_probs_list: List[Dict[str, float]] = json.loads(row["letter_probs_json"])
            correct_letter = (row["correct_answer_letter"] or "").strip().upper() or None

            if not isinstance(letter_probs_list, list):
                logger.warning(f"Skipping row id={row['id']} due to invalid JSON type (not a list).")
                continue

            # --- Derive predictions from letter probabilities ---
            predictions: List[Optional[str]] = []
            for probs in letter_probs_list:
                if not probs:  # Handle empty dictionaries like {}
                    predictions.append(None)
                else:
                    # Find the letter with the highest probability
                    prediction = max(probs, key=probs.get)
                    predictions.append(prediction)

            # --- Recalculate metrics using the existing function ---
            metrics = compute_metrics(predictions, correct_letter)

            # --- Update the row in the database ---
            cur.execute(
                f"""
                UPDATE {table_name}
                SET
                    predictions_json = ?,
                    first_prediction_index = ?,
                    first_correct_index = ?,
                    stabilized_index = ?,
                    stabilized_value = ?,
                    num_changes = ?,
                    correct_at_first_chunk = ?,
                    overall_correct = ?
                WHERE id = ?
                """,
                (
                    json.dumps(predictions),
                    metrics["first_prediction_index"],
                    metrics["first_correct_index"],
                    metrics["stabilized_index"],
                    metrics["stabilized_value"],
                    metrics["num_changes"],
                    1 if metrics["correct_at_first_chunk"] else 0,
                    1 if metrics["overall_correct"] else 0,
                    row["id"],
                ),
            )
            update_count += 1

        except json.JSONDecodeError:
            logger.warning(f"Skipping row id={row['id']} due to JSON decoding error.")
        except Exception as e:
            logger.error(f"An unexpected error occurred for row id={row['id']}: {e}")

    conn.commit()
    conn.close()

    logger.info(f"Backfill complete. Updated {update_count} rows.")
    return 0


if __name__ == "__main__":
    sys.exit(fire.Fire(backfill_metrics))
