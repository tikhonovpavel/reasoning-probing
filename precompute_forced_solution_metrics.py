import json
import logging
import re
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from rc_utils import (
    QWEN3_SPECIAL_STOPPING_PROMPT,
    split_reasoning_chain,
    extract_think_content,
    compose_user_prompt,
)
import fire


# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Constants & Types
# ----------------------------------------------------------------------------
# moved to rc_utils


@dataclass
class RowItem:
    id: int
    model_path: str
    question_id: Optional[str]
    question_text: str
    choices: str
    correct_answer_letter: Optional[str]
    full_prompt_text: str


# ----------------------------------------------------------------------------
# DB Access
# ----------------------------------------------------------------------------
def ensure_metrics_table(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS reasoning_trace_forced_solution_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_id INTEGER NOT NULL,
            model_path TEXT NOT NULL,
            model_name TEXT NOT NULL,
            system_prompt TEXT NOT NULL,
            num_chunks INTEGER NOT NULL,
            predictions_json TEXT NOT NULL,
            continuation_texts_json TEXT NOT NULL DEFAULT '[]',
            token_confidences_json TEXT NOT NULL DEFAULT '[]',
            first_prediction_index INTEGER,
            first_correct_index INTEGER,
            stabilized_index INTEGER,
            stabilized_value TEXT,
            num_changes INTEGER NOT NULL,
            correct_at_first_chunk INTEGER NOT NULL,
            overall_correct INTEGER NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(trace_id, model_path, model_name, system_prompt)
        )
        """
    )
    # Best-effort migration: add missing columns with sane defaults
    cur.execute("PRAGMA table_info(reasoning_trace_forced_solution_metrics)")
    cols = {row[1] for row in cur.fetchall()}
    if "continuation_texts_json" not in cols:
        cur.execute("ALTER TABLE reasoning_trace_forced_solution_metrics ADD COLUMN continuation_texts_json TEXT NOT NULL DEFAULT '[]'")
    if "token_confidences_json" not in cols:
        cur.execute("ALTER TABLE reasoning_trace_forced_solution_metrics ADD COLUMN token_confidences_json TEXT NOT NULL DEFAULT '[]'")
    conn.commit()


def fetch_rows(
    conn: sqlite3.Connection,
    where_model_path: Optional[str],
    limit: Optional[int],
    offset: int,
) -> List[RowItem]:
    cur = conn.cursor()
    params: List[object] = []
    where_clause = ""
    if where_model_path:
        where_clause = "WHERE model_path = ?"
        params.append(where_model_path)
    sql = f"""
        SELECT id, model_path, question_id, question_text, choices, correct_answer_letter,
               full_prompt_text
        FROM reasoning_traces_qpqa
        {where_clause}
        ORDER BY id ASC
        LIMIT {limit if limit is not None else -1} OFFSET {offset}
    """
    cur.execute(sql, params)
    rows: List[RowItem] = []
    for r in cur.fetchall():
        rows.append(
            RowItem(
                id=r[0],
                model_path=r[1],
                question_id=r[2],
                question_text=r[3] or "",
                choices=r[4] or "",
                correct_answer_letter=(r[5] or "").strip() or None,
                full_prompt_text=r[6] or "",
            )
        )
    return rows


def has_existing_metrics(
    conn: sqlite3.Connection, trace_id: int, model_path: str, model_name: str, system_prompt: str
) -> bool:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT 1 FROM reasoning_trace_forced_solution_metrics
        WHERE trace_id = ? AND model_path = ? AND model_name = ? AND system_prompt = ?
        LIMIT 1
        """,
        (trace_id, model_path, model_name, system_prompt),
    )
    return cur.fetchone() is not None


def upsert_metrics(
    conn: sqlite3.Connection,
    *,
    trace_id: int,
    model_path: str,
    model_name: str,
    system_prompt: str,
    num_chunks: int,
    predictions: List[Optional[str]],
    continuation_texts: List[str],
    token_confidences: List[List[float]],
    first_prediction_index: Optional[int],
    first_correct_index: Optional[int],
    stabilized_index: Optional[int],
    stabilized_value: Optional[str],
    num_changes: int,
    correct_at_first_chunk: bool,
    overall_correct: bool,
):
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO reasoning_trace_forced_solution_metrics (
            trace_id, model_path, model_name, system_prompt, num_chunks, predictions_json,
            continuation_texts_json, token_confidences_json,
            first_prediction_index, first_correct_index, stabilized_index, stabilized_value,
            num_changes, correct_at_first_chunk, overall_correct, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(trace_id, model_path, model_name, system_prompt)
        DO UPDATE SET
            num_chunks = excluded.num_chunks,
            predictions_json = excluded.predictions_json,
            continuation_texts_json = excluded.continuation_texts_json,
            token_confidences_json = excluded.token_confidences_json,
            first_prediction_index = excluded.first_prediction_index,
            first_correct_index = excluded.first_correct_index,
            stabilized_index = excluded.stabilized_index,
            stabilized_value = excluded.stabilized_value,
            num_changes = excluded.num_changes,
            correct_at_first_chunk = excluded.correct_at_first_chunk,
            overall_correct = excluded.overall_correct,
            created_at = excluded.created_at
        """,
        (
            trace_id,
            model_path,
            model_name,
            system_prompt,
            num_chunks,
            json.dumps(predictions),
            json.dumps(continuation_texts),
            json.dumps(token_confidences),
            first_prediction_index,
            first_correct_index,
            stabilized_index,
            stabilized_value,
            num_changes,
            1 if correct_at_first_chunk else 0,
            1 if overall_correct else 0,
            datetime.utcnow().isoformat(timespec="seconds"),
        ),
    )
    conn.commit()


# ----------------------------------------------------------------------------
# Text Utilities are imported from rc_utils
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Model Wrapper
# ----------------------------------------------------------------------------
class Qwen3Forcing:
    def __init__(self, model_name: str, device_map: str = "auto"):
        logger.info("Loading model: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        self.model_name = model_name
        logger.info("Model loaded")

    @torch.no_grad()
    def forced_solution(
        self,
        prompt_text: str,
        think_content: str,
        system_prompt: Optional[str],
        max_new_tokens: int = 20,
    ) -> Optional[tuple[str, List[float]]]:
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt_text})

            header = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            templated = header + f"<think>\n{think_content}{QWEN3_SPECIAL_STOPPING_PROMPT}"

            inputs = self.tokenizer(templated, return_tensors="pt")
            gen_out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                output_scores=True,
                return_dict_in_generate=True,
            )
            output_ids = gen_out.sequences
            scores = gen_out.scores or []
            prompt_len = inputs["input_ids"].shape[1]
            new_ids = output_ids[0, prompt_len:]
            continuation = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

            # Compute per-token confidences (softmax prob for chosen token at each step)
            token_probs: List[float] = []
            for t, logits in enumerate(scores):
                probs = torch.nn.functional.softmax(logits[0], dim=-1)
                tok_id = new_ids[t].item()
                prob = probs[tok_id].detach().float().item()
                token_probs.append(float(prob))

            return continuation, token_probs
        except Exception as e:
            logger.error("Forced solution generation failed: %s", e)
            return None


LETTER_RE = re.compile(r"\b([A-Da-d])\b")


def extract_letter(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    m = LETTER_RE.search(text)
    if not m:
        return None
    return m.group(1).upper()


def compute_predictions_per_prefix(
    forcing: Qwen3Forcing,
    prompt_text: str,
    chunks: List[str],
    system_prompt: Optional[str],
    max_new_tokens: int,
) -> tuple[List[Optional[str]], List[str], List[List[float]]]:
    predictions: List[Optional[str]] = []
    continuations: List[str] = []
    confidences: List[List[float]] = []
    for i in range(len(chunks)):
        think_prefix = "\n\n".join(chunks[: i + 1])
        forced = forcing.forced_solution(prompt_text, think_prefix, system_prompt, max_new_tokens)
        if forced is None:
            continuations.append("")
            confidences.append([])
            predictions.append(None)
            continue
        cont_text, token_probs = forced
        letter = extract_letter(cont_text)
        predictions.append(letter)
        continuations.append(cont_text)
        confidences.append(token_probs)
    return predictions, continuations, confidences


def compute_metrics(
    predictions: List[Optional[str]],
    correct_letter: Optional[str],
) -> dict:
    n = len(predictions)
    first_prediction_index: Optional[int] = None
    for i, p in enumerate(predictions):
        if p is not None:
            first_prediction_index = i
            break

    first_correct_index: Optional[int] = None
    if correct_letter:
        for i, p in enumerate(predictions):
            if p == correct_letter:
                first_correct_index = i
                break

    # Stabilization: earliest s.t. all following equal and not None
    stabilized_index: Optional[int] = None
    stabilized_value: Optional[str] = None
    for s in range(n):
        val = predictions[s]
        if val is None:
            continue
        ok = True
        for j in range(s, n):
            if predictions[j] != val:
                ok = False
                break
        if ok:
            stabilized_index = s
            stabilized_value = val
            break

    num_changes = 0
    last: Optional[str] = None
    for p in predictions:
        if p is None:
            continue
        if last is None:
            last = p
            continue
        if p != last:
            num_changes += 1
            last = p

    correct_at_first_chunk = bool(n >= 1 and correct_letter and predictions[0] == correct_letter)
    overall_correct = bool(n >= 1 and correct_letter and predictions[-1] == correct_letter)

    return {
        "first_prediction_index": first_prediction_index,
        "first_correct_index": first_correct_index,
        "stabilized_index": stabilized_index,
        "stabilized_value": stabilized_value,
        "num_changes": num_changes,
        "correct_at_first_chunk": correct_at_first_chunk,
        "overall_correct": overall_correct,
    }


# ----------------------------------------------------------------------------
# Fire CLI entrypoint
# ----------------------------------------------------------------------------
def run(
    sqlite_db: str,
    where_model_path: str = "Qwen/Qwen3-32B",
    model_name: str = "Qwen/Qwen3-32B",
    system_prompt: str = "Answer only with a letter of a correct choice.",
    limit: Optional[int] = None,
    offset: int = 0,
    only_missing: bool = False,
    max_new_tokens: int = 20,
):
    """Precompute Forced Solution stability metrics into SQLite.

    Args mirror the former argparse flags. Use underscores in CLI: e.g., --only_missing.
    """

    # Load model
    forcing = Qwen3Forcing(model_name)

    # Connect DB
    conn = sqlite3.connect(sqlite_db)
    ensure_metrics_table(conn)

    rows = fetch_rows(conn, where_model_path, limit, offset)
    if not rows:
        logger.info("No rows found")
        return 0

    for row in tqdm(rows, desc="Processing traces"):
        if only_missing and has_existing_metrics(conn, row.id, row.model_path, forcing.model_name, system_prompt):
            continue

        think = extract_think_content(row.full_prompt_text)
        if not think:
            # Store empty metrics for trace to avoid revisiting repeatedly
            upsert_metrics(
                conn,
                trace_id=row.id,
                model_path=row.model_path,
                model_name=forcing.model_name,
                system_prompt=system_prompt,
                num_chunks=0,
                predictions=[],
                continuation_texts=[],
                token_confidences=[],
                first_prediction_index=None,
                first_correct_index=None,
                stabilized_index=None,
                stabilized_value=None,
                num_changes=0,
                correct_at_first_chunk=False,
                overall_correct=False,
            )
            continue

        chunks = split_reasoning_chain(think)
        if not chunks:
            upsert_metrics(
                conn,
                trace_id=row.id,
                model_path=row.model_path,
                model_name=forcing.model_name,
                system_prompt=system_prompt,
                num_chunks=0,
                predictions=[],
                continuation_texts=[],
                token_confidences=[],
                first_prediction_index=None,
                first_correct_index=None,
                stabilized_index=None,
                stabilized_value=None,
                num_changes=0,
                correct_at_first_chunk=False,
                overall_correct=False,
            )
            continue

        prompt_text = compose_user_prompt(row.question_text, row.choices)
        predictions, continuation_texts, token_confidences = compute_predictions_per_prefix(
            forcing,
            prompt_text,
            chunks,
            system_prompt,
            max_new_tokens,
        )
        metrics = compute_metrics(predictions, (row.correct_answer_letter or "").strip().upper() or None)

        upsert_metrics(
            conn,
            trace_id=row.id,
            model_path=row.model_path,
            model_name=forcing.model_name,
            system_prompt=system_prompt,
            num_chunks=len(chunks),
            predictions=predictions,
            continuation_texts=continuation_texts,
            token_confidences=token_confidences,
            first_prediction_index=metrics["first_prediction_index"],
            first_correct_index=metrics["first_correct_index"],
            stabilized_index=metrics["stabilized_index"],
            stabilized_value=metrics["stabilized_value"],
            num_changes=metrics["num_changes"],
            correct_at_first_chunk=metrics["correct_at_first_chunk"],
            overall_correct=metrics["overall_correct"],
        )

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(fire.Fire(run))


