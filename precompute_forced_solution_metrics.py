import json
import logging
import re
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional, Dict, Tuple

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
            letter_probs_json TEXT NOT NULL DEFAULT '[]',
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
    if "letter_probs_json" not in cols:
        cur.execute("ALTER TABLE reasoning_trace_forced_solution_metrics ADD COLUMN letter_probs_json TEXT NOT NULL DEFAULT '[]'")
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
    letter_probs: List[Dict[str, float]],
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
            letter_probs_json,
            first_prediction_index, first_correct_index, stabilized_index, stabilized_value,
            num_changes, correct_at_first_chunk, overall_correct, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(trace_id, model_path, model_name, system_prompt)
        DO UPDATE SET
            num_chunks = excluded.num_chunks,
            predictions_json = excluded.predictions_json,
            continuation_texts_json = excluded.continuation_texts_json,
            token_confidences_json = excluded.token_confidences_json,
            letter_probs_json = excluded.letter_probs_json,
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
            json.dumps(letter_probs),
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


def update_letter_probs_only(
    conn: sqlite3.Connection,
    *,
    trace_id: int,
    model_path: str,
    model_name: str,
    system_prompt: str,
    letter_probs: List[Dict[str, float]],
):
    """Efficiently update only the letter_probs_json column for a given trace."""
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE reasoning_trace_forced_solution_metrics
        SET letter_probs_json = ?
        WHERE trace_id = ? AND model_path = ? AND model_name = ? AND system_prompt = ?
        """,
        (
            json.dumps(letter_probs),
            trace_id,
            model_path,
            model_name,
            system_prompt,
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

        # Precompute candidate token ids for letters A..D
        self.letter_to_token_ids: Dict[str, List[int]] = {}
        for letter in ["A", "B", "C", "D"]:
            variants = [letter, f" {letter}"]
            ids: List[int] = []
            for v in variants:
                enc = self.tokenizer.encode(v, add_special_tokens=False)
                if len(enc) == 1:
                    ids.append(enc[0])
            # Deduplicate
            ids = list(dict.fromkeys(ids))
            self.letter_to_token_ids[letter] = ids

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

    @torch.no_grad()
    def letter_probs_first_nonspace(
        self,
        system_prompt: Optional[str],
        user_prompt: str,
        think_prefix: str,
        whitespace_topk: int = 1,
        topk_probe: int = 50,
    ) -> Dict[str, float]:
        """Probability for A..D at the first non-whitespace position after </think>.

        Approximates P(letter) = P(letter at t0) + Σ_{s in top-K whitespace} P(s at t0) * P(letter at t1 | s).
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        header = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        templated = header + f"<think>\n{think_prefix}{QWEN3_SPECIAL_STOPPING_PROMPT}"

        inputs = self.tokenizer(templated, return_tensors="pt")
        outputs = self.model(**inputs)
        logits = outputs.logits  # [1, seq_len, vocab]
        next_logits = logits[0, -1]
        probs = torch.softmax(next_logits.float(), dim=-1)

        # Direct contribution at t0
        direct: Dict[str, float] = {}
        for letter, ids in self.letter_to_token_ids.items():
            if not ids:
                direct[letter] = 0.0
                continue
            direct[letter] = max(float(probs[i].item()) for i in ids)

        # Identify whitespace candidates among top-K next tokens
        top_vals, top_ids = torch.topk(probs, k=min(topk_probe, probs.shape[-1]))
        whitespace_ids: List[Tuple[int, float]] = []
        for val, tid in zip(top_vals.tolist(), top_ids.tolist()):
            text = self.tokenizer.decode([tid])
            if text != "" and text.isspace():
                whitespace_ids.append((tid, float(val)))
            if len(whitespace_ids) >= whitespace_topk:
                break

        # Marginalize over whitespace candidates
        contrib = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}
        if whitespace_ids:
            base_ids = inputs["input_ids"]
            for tid, p_s in whitespace_ids:
                concat = torch.cat([base_ids, torch.tensor([[tid]], dtype=base_ids.dtype)], dim=1)
                attn = torch.ones_like(concat)
                out2 = self.model(input_ids=concat, attention_mask=attn)
                logits2 = out2.logits[0, -1]
                probs2 = torch.softmax(logits2.float(), dim=-1)
                for letter, ids in self.letter_to_token_ids.items():
                    if not ids:
                        continue
                    p2 = max(float(probs2[i].item()) for i in ids)
                    contrib[letter] += p_s * p2

        out = {}
        for letter in ["A", "B", "C", "D"]:
            out[letter] = min(1.0, direct.get(letter, 0.0) + contrib.get(letter, 0.0))
        return out


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
) -> tuple[List[Optional[str]], List[str], List[List[float]], List[Dict[str, float]]]:
    predictions: List[Optional[str]] = []
    continuations: List[str] = []
    confidences: List[List[float]] = []
    letter_probs: List[Dict[str, float]] = []
    for i in range(len(chunks)):
        think_prefix = "\n\n".join(chunks[: i + 1])
        forced = forcing.forced_solution(prompt_text, think_prefix, system_prompt, max_new_tokens)
        if forced is None:
            continuations.append("")
            confidences.append([])
            predictions.append(None)
            letter_probs.append({})
            continue
        cont_text, token_probs = forced
        letter = extract_letter(cont_text)
        predictions.append(letter)
        continuations.append(cont_text)
        confidences.append(token_probs)

        # Also compute letter probabilities
        probs = forcing.letter_probs_first_nonspace(system_prompt, prompt_text, think_prefix)
        letter_probs.append(probs)

    return predictions, continuations, confidences, letter_probs


def compute_letter_probs_only(
    forcing: Qwen3Forcing,
    prompt_text: str,
    chunks: List[str],
    system_prompt: Optional[str],
) -> List[Dict[str, float]]:
    """Efficiently compute only letter probabilities for all prefixes."""
    letter_probs: List[Dict[str, float]] = []
    for i in range(len(chunks)):
        think_prefix = "\n\n".join(chunks[: i + 1])
        probs = forcing.letter_probs_first_nonspace(system_prompt, prompt_text, think_prefix)
        letter_probs.append(probs)
    return letter_probs


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
    only_missing_letter_probs: bool = False,
    only_with_critical_chunk: bool = False,
    max_new_tokens: int = 20,
    fast_backfill: bool = False,
):
    """Precompute Forced Solution stability metrics into SQLite.

    Args mirror the former argparse flags. Use underscores in CLI: e.g., --only_missing.
    """

    # Load model
    forcing = Qwen3Forcing(model_name)

    # Connect DB
    conn = sqlite3.connect(sqlite_db)
    ensure_metrics_table(conn)

    if fast_backfill:
        logger.info("--- Running in FAST BACKFILL mode: updating only letter_probs_json ---")
        all_rows = fetch_rows(conn, where_model_path, limit, offset)

        cur = conn.cursor()
        cur.execute("SELECT trace_id FROM reasoning_trace_forced_solution_metrics WHERE letter_probs_json IS NOT NULL AND letter_probs_json != '[]'")
        processed_ids = {row[0] for row in cur.fetchall()}
        
        initial_count = len(all_rows)
        rows_to_process = [row for row in all_rows if row.id not in processed_ids]
        logger.info(f"Filtered {initial_count} rows down to {len(rows_to_process)} for fast backfill.")

        for row in tqdm(rows_to_process, desc="Fast backfilling letter_probs"):
            think = extract_think_content(row.full_prompt_text)
            if not think:
                continue
            chunks = split_reasoning_chain(think)
            if not chunks:
                continue

            prompt_text = compose_user_prompt(row.question_text, row.choices)
            
            letter_probs = compute_letter_probs_only(forcing, prompt_text, chunks, system_prompt)

            update_letter_probs_only(
                conn,
                trace_id=row.id,
                model_path=row.model_path,
                model_name=forcing.model_name,
                system_prompt=system_prompt,
                letter_probs=letter_probs,
            )
        
        logger.info("Fast backfill complete.")
        return 0

    # --- Selective Processing Logic ---
    rows = fetch_rows(conn, where_model_path, limit, offset)
    if not rows:
        logger.info("No rows found")
        return 0

    # Filter rows *before* processing, so tqdm shows the correct total.
    if only_missing_letter_probs or only_with_critical_chunk:
        initial_count = len(rows)
        
        # 1. Get IDs that already have letter_probs filled
        processed_ids = set()
        if only_missing_letter_probs:
            cur = conn.cursor()
            cur.execute("SELECT trace_id FROM reasoning_trace_forced_solution_metrics WHERE letter_probs_json IS NOT NULL AND letter_probs_json != '[]'")
            processed_ids = {row[0] for row in cur.fetchall()}
            logger.info(f"Found {len(processed_ids)} rows with existing letter_probs, will filter them out.")

        # 2. Get IDs that have a "critical chunk" to focus on them if requested
        critical_ids = set()
        if only_with_critical_chunk:
            cur = conn.cursor()
            # A "critical chunk" exists if the first correct answer was not on the first chunk
            cur.execute("SELECT trace_id FROM reasoning_trace_forced_solution_metrics WHERE first_correct_index IS NOT NULL AND first_correct_index > 0")
            critical_ids = {row[0] for row in cur.fetchall()}
            logger.info(f"Found {len(critical_ids)} rows with a critical chunk. Filtering to process only these.")

        # Apply filters
        original_rows = rows
        rows = []
        for row in original_rows:
            if only_missing_letter_probs and row.id in processed_ids:
                continue
            if only_with_critical_chunk and row.id not in critical_ids:
                continue
            rows.append(row)
        
        logger.info(f"Filtered {initial_count} rows down to {len(rows)} for processing.")


    for row in tqdm(rows, desc="Processing traces"):
        # Filters have already been applied above
        think = extract_think_content(row.full_prompt_text)
        if not think:
            # Note: We don't upsert empty metrics here anymore because this mode is for backfilling.
            # If a row has no think content, it will just be skipped.
            continue

        chunks = split_reasoning_chain(think)
        if not chunks:
            continue

        prompt_text = compose_user_prompt(row.question_text, row.choices)
        predictions, continuation_texts, token_confidences, letter_probs = compute_predictions_per_prefix(
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
            letter_probs=letter_probs,
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


