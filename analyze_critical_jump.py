import json
import logging
import sqlite3
from typing import Dict, List, Optional, Tuple

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rc_utils import (
    QWEN3_SPECIAL_STOPPING_PROMPT,
    split_reasoning_chain,
    extract_think_content,
    compose_user_prompt,
)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Qwen3Scorer:
    def __init__(self, model_name: str):
        logger.info("Loading model: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        self.model.eval()
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
    def letter_probs(
        self,
        system_prompt: Optional[str],
        user_prompt: str,
        think_prefix: str,
    ) -> Dict[str, float]:
        """Return probability for A..D as next token given the constructed prompt.

        We build the exact same header as in precompute_forced_solution_metrics.py and
        take softmax over logits at the next position.
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

        out: Dict[str, float] = {}
        for letter, ids in self.letter_to_token_ids.items():
            if not ids:
                out[letter] = 0.0
                continue
            # Use max prob over single-token variants
            p = max(float(probs[i].item()) for i in ids)
            out[letter] = p
        return out


def run(
    sqlite_db: str,
    model_name: str = "Qwen/Qwen3-32B",
    system_prompt: str = "Answer only with a letter of a correct choice.",
    where_model_path: str = "Qwen/Qwen3-32B",
    min_rel_pos: float = 0.0,
    max_rel_pos: float = 1.0,
    require_stabilized: bool = True,
    head_limit: Optional[int] = None,
):
    """Analyze local confidence jump at the first-correct chunk (no DB writes).

    - Select traces from reasoning_trace_forced_solution_metrics where first_correct_index > 0,
      overall_correct = 1, and (optionally) stabilized.
    - Recompute P(GT | c1..ck) and P(GT | c1..c{k-1}) under the exact same prompt construction
      as precompute_forced_solution_metrics.py, then report Δ = Pk - Pk-1.
    - Plot histogram of Δ and scatter of Δ vs (k/num_chunks).
    """
    scorer = Qwen3Scorer(model_name)

    conn = sqlite3.connect(sqlite_db)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    conds = [
        "m.first_correct_index IS NOT NULL",
        "m.first_correct_index > 0",
        "m.overall_correct = 1",
        "m.model_name = ?",
        "m.model_path = ?",
        "m.system_prompt = ?",
    ]
    params = [model_name, where_model_path, system_prompt]
    if require_stabilized:
        conds.append("m.stabilized_index IS NOT NULL")

    where_sql = " AND ".join(conds)
    sql = f"""
        SELECT m.trace_id, m.num_chunks, m.first_correct_index, m.stabilized_index,
               q.question_text, q.choices, q.full_prompt_text
        FROM reasoning_trace_forced_solution_metrics m
        JOIN reasoning_traces_qpqa q ON q.id = m.trace_id
        WHERE {where_sql}
        ORDER BY m.trace_id ASC
    """
    if head_limit is not None and head_limit > 0:
        sql += f" LIMIT {int(head_limit)}"

    rows = list(cur.execute(sql, params))
    if not rows:
        logger.info("No rows matched the criteria.")
        return 0

    records: List[Dict] = []

    for r in rows:
        trace_id = r["trace_id"]
        num_chunks = int(r["num_chunks"]) if r["num_chunks"] is not None else 0
        k = int(r["first_correct_index"]) if r["first_correct_index"] is not None else -1
        if num_chunks <= 0 or k <= 0 or k >= num_chunks:
            continue

        # Relative position filter
        rel = float(k) / float(num_chunks)
        if not (min_rel_pos <= rel <= max_rel_pos):
            continue

        full_prompt_text = r["full_prompt_text"] or ""
        think = extract_think_content(full_prompt_text)
        if not think:
            continue
        chunks = split_reasoning_chain(think)
        if len(chunks) != num_chunks:
            # Fall back to computed chunks length if mismatch
            num_chunks = len(chunks)
            if k >= num_chunks:
                continue

        user_prompt = compose_user_prompt(r["question_text"] or "", r["choices"] or "")

        # Construct prefixes
        prefix_k = "\n\n".join(chunks[: k + 1])
        prefix_km1 = "\n\n".join(chunks[: k])

        # Determine GT letter from stabilized_value when available, else approximate via DB join not provided here
        # Safer approach: compute both distributions and pick GT from the FOUR options that matches the stabilized_value if present
        # But stabilized_value is the letter at stabilization, which equals final answer; we use it if present, else skip.
        # Fetch stabilized_value via m.stabilized_value in the initial query? Not selected above; add quick fetch now:
        # Simpler: query once more for this trace to get stabilized_value.
        cur2 = conn.cursor()
        cur2.execute(
            "SELECT stabilized_value FROM reasoning_trace_forced_solution_metrics WHERE trace_id = ? AND model_name = ? AND model_path = ? AND system_prompt = ?",
            (trace_id, model_name, where_model_path, system_prompt),
        )
        row_sv = cur2.fetchone()
        gt_letter = (row_sv[0] or "").strip().upper() if row_sv and row_sv[0] else None
        if gt_letter not in {"A", "B", "C", "D"}:
            # As a fallback, skip this trace
            continue

        probs_k = scorer.letter_probs(system_prompt, user_prompt, prefix_k)
        probs_km1 = scorer.letter_probs(system_prompt, user_prompt, prefix_km1)

        pk = float(probs_k.get(gt_letter, 0.0))
        pkm1 = float(probs_km1.get(gt_letter, 0.0))
        delta = pk - pkm1

        records.append({
            "trace_id": trace_id,
            "k": k,
            "num_chunks": num_chunks,
            "rel_pos": rel,
            "gt": gt_letter,
            "p_k": pk,
            "p_km1": pkm1,
            "delta": delta,
        })

    if not records:
        logger.info("No eligible examples after filtering.")
        return 0

    df = pd.DataFrame.from_records(records)

    # Summary
    print("=== Critical jump summary ===")
    print(f"n: {len(df)}")
    print(f"delta_mean: {df['delta'].mean():.4f}")
    print(f"delta_median: {df['delta'].median():.4f}")
    print(f"delta_p10: {df['delta'].quantile(0.10):.4f}")
    print(f"delta_p90: {df['delta'].quantile(0.90):.4f}")
    big_drop = (df["delta"] <= -0.2).mean()
    big_gain = (df["delta"] >= 0.2).mean()
    print(f"share_large_negative_delta(<=-0.2): {big_drop:.3f}")
    print(f"share_large_positive_delta(>=0.2): {big_gain:.3f}")

    # Plots
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram of delta
    ax = axes[0]
    sns.histplot(df["delta"], bins=30, kde=True, ax=ax, color="#1f77b4")
    ax.set_title("Δ = P(GT | c1..ck) − P(GT | c1..c{k−1})")
    ax.set_xlabel("Delta confidence for GT letter")

    # Scatter: delta vs rel_pos
    ax = axes[1]
    sns.regplot(data=df, x="rel_pos", y="delta", ax=ax,
                scatter_kws={'alpha': 0.5, 'color': '#2ca02c'},
                line_kws={'color': '#d62728'})
    ax.set_title("Delta vs first-correct relative position")
    ax.set_xlabel("k / num_chunks")
    ax.set_ylabel("Delta confidence (GT)")

    plt.tight_layout()
    plt.show()

    return 0


if __name__ == "__main__":
    fire.Fire(run)


