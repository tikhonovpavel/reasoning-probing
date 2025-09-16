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
from tqdm.auto import tqdm

from rc_utils import (
    QWEN3_SPECIAL_STOPPING_PROMPT,
    split_reasoning_chain,
    extract_think_content,
    compose_user_prompt,
)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Qwen3Scorer:
    def __init__(self, model_name: str, device_map: str = "auto"):
        logger.info("Loading model: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        self.model.eval()
        self.device = self.model.device
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

        inputs = self.tokenizer(templated, return_tensors="pt").to(self.device)
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

    @torch.no_grad()
    def letter_probs_first_nonspace(
        self,
        system_prompt: Optional[str],
        user_prompt: str,
        think_prefix: str,
        whitespace_topk: int = 3,
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

        inputs = self.tokenizer(templated, return_tensors="pt").to(self.device)
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
                new_id_tensor = torch.tensor([[tid]], dtype=base_ids.dtype, device=self.device)
                concat = torch.cat([base_ids, new_id_tensor], dim=1)
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

    @torch.no_grad()
    def short_rollout(
        self,
        system_prompt: Optional[str],
        user_prompt: str,
        think_prefix: str,
        max_new_tokens: int = 5,
    ) -> List[Tuple[int, str, float]]:
        """Deterministic short rollout returning list of (token_id, text, prob)."""
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
        inputs = self.tokenizer(templated, return_tensors="pt").to(self.device)
        gen = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )
        seq = gen.sequences
        scores = gen.scores or []
        prompt_len = inputs["input_ids"].shape[1]
        new_ids = seq[0, prompt_len:]
        triples: List[Tuple[int, str, float]] = []
        for t, tok_id in enumerate(new_ids.tolist()):
            logits_t = scores[t][0]
            probs_t = torch.softmax(logits_t.float(), dim=-1)
            prob = float(probs_t[tok_id].item())
            text = self.tokenizer.decode([tok_id])
            triples.append((tok_id, text, prob))
        return triples


def run(
    sqlite_db: str,
    model_name: str = "Qwen/Qwen3-32B",
    system_prompt: str = "Answer only with a letter of a correct choice.",
    where_model_path: str = "Qwen/Qwen3-32B",
    min_rel_pos: float = 0.0,
    max_rel_pos: float = 1.0,
    require_stabilized: bool = True,
    head_limit: Optional[int] = None,
    dry_run: bool = False,
    trace_ids: Optional[str] = None,
    whitespace_topk: int = 3,
    debug_rollout_tokens: int = 5,
    device_map: str = "auto",
):
    """Analyze local confidence jump at the first-correct chunk (no DB writes).

    - Select traces from reasoning_trace_forced_solution_metrics where first_correct_index > 0,
      overall_correct = 1, and (optionally) stabilized.
    - Recompute P(GT | c1..ck) and P(GT | c1..c{k-1}) under the exact same prompt construction
      as precompute_forced_solution_metrics.py, then report Δ = Pk - Pk-1.
    - Plot histogram of Δ and scatter of Δ vs (k/num_chunks).
    """
    scorer = Qwen3Scorer(model_name, device_map=device_map)

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
        return

    records: List[Dict] = []

    wanted_ids: Optional[set] = None
    if trace_ids:
        try:
            wanted_ids = {int(x.strip()) for x in trace_ids.split(",") if x.strip()}
        except Exception:
            wanted_ids = None

    for r in tqdm(rows, desc="Scoring boundaries"):
        trace_id = r["trace_id"]
        if wanted_ids is not None and trace_id not in wanted_ids:
            continue
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

        # Compute letter probabilities at first non-space
        probs_k = scorer.letter_probs_first_nonspace(system_prompt, user_prompt, prefix_k, whitespace_topk=whitespace_topk)
        probs_km1 = scorer.letter_probs_first_nonspace(system_prompt, user_prompt, prefix_km1, whitespace_topk=whitespace_topk)

        pk = float(probs_k.get(gt_letter, 0.0))
        pkm1 = float(probs_km1.get(gt_letter, 0.0))
        delta = pk - pkm1

        rec = {
            "trace_id": trace_id,
            "k": k,
            "num_chunks": num_chunks,
            "rel_pos": rel,
            "gt": gt_letter,
            "p_k": pk,
            "p_km1": pkm1,
            "delta": delta,
        }

        if dry_run and (wanted_ids is None or trace_id in wanted_ids):
            print(f"\n=== DRY TRACE {trace_id} (k={k}/{num_chunks}, rel={rel:.3f}) ===")
            # Next-token top-10 for both contexts
            for label, prefix in [("km1", prefix_km1), ("k", prefix_k)]:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": user_prompt})
                header = scorer.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                templ = header + f"<think>\n{prefix}{QWEN3_SPECIAL_STOPPING_PROMPT}"
                inp = scorer.tokenizer(templ, return_tensors="pt").to(scorer.device)
                out = scorer.model(**inp)
                nxt = out.logits[0, -1]
                pr = torch.softmax(nxt.float(), dim=-1)
                vals, ids = torch.topk(pr, k=10)
                print(f"-- next-token top10 [{label}] --")
                for v, tid in zip(vals.tolist(), ids.tolist()):
                    tx = scorer.tokenizer.decode([tid])
                    print(f"  id={tid:>6} prob={v:.4f} text={repr(tx)} isspace={tx.isspace() if tx != '' else False}")
            # Letter probs and short rollout
            print(f"P_km1: {probs_km1}")
            print(f"P_k  : {probs_k}")
            print(f"delta: {delta:.4f}")
            ro_km1 = scorer.short_rollout(system_prompt, user_prompt, prefix_km1, max_new_tokens=debug_rollout_tokens)
            ro_k = scorer.short_rollout(system_prompt, user_prompt, prefix_k, max_new_tokens=debug_rollout_tokens)
            print("-- rollout km1 --")
            for (tid, tx, p) in ro_km1:
                print(f"  id={tid:>6} prob={p:.4f} text={repr(tx)} isspace={tx.isspace() if tx != '' else False}")
            print("-- rollout k --")
            for (tid, tx, p) in ro_k:
                print(f"  id={tid:>6} prob={p:.4f} text={repr(tx)} isspace={tx.isspace() if tx != '' else False}")

        records.append(rec)

    if not records:
        logger.info("No eligible examples after filtering.")
        return

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

    # Plots (skip plots in dry mode with explicit trace_ids)
    if dry_run and wanted_ids is not None:
        return

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
    plt.savefig("analysis_results/critical_jump.png")
    plt.show()


if __name__ == "__main__":
    fire.Fire(run)
