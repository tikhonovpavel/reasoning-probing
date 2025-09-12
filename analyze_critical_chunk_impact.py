# analyze_critical_chunk_impact.py

import sqlite3
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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
import logging

# --- Configure logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AnswerProbabilityMeasurer:
    """
    A class to measure the probability distribution over answer choices (A, B, C, D)
    given a specific context.
    """
    def __init__(self, model_name: str):
        logger.info(f"Loading model: {model_name}")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        self._get_answer_token_ids()
        logger.info("Model loaded and answer token IDs identified.")

    def _get_answer_token_ids(self):
        """Find the token IDs for the answer choices ' A', ' B', ' C', ' D'."""
        self.answer_token_ids = {}
        for letter in ["A", "B", "C", "D"]:
            # Models are often sensitive to the leading space after the prompt
            token_id = self.tokenizer.encode(f" {letter}", add_special_tokens=False)
            if len(token_id) == 1:
                self.answer_token_ids[letter] = token_id[0]
            else:
                logger.warning(f"Could not find a single token for ' {letter}'. This may affect probability calculation.")
        
        if len(self.answer_token_ids) != 4:
            raise RuntimeError(f"Failed to identify all answer token IDs. Found: {self.answer_token_ids}")
        logger.info(f"Answer token IDs: {self.answer_token_ids}")

    @torch.no_grad()
    def get_answer_distribution(
        self, prompt_text: str, think_content: str, system_prompt: str
    ) -> dict[str, float] | None:
        """
        Computes the probability distribution over {A, B, C, D} for the first generated token.
        """
        try:
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt_text}]
            header = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            templated = header + f"<think>\n{think_content}{QWEN3_SPECIAL_STOPPING_PROMPT}"
            
            inputs = self.tokenizer(templated, return_tensors="pt").to(self.model.device)
            outputs = self.model(**inputs)
            
            # Get logits for the very next token
            next_token_logits = outputs.logits[0, -1, :]
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            
            dist = {}
            for letter, token_id in self.answer_token_ids.items():
                dist[letter] = probs[token_id].item()
            
            return dist
        except Exception as e:
            logger.error(f"Failed to get answer distribution: {e}")
            return None


def main(
    db_path: str = "./reasoning_traces.sqlite",
    model_name: str = "Qwen/Qwen3-32B",
    system_prompt: str = "Answer only with a letter of a correct choice.",
):
    """
    Analyzes the impact of the 'critical chunk' on the probability of the correct answer.
    """
    # --- Load Data ---
    conn = sqlite3.connect(db_path)
    
    # Load metrics
    metrics_query = "SELECT * FROM reasoning_trace_forced_solution_metrics WHERE model_name = ?"
    df_metrics = pd.read_sql_query(metrics_query, conn, params=(model_name,))
    
    # Load original trace data to get ground truth and chunks
    traces_query = "SELECT id, question_text, choices, correct_answer_letter, full_prompt_text FROM reasoning_traces_qpqa"
    df_traces = pd.read_sql_query(traces_query, conn)
    df_traces.rename(columns={"id": "trace_id"}, inplace=True)
    
    df = pd.merge(df_metrics, df_traces, on="trace_id")

    # --- Filter for "Critical" Examples ---
    df["first_correct_index_num"] = pd.to_numeric(df["first_correct_index"], errors="coerce")
    
    # Conditions for a critical transition:
    # 1. An initial mistake was made (correct answer is not at chunk 0).
    # 2. The model eventually found the correct answer.
    # 3. The answer stabilized on the correct one.
    # 4. There's at least one chunk before the critical one to compare against.
    mask = (
        df["first_correct_index_num"].notna() &
        (df["first_correct_index_num"] > 0) &
        (df["overall_correct"] == 1) &
        (df["stabilized_value"] == df["correct_answer_letter"])
    )
    df_critical = df[mask].copy()
    
    if df_critical.empty:
        print("No 'critical transition' examples found with the given filters. Exiting.")
        return

    print(f"Found {len(df_critical)} examples of 'critical transitions' to analyze.")

    # --- Analysis ---
    measurer = AnswerProbabilityMeasurer(model_name)
    results = []

    for _, row in tqdm(df_critical.iterrows(), total=len(df_critical), desc="Analyzing critical chunks"):
        k = int(row["first_correct_index_num"])
        prompt_text = compose_user_prompt(row["question_text"], row["choices"])
        think_content = extract_think_content(row["full_prompt_text"])
        chunks = split_reasoning_chain(think_content)
        gt_letter = row["correct_answer_letter"]

        if gt_letter not in ["A", "B", "C", "D"]:
            continue

        # Context BEFORE the critical chunk (c_1, ..., c_{k-1})
        # Note: first_correct_index is 0-based, so index k corresponds to chunk k+1.
        # The context before is chunks up to k-1.
        context_before = "\n\n".join(chunks[:k])
        dist_before = measurer.get_answer_distribution(prompt_text, context_before, system_prompt)
        
        # Context AFTER the critical chunk (c_1, ..., c_k)
        context_after = "\n\n".join(chunks[:k+1])
        dist_after = measurer.get_answer_distribution(prompt_text, context_after, system_prompt)

        if dist_before and dist_after:
            p_correct_before = dist_before.get(gt_letter, 0.0)
            p_correct_after = dist_after.get(gt_letter, 0.0)
            results.append({
                "trace_id": row["trace_id"],
                "k": k,
                "p_correct_before": p_correct_before,
                "p_correct_after": p_correct_after,
                "delta_p": p_correct_after - p_correct_before,
            })

    if not results:
        print("Analysis finished, but no valid results were produced.")
        return
        
    df_results = pd.DataFrame(results)

    # --- Visualization ---
    mean_delta = df_results["delta_p"].mean()
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df_results["delta_p"], kde=True, bins=30)
    plt.axvline(mean_delta, color='r', linestyle='--', label=f'Mean Delta = {mean_delta:.3f}')
    plt.title(f"Impact of the Critical Chunk on P(correct answer)\n(N={len(df_results)})")
    plt.xlabel("Change in Probability (P_after - P_before)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)
    sns.despine()
    
    print("\n=== Analysis Complete ===")
    print(df_results.describe())
    
    plt.show()


if __name__ == "__main__":
    fire.Fire(main)
