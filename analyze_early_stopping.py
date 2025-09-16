import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TraceMetrics:
    trace_id: int
    num_chunks: int
    stabilized_index: Optional[int]
    stabilized_value: Optional[str]
    letter_probs: List[Dict[str, float]]
    predictions: List[Optional[str]]


def parse_letter_probs(raw: str) -> List[Dict[str, float]]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            out: List[Dict[str, float]] = []
            for item in data:
                if isinstance(item, dict):
                    # Normalize keys to A..D only and cast to float
                    filtered = {k.upper(): float(v) for k, v in item.items() if k and k.upper() in {"A", "B", "C", "D"}}
                    # Ensure missing letters are explicit zeros to avoid KeyError
                    for letter in ["A", "B", "C", "D"]:
                        if letter not in filtered:
                            filtered[letter] = 0.0
                    out.append(filtered)
                else:
                    out.append({"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0})
            return out
    except Exception:
        return []
    return []


def parse_predictions(raw: str) -> List[Optional[str]]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            out: List[Optional[str]] = []
            for item in data:
                if item is None:
                    out.append(None)
                else:
                    s = str(item).strip().upper()
                    out.append(s if s in {"A", "B", "C", "D"} else None)
            return out
    except Exception:
        return []
    return []


def top1_top2_from_probs(probs: Dict[str, float]) -> Tuple[str, float, float]:
    pairs = sorted(((letter, float(p)) for letter, p in probs.items()), key=lambda x: x[1], reverse=True)
    if not pairs:
        return "A", 0.0, 0.0
    top1_letter, top1_prob = pairs[0]
    top2_prob = pairs[1][1] if len(pairs) >= 2 else 0.0
    return top1_letter, top1_prob, top2_prob


def ensure_output_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_traces(
    sqlite_db: str,
    *,
    model_name: Optional[str],
    where_model_path: Optional[str],
    system_prompt: Optional[str],
    head_limit: Optional[int] = None,
    require_stabilized: bool = True,
    require_letter_probs: bool = True,
) -> List[TraceMetrics]:
    conn = sqlite3.connect(sqlite_db)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    conds: List[str] = []
    params: List[object] = []
    if model_name:
        conds.append("model_name = ?")
        params.append(model_name)
    if where_model_path:
        conds.append("model_path = ?")
        params.append(where_model_path)
    if system_prompt:
        conds.append("system_prompt = ?")
        params.append(system_prompt)
    if require_stabilized:
        conds.append("stabilized_index IS NOT NULL")
        conds.append("stabilized_value IS NOT NULL")
    if require_letter_probs:
        conds.append("letter_probs_json IS NOT NULL")
        conds.append("letter_probs_json != '[]'")

    where_sql = (" WHERE " + " AND ".join(conds)) if conds else ""
    sql = f"""
        SELECT trace_id, num_chunks, stabilized_index, stabilized_value,
               predictions_json, letter_probs_json
        FROM reasoning_trace_forced_solution_metrics
        {where_sql}
        ORDER BY trace_id ASC
    """
    if head_limit is not None and head_limit > 0:
        sql += f" LIMIT {int(head_limit)}"

    rows = list(cur.execute(sql, params))
    traces: List[TraceMetrics] = []
    for r in rows:
        num_chunks = int(r["num_chunks"]) if r["num_chunks"] is not None else 0
        if num_chunks <= 0:
            continue
        letter_probs = parse_letter_probs(r["letter_probs_json"] or "")
        if not letter_probs:
            continue
        # Truncate or pad to num_chunks for safety
        letter_probs = letter_probs[:num_chunks]
        if len(letter_probs) < num_chunks:
            letter_probs = letter_probs + [{"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}] * (num_chunks - len(letter_probs))

        predictions = parse_predictions(r["predictions_json"] or "")
        predictions = predictions[:num_chunks]
        if len(predictions) < num_chunks:
            predictions = predictions + [None] * (num_chunks - len(predictions))

        traces.append(
            TraceMetrics(
                trace_id=int(r["trace_id"]),
                num_chunks=num_chunks,
                stabilized_index=int(r["stabilized_index"]) if r["stabilized_index"] is not None else None,
                stabilized_value=(r["stabilized_value"] or "").strip().upper() or None,
                letter_probs=letter_probs,
                predictions=predictions,
            )
        )

    conn.close()
    logger.info("Loaded %d traces", len(traces))
    return traces


def simulate_early_stop(
    traces: List[TraceMetrics],
    *,
    tau: float,
    gamma: float,
    delta_min: float,
    window: int,
) -> pd.DataFrame:
    records: List[Dict] = []

    for t in traces:
        letter_history: List[str] = []
        prev_top1_prob: float = 0.0
        stop_record: Optional[Dict] = None

        for k in range(t.num_chunks):
            probs_k = t.letter_probs[k]
            top1_letter, top1_prob, top2_prob = top1_top2_from_probs(probs_k)
            margin = top1_prob - top2_prob
            delta_top1 = top1_prob - prev_top1_prob

            letter_history.append(top1_letter)
            if len(letter_history) > window:
                letter_history.pop(0)

            window_ok = (window <= 1) or (len(letter_history) == window and all(l == letter_history[0] for l in letter_history))
            tau_ok = top1_prob >= tau
            gamma_ok = margin >= gamma
            delta_ok = (k == 0 and delta_min <= top1_prob) or (delta_top1 >= delta_min)

            if window_ok and tau_ok and gamma_ok and delta_ok:
                stop_record = {
                    "trace_id": t.trace_id,
                    "k_stop": k,
                    "num_chunks": t.num_chunks,
                    "pred_letter": top1_letter,
                    "pred_prob": top1_prob,
                    "margin": margin,
                    "delta_top1": delta_top1,
                    "stabilized_index": t.stabilized_index,
                    "stabilized_value": t.stabilized_value,
                }
                break

            prev_top1_prob = top1_prob

        if stop_record is None:
            # Not triggered
            records.append({
                "trace_id": t.trace_id,
                "k_stop": None,
                "num_chunks": t.num_chunks,
                "pred_letter": None,
                "pred_prob": None,
                "margin": None,
                "delta_top1": None,
                "stabilized_index": t.stabilized_index,
                "stabilized_value": t.stabilized_value,
                "success": None,
                "lead_time": None,
                "saving": 0,
                "rel_pos_stop": None,
            })
        else:
            success = bool(stop_record["pred_letter"] is not None and t.stabilized_value is not None and stop_record["pred_letter"] == t.stabilized_value)
            k_stop = int(stop_record["k_stop"])  # earliest satisfied index
            lead_time = None
            if t.stabilized_index is not None:
                lead_time = t.stabilized_index - k_stop
            saving = max(0, t.num_chunks - 1 - k_stop)
            rel_pos = float(k_stop) / float(t.num_chunks) if t.num_chunks > 0 else None

            stop_record.update({
                "success": success,
                "lead_time": lead_time,
                "saving": saving,
                "rel_pos_stop": rel_pos,
            })
            records.append(stop_record)

    df = pd.DataFrame.from_records(records)
    return df


def summarize_and_plot(
    df: pd.DataFrame,
    *,
    out_dir: str,
    tag: str,
    tau_values: List[float],
    gamma_values: List[float],
    print_text_summaries: bool = True,
):
    ensure_output_dir(out_dir)
    sns.set_theme(style="whitegrid")

    # Precision/Coverage vs tau for different gamma values (window and delta kept in tag)
    summary_rows: List[Dict] = []
    for gamma in gamma_values:
        for tau in tau_values:
            dfg = df[(df["gamma"] == gamma) & (df["tau"] == tau)]
            n_total = int(dfg["trace_id"].nunique())
            n_trig = int(dfg["k_stop"].notna().sum())
            n_success = int(dfg["success"].fillna(False).sum())
            precision = (n_success / n_trig) if n_trig > 0 else np.nan
            coverage = (n_trig / n_total) if n_total > 0 else 0.0
            error_overall = ((n_trig - n_success) / n_total) if n_total > 0 else 0.0
            mean_saving_trig = dfg.loc[dfg["k_stop"].notna(), "saving"].mean() if n_trig > 0 else 0.0
            mean_saving_all = dfg["saving"].mean() if n_total > 0 else 0.0
            summary_rows.append({
                "tau": tau,
                "gamma": gamma,
                "precision": precision,
                "coverage": coverage,
                "error_overall": error_overall,
                "mean_saving_trig": mean_saving_trig,
                "mean_saving_all": mean_saving_all,
            })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(out_dir, f"summary_{tag}.csv"), index=False)

    if print_text_summaries:
        print("\n--- Precision/Coverage/Saving Summary ---")
        print(f"Tag: {tag}")
        print(summary.to_string())

    # Plot Precision and Coverage vs tau per gamma
    fig, ax1 = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette("tab10", n_colors=max(1, len(gamma_values)))
    for i, gamma in enumerate(gamma_values):
        sub = summary[summary["gamma"] == gamma].sort_values("tau")
        ax1.plot(sub["tau"], sub["precision"], label=f"precision, γ={gamma}", color=colors[i], linestyle="-")
        ax1.plot(sub["tau"], sub["coverage"], label=f"coverage, γ={gamma}", color=colors[i], linestyle="--")
    ax1.set_xlabel("tau (top1 probability threshold)")
    ax1.set_ylabel("metric value")
    ax1.set_title("Precision and Coverage vs tau")
    ax1.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"precision_coverage_vs_tau_{tag}.png"))
    plt.close(fig)

    # Reliability diagram: bin predicted prob at stop and show empirical success
    stops = df[df["k_stop"].notna()].copy()
    if not stops.empty:
        bins = np.linspace(0.0, 1.0, 11)
        stops["prob_bin"] = pd.cut(stops["pred_prob"].astype(float), bins=bins, include_lowest=True)
        rel = stops.groupby("prob_bin")["success"].mean().reset_index()
        centers = [(i + j) / 2.0 for i, j in zip(bins[:-1], bins[1:])]
        rel["center"] = centers

        if print_text_summaries:
            print("\n--- Reliability Diagram Data ---")
            print(f"Tag: {tag}")
            print(rel.to_string())

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(rel["center"], rel["success"], marker="o")
        ax.plot([0, 1], [0, 1], linestyle=":", color="gray")
        ax.set_xlabel("Predicted probability at stop (top1)")
        ax.set_ylabel("Empirical success rate")
        ax.set_title("Reliability diagram")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"reliability_{tag}.png"))
        plt.close(fig)

    # Savings vs overall error frontier
    fig, ax = plt.subplots(figsize=(7, 5))
    for gamma in gamma_values:
        sub = summary[summary["gamma"] == gamma]
        ax.plot(sub["mean_saving_all"], sub["error_overall"], marker="o", label=f"γ={gamma}")
    ax.set_xlabel("Average saving per trace (chunks)")
    ax.set_ylabel("Overall error rate (wrong early stops / all traces)")
    ax.set_title("Saving vs overall error")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"saving_vs_error_{tag}.png"))
    plt.close(fig)

    # Delta distributions for success vs failure
    sf = df[df["k_stop"].notna()].copy()
    if not sf.empty:
        if print_text_summaries:
            print("\n--- Delta_top1 Distribution Summary ---")
            print(f"Tag: {tag}")
            delta_summary = sf.groupby("success")["delta_top1"].describe()
            print(delta_summary.to_string())

        fig, ax = plt.subplots(figsize=(7, 5))
        sns.kdeplot(sf[sf["success"] == True]["delta_top1"].astype(float), label="success", ax=ax)
        sns.kdeplot(sf[sf["success"] == False]["delta_top1"].astype(float), label="failure", ax=ax)
        ax.set_xlabel("delta_top1 at stop")
        ax.set_title("Delta distribution: success vs failure")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"delta_success_vs_failure_{tag}.png"))
        plt.close(fig)


def grid_evaluate(
    traces: List[TraceMetrics],
    *,
    tau_list: List[float],
    gamma_list: List[float],
    delta_min_list: List[float],
    window_list: List[int],
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for window in window_list:
        for delta_min in delta_min_list:
            for gamma in gamma_list:
                for tau in tau_list:
                    df = simulate_early_stop(traces, tau=tau, gamma=gamma, delta_min=delta_min, window=window)
                    df["tau"] = tau
                    df["gamma"] = gamma
                    df["delta_min"] = delta_min
                    df["window"] = window
                    frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def parse_float_list(s, default: List[float]) -> List[float]:
    """Parse CLI input into list[float]. Accepts str with commas, list/tuple, or single number.

    Fire может передать аргумент как tuple/list при множественных значениях.
    """
    if s is None:
        return default
    out: List[float] = []
    # If iterable of values
    if isinstance(s, (list, tuple)):
        for item in s:
            if item is None:
                continue
            if isinstance(item, (int, float)):
                out.append(float(item))
            else:
                try:
                    text = str(item)
                    parts = [p.strip() for p in text.split(",") if p.strip()]
                    for p in parts:
                        try:
                            out.append(float(p))
                        except Exception:
                            pass
                except Exception:
                    pass
        return out if out else default
    # If single numeric
    if isinstance(s, (int, float)):
        return [float(s)]
    # Fallback: string with commas
    try:
        text = str(s)
        if text.strip() == "":
            return default
        parts = [p.strip() for p in text.split(",") if p.strip()]
        for p in parts:
            try:
                out.append(float(p))
            except Exception:
                pass
        return out if out else default
    except Exception:
        return default


def parse_int_list(s, default: List[int]) -> List[int]:
    """Parse CLI input into list[int]. Accepts str with commas, list/tuple, or single number."""
    if s is None:
        return default
    out: List[int] = []
    if isinstance(s, (list, tuple)):
        for item in s:
            if item is None:
                continue
            if isinstance(item, (int, float)):
                try:
                    out.append(int(item))
                except Exception:
                    pass
            else:
                try:
                    text = str(item)
                    parts = [p.strip() for p in text.split(",") if p.strip()]
                    for p in parts:
                        try:
                            out.append(int(p))
                        except Exception:
                            pass
                except Exception:
                    pass
        return out if out else default
    if isinstance(s, (int, float)):
        try:
            return [int(s)]
        except Exception:
            return default
    try:
        text = str(s)
        if text.strip() == "":
            return default
        parts = [p.strip() for p in text.split(",") if p.strip()]
        for p in parts:
            try:
                out.append(int(p))
            except Exception:
                pass
        return out if out else default
    except Exception:
        return default


def run(
    sqlite_db: str = "reasoning_traces.sqlite",
    model_name: str = "Qwen/Qwen3-32B",
    where_model_path: str = "Qwen/Qwen3-32B",
    system_prompt: str = "Answer only with a letter of a correct choice.",
    head_limit: Optional[int] = None,
    require_stabilized: bool = True,
    require_letter_probs: bool = True,
    tau_list: Optional[str] = None,
    gamma_list: Optional[str] = None,
    delta_min_list: Optional[str] = None,
    window_list: Optional[str] = None,
    out_dir: str = "analysis_results/early_stopping",
    print_text_summaries: bool = True,
):
    """Analyze early stopping rules based on letter probability dynamics.

    - Loads per-prefix letter probabilities and stabilization labels from the metrics table.
    - Simulates early stopping under a grid of thresholds.
    - Writes a CSV with per-trace decisions and generates summary plots.
    """

    # Defaults for grids
    taus = parse_float_list(tau_list, [0.6, 0.7, 0.8, 0.85, 0.9, 0.95])
    gammas = parse_float_list(gamma_list, [0.0, 0.1, 0.2])
    deltas = parse_float_list(delta_min_list, [0.0])
    windows = parse_int_list(window_list, [1])

    logger.info("Grid sizes: |tau|=%d |gamma|=%d |delta|=%d |window|=%d", len(taus), len(gammas), len(deltas), len(windows))

    traces = load_traces(
        sqlite_db,
        model_name=model_name,
        where_model_path=where_model_path,
        system_prompt=system_prompt,
        head_limit=head_limit,
        require_stabilized=require_stabilized,
        require_letter_probs=require_letter_probs,
    )
    if not traces:
        logger.info("No traces loaded; exiting")
        return

    df_all = grid_evaluate(traces, tau_list=taus, gamma_list=gammas, delta_min_list=deltas, window_list=windows)
    ensure_output_dir(out_dir)
    df_all.to_csv(os.path.join(out_dir, "early_stop_decisions.csv"), index=False)

    # Plot for the first delta and window as a representative slice
    sel_delta = deltas[0]
    sel_window = windows[0]
    tag = f"delta{sel_delta}_w{sel_window}"
    df_slice = df_all[(df_all["delta_min"] == sel_delta) & (df_all["window"] == sel_window)].copy()
    if df_slice.empty:
        logger.info("No slice to plot; check grid settings")
        return

    summarize_and_plot(
        df_slice,
        out_dir=out_dir,
        tag=tag,
        tau_values=taus,
        gamma_values=gammas,
        print_text_summaries=print_text_summaries,
    )
    logger.info("Done. Outputs in %s", out_dir)


if __name__ == "__main__":
    fire.Fire(run)


