import sqlite3, json, numpy as np, pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr, pearsonr


# --- Config (relative path) ---
DB_PATH = "./reasoning_traces.sqlite"
FILTER_MODEL_PATH = "Qwen/Qwen3-32B"   # set to None to disable
FILTER_MODEL_NAME = "Qwen/Qwen3-32B"   # set to None to disable
FILTER_SYSTEM_PROMPT = "Answer only with a letter of a correct choice."  # set to None to disable

# --- Load ---
conn = sqlite3.connect(DB_PATH)
conds, params = [], []
if FILTER_MODEL_PATH:
    conds.append("model_path = ?"); params.append(FILTER_MODEL_PATH)
if FILTER_MODEL_NAME:
    conds.append("model_name = ?"); params.append(FILTER_MODEL_NAME)
if FILTER_SYSTEM_PROMPT:
    conds.append("system_prompt = ?"); params.append(FILTER_SYSTEM_PROMPT)
where_clause = ("WHERE " + " AND ".join(conds)) if conds else ""

query = f"""
SELECT
  trace_id, model_path, model_name, system_prompt, num_chunks,
  predictions_json, continuation_texts_json, token_confidences_json,
  first_prediction_index, first_correct_index, stabilized_index, stabilized_value,
  num_changes, correct_at_first_chunk, overall_correct, created_at
FROM reasoning_trace_forced_solution_metrics
{where_clause}
"""
df = pd.read_sql_query(query, conn, params=params)

def parse_json_list(s, default):
    try:
        v = json.loads(s) if isinstance(s, str) else s
        return v if isinstance(v, list) else default
    except Exception:
        return default

df["predictions"]   = df["predictions_json"].apply(lambda s: parse_json_list(s, []))
df["continuations"] = df["continuation_texts_json"].apply(lambda s: parse_json_list(s, []))
df["token_confs"]   = df["token_confidences_json"].apply(lambda s: parse_json_list(s, []))

# Derived
df["has_chunks"] = df["num_chunks"] > 0
df["stabilized_index_num"] = pd.to_numeric(df["stabilized_index"], errors="coerce")
mask_stab = df["stabilized_index_num"].notna() & df["has_chunks"]
df["stab_pos_rel"] = np.where(
    mask_stab,
    df["stabilized_index_num"].astype(float) / df["num_chunks"].astype(float),
    np.nan,
)

# Flatten token confidence lengths across prefixes
tok_lens = []
for row in df["token_confs"]:
    if isinstance(row, list):
        for step in row:
            if isinstance(step, list):
                tok_lens.append(len(step))
tok_lens = np.array(tok_lens, dtype=int) if len(tok_lens) else np.array([], dtype=int)

# --- Summary (text-only) ---
print("=== Summary ===")
print(f"rows_total: {len(df)}")
print(f"rows_with_chunks: {int(df['has_chunks'].sum())}")
print(f"rows_without_think: {int((~df['has_chunks']).sum())}")
print(f"correct_at_first_chunk: {int((df['correct_at_first_chunk'] == 1).sum())}")
print(f"overall_correct: {int((df['overall_correct'] == 1).sum())}")
print(f"stabilized_index==0: {int((df['stabilized_index_num'] == 0).sum())}")
print(f"no_stabilization: {int(df['stabilized_index_num'].isna().sum())}")
print(f"num_changes_mean: {float(df['num_changes'].mean()) if len(df) else float('nan'):.3f}")
print(f"num_chunks_mean: {float(df['num_chunks'].mean()) if len(df) else float('nan'):.3f}")
print(f"stab_pos_rel_mean: {float(df['stab_pos_rel'].mean(skipna=True)) if len(df) else float('nan'):.3f}")
print(f"avg_tokens_per_prefix_overall: {float(tok_lens.mean()) if tok_lens.size else 0.0:.3f}")

# --- Visualization (distributions) + text for each ---
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(3, 3, figsize=(18, 15))

# (1) num_changes distribution
ax = axes[0,0]
if len(df):
    sns.histplot(df, x="num_changes", discrete=True, stat="probability", shrink=0.8, ax=ax, color="#1f77b4")
ax.set_title("num_changes")
ax.set_xlabel("Number of answer changes across prefixes")
print("\n=== Text: num_changes distribution ===")
if len(df):
    probs = df["num_changes"].value_counts(normalize=True).sort_index()
    for k, p in probs.items():
        print(f"num_changes={k}: p={p:.3f}")
else:
    print("No data")

# (2) num_chunks distribution
ax = axes[0,1]
if len(df):
    sns.histplot(df, x="num_chunks", discrete=True, stat="probability", shrink=0.8, ax=ax, color="#ff7f0e")
ax.set_title("num_chunks")
ax.set_xlabel("Number of chunks in <think>")
print("\n=== Text: num_chunks distribution ===")
if len(df):
    probs = df["num_chunks"].value_counts(normalize=True).sort_index()
    for k, p in probs.items():
        print(f"num_chunks={k}: p={p:.3f}")
else:
    print("No data")

# (3) Stabilization position relative distribution (0..1)
ax = axes[0,2]
valid_rel = df["stab_pos_rel"].dropna()
if len(valid_rel):
    sns.histplot(valid_rel, bins=20, stat="probability", kde=True, ax=ax, color="#2ca02c")
ax.set_title("Stabilization position")
ax.set_xlabel("stabilized_index / num_chunks")
print("\n=== Text: stabilization position relative ===")
if len(valid_rel):
    counts, bin_edges = np.histogram(valid_rel, bins=20, range=(0, 1))
    total = counts.sum() if counts.sum() > 0 else 1
    print(f"mean: {valid_rel.mean():.4f}")
    print(f"median: {valid_rel.median():.4f}")
    print(f"p10: {valid_rel.quantile(0.10):.4f}")
    print(f"p90: {valid_rel.quantile(0.90):.4f}")
    for i in range(len(counts)):
        p = counts[i] / total
        print(f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f}): p={p:.3f}")
else:
    print("No data")

# (4) Correctness proportions (bar as proportion)
ax = axes[1,0]
total_rows = max(1, len(df))
corr_props = pd.DataFrame({
    "Metric": ["correct_at_first_chunk", "overall_correct"],
    "Proportion": [
        int((df["correct_at_first_chunk"] == 1).sum()) / total_rows,
        int((df["overall_correct"] == 1).sum()) / total_rows,
    ],
})
sns.barplot(data=corr_props, x="Metric", y="Proportion", ax=ax, palette=["#9467bd", "#8c564b"])
ax.set_title("Correctness\n(among all examples)")
ax.set_xlabel("")
ax.set_ylabel("Proportion")
print("\n=== Text: correctness proportions ===")
for _, row in corr_props.iterrows():
    print(f"{row['Metric']}: p={row['Proportion']:.3f}")

# (5) First time correct (first_correct_index) distribution
ax = axes[1,1]
first_correct = pd.to_numeric(df["first_correct_index"], errors="coerce")
valid_fc = first_correct.dropna().astype(int)
# Filter out values >= 30
valid_fc = valid_fc[valid_fc < 30]
missing_fc = int(first_correct.isna().sum())
if len(valid_fc):
    sns.histplot(valid_fc, discrete=True, stat="probability", shrink=0.8, ax=ax, color="#17becf")
ax.set_title("First correct index\n(among examples with at least one correct prediction)")
ax.set_xlabel("k where first_correct_index == k")
print("\n=== Text: first_correct_index distribution ===")
if len(valid_fc):
    probs = valid_fc.value_counts(normalize=True).sort_index()
    for k, p in probs.items():
        print(f"first_correct_index={k}: p={p:.3f}")
    print(f"missing_first_correct: count={missing_fc}, p={missing_fc / max(1,len(df)):.3f}")
else:
    print("No data")

# (6) Token confidences length per prefix (distribution)
ax = axes[1,2]
if tok_lens.size:
    bins = np.arange(-0.5, tok_lens.max() + 1.5, 1)
    sns.histplot(tok_lens, bins=bins, discrete=True, stat="probability", ax=ax, color="#d62728")
ax.set_title("Generated length per prefix")
ax.set_xlabel("Length (tokens)")
ax.set_ylabel("Probability")
print("\n=== Text: generated length per prefix ===")
if tok_lens.size:
    probs = pd.Series(tok_lens).value_counts(normalize=True).sort_index()
    print(f"mean: {tok_lens.mean():.3f}")
    print(f"median: {np.median(tok_lens):.3f}")
    print(f"min: {tok_lens.min()}  max: {tok_lens.max()}")
    for k, p in probs.items():
        print(f"length={k}: p={p:.3f}")
else:
    print("No data")

# (7) Reasoning length vs. knowing answer at first chunk (overlaid histograms)
ax = axes[2,0]
if len(df) and df["correct_at_first_chunk"].nunique() > 1:
    sns.histplot(data=df, x="num_chunks", hue="correct_at_first_chunk", 
                 ax=ax, kde=True, stat="probability", common_norm=False,
                 palette=["#e377c2", "#7f7f7f"], element="step")
    handles, labels = ax.get_legend_handles_labels()
    if handles: # Add legend only if plot was drawn
        ax.legend(handles, ["Doesn't know answer at chunk 1", "Knows answer at chunk 1"], title="Initial State")
ax.set_title("Reasoning length vs. early knowledge")
ax.set_xlabel("Number of chunks")
ax.set_ylabel("Probability Density")
print("\n=== Text: Reasoning length vs. early knowledge ===")
if len(df) and df["correct_at_first_chunk"].notna().any() and df["num_chunks"].notna().any():
    group_knows = df[df["correct_at_first_chunk"] == 1]["num_chunks"]
    group_doesnt_know = df[df["correct_at_first_chunk"] == 0]["num_chunks"]
    
    print("\n-- Group: Knows answer at chunk 1 --")
    if not group_knows.empty:
        print(f"  count: {len(group_knows)}")
        print(f"  mean_chunks: {group_knows.mean():.2f}")
        print(f"  median_chunks: {group_knows.median():.2f}")
    else:
        print("  No data")

    print("\n-- Group: Doesn't know answer at chunk 1 --")
    if not group_doesnt_know.empty:
        print(f"  count: {len(group_doesnt_know)}")
        print(f"  mean_chunks: {group_doesnt_know.mean():.2f}")
        print(f"  median_chunks: {group_doesnt_know.median():.2f}")
    else:
        print("  No data")

    # Point-Biserial Correlation: appropriate for binary vs continuous variable
    corr, p_value = pointbiserialr(df["correct_at_first_chunk"], df["num_chunks"])
    print("\n-- Correlation --")
    print(f"Point-Biserial Correlation: {corr:.3f}")
    print(f"p-value: {p_value:.3f}")
    if p_value < 0.05:
        print("Note: The correlation is statistically significant.")
    else:
        print("Note: The correlation is not statistically significant.")
else:
    print("No data")

# (8) Correlation: Reasoning Length vs. First Correct Position
ax = axes[2,1]
# Prepare data for this plot
df_corr = df.copy()
df_corr["first_correct_index_num"] = pd.to_numeric(df_corr["first_correct_index"], errors="coerce")
# --- MODIFICATION: Exclude examples where the correct answer was found in the first chunk ---
df_corr = df_corr[df_corr["first_correct_index_num"] != 0].copy()
df_corr.dropna(subset=["first_correct_index_num", "num_chunks"], inplace=True)
df_corr = df_corr[df_corr["num_chunks"] > 0]
df_corr["first_correct_rel_pos"] = df_corr["first_correct_index_num"] / df_corr["num_chunks"]

if not df_corr.empty:
    sns.regplot(data=df_corr, x="first_correct_rel_pos", y="num_chunks", ax=ax,
                scatter_kws={'alpha':0.4, 'color': '#1f77b4'},
                line_kws={'color': '#d62728'})
ax.set_title("Correlation: Reasoning Length vs. 'When Answer Was Found'\n(excluding examples solved at chunk 1)")
ax.set_xlabel("Relative position of first correct answer (0=start, 1=end)")
ax.set_ylabel("Total number of chunks")
print("\n=== Text: Correlation of Reasoning Length vs. First Correct Position (excluding chunk 1 solvers) ===")
if not df_corr.empty and len(df_corr) > 1:
    corr, p_value = pearsonr(df_corr["first_correct_rel_pos"], df_corr["num_chunks"])
    print(f"Pearson Correlation: {corr:.3f}")
    print(f"p-value: {p_value:.3f}")
    if p_value < 0.05:
        print("Note: The correlation is statistically significant.")
    else:
        print("Note: The correlation is not statistically significant.")
else:
    print("No data for correlation analysis.")

# Hide unused subplots
axes[2, 2].set_visible(False)

plt.tight_layout(pad=3.0)
sns.despine()
plt.show()