import logging
import sqlite3
from typing import Optional

import fire
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run(
    sqlite_db: str = 'reasoning_traces.sqlite',
    source_table_name: str = "reasoning_traces_gpqa",
    model_name: str = "Qwen/Qwen3-32B",
    where_model_path: str = "Qwen/Qwen3-32B",
    system_prompt: str = "Answer only with a letter of a correct choice.",
    head_limit: Optional[int] = None,
    out_dir: str = "analysis_results/accuracy_vs_length",
    bin_percent_threshold: float = 8.0,
):
    """
    Analyzes the relationship between reasoning chain length (num_chunks) and accuracy.
    Groups results into bins containing at least a certain percentage of the total data.
    """
    conn = sqlite3.connect(sqlite_db)
    
    conds = [
        "source_table = ?",
        "stabilized_index IS NOT NULL",
        "model_name = ?",
        "model_path = ?",
        "system_prompt = ?",
    ]
    params = [source_table_name, model_name, where_model_path, system_prompt]

    where_sql = " AND ".join(conds)
    sql = f"""
        SELECT num_chunks, overall_correct, num_changes
        FROM reasoning_trace_forced_solution_metrics
        WHERE {where_sql}
        ORDER BY num_chunks
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

    df['num_changes_category'] = df['num_changes'].apply(
        lambda n: '0' if n == 0 else '1' if n == 1 else '2' if n == 2 else '3+'
    )

    # --- Analysis 1: Raw per-chunk accuracy (for text output) ---
    accuracy_by_length = df.groupby('num_chunks')['overall_correct'].agg(['mean', 'count']).reset_index()
    accuracy_by_length = accuracy_by_length.rename(columns={'mean': 'accuracy'})
    
    print("\n--- Accuracy vs. Reasoning Chain Length (Raw) ---")
    print(accuracy_by_length.to_string(index=False))
    print("-" * 40)

    # --- Analysis 2: Binned accuracy analysis with breakdown by num_changes ---
    per_chunk_counts = df.groupby('num_chunks').size().reset_index(name='count')
    
    total_examples = len(df)
    min_bin_size = int(total_examples * (bin_percent_threshold / 100.0))

    binned_results = []
    current_bin_chunks = []
    current_bin_count = 0

    for _, row in per_chunk_counts.iterrows():
        current_bin_chunks.append(row['num_chunks'])
        current_bin_count += row['count']

        if current_bin_count >= min_bin_size:
            binned_results.append({
                'num_chunks_range': f"{min(current_bin_chunks)}-{max(current_bin_chunks)}",
                'chunks': list(current_bin_chunks)
            })
            current_bin_chunks = []
            current_bin_count = 0

    # Handle the last bin if it's not empty
    if current_bin_count > 0:
        binned_results.append({
            'num_chunks_range': f"{min(current_bin_chunks)}-{max(current_bin_chunks)}",
            'chunks': list(current_bin_chunks)
        })

    # --- Detailed analysis for each bin ---
    print(f"\n--- Accuracy vs. Reasoning Chain Length (Binned at >{bin_percent_threshold}%) ---")
    
    plot_data = []
    change_categories = ['0', '1', '2', '3+']

    for bin_info in binned_results:
        bin_df = df[df['num_chunks'].isin(bin_info['chunks'])]
        
        total_in_bin = len(bin_df)
        if total_in_bin == 0:
            continue

        print(f"\n--- Bin: {bin_info['num_chunks_range']} (Total: {total_in_bin} examples, {total_in_bin/total_examples*100:.1f}%) ---")
        
        breakdown = bin_df.groupby('num_changes_category')['overall_correct'].agg(['mean', 'count']).rename(columns={'mean': 'accuracy'})
        breakdown['percentage_of_bin'] = (breakdown['count'] / total_in_bin) * 100
        breakdown['percentage_of_total'] = (breakdown['count'] / total_examples) * 100
        
        # Ensure all categories are present for consistent plotting
        breakdown = breakdown.reindex(change_categories, fill_value=0)
        
        print(breakdown.to_string(float_format="%.2f"))

        for category, row in breakdown.iterrows():
            plot_data.append({
                'num_chunks_range': bin_info['num_chunks_range'],
                'num_changes_category': f"{category} changes",
                'accuracy': row['accuracy'],
                'count': row['count'],
                'percentage_of_bin': row['percentage_of_bin'],
                'percentage_of_total': row['percentage_of_total'],
            })

    print("-" * 60)
    
    if not plot_data:
        logger.info("No data to plot after binning.")
        return

    plot_df = pd.DataFrame(plot_data)

    # Enforce correct numerical sorting for the x-axis bins
    # The pivot operation sorts the index lexicographically ('10-12' comes before '2-5').
    # We fix this by converting the range column to a categorical type with the correct order.
    ordered_ranges = [b['num_chunks_range'] for b in binned_results]
    plot_df['num_chunks_range'] = pd.Categorical(
        plot_df['num_chunks_range'], categories=ordered_ranges, ordered=True
    )

    # --- Plotting Binned Results ---
    os.makedirs(out_dir, exist_ok=True)
    
    # --- New Stacked Bar Plot Logic ---
    # The total height of the bar is the bin's overall accuracy.
    # Each segment's height is its weighted contribution to that total accuracy.
    plot_df['total_in_bin'] = plot_df.groupby('num_chunks_range')['count'].transform('sum')
    # Calculate segment height: (accuracy * count) / total_count
    # This ensures sum of segments equals the bin's weighted average accuracy
    plot_df['segment_height'] = (plot_df['accuracy'] * plot_df['count']) / plot_df['total_in_bin']
    plot_df['segment_height'] = plot_df['segment_height'].fillna(0)

    # Pivot data for plotting
    segment_pivot = plot_df.pivot(
        index='num_chunks_range',
        columns='num_changes_category',
        values='segment_height'
    )
    accuracy_pivot = plot_df.pivot(
        index='num_chunks_range',
        columns='num_changes_category',
        values='accuracy'
    )
    count_pivot = plot_df.pivot(
        index='num_chunks_range',
        columns='num_changes_category',
        values='count'
    )

    # Ensure consistent column order
    category_order = [f"{i} changes" for i in ['0', '1', '2', '3+']]
    segment_pivot = segment_pivot.reindex(columns=category_order, fill_value=0)
    accuracy_pivot = accuracy_pivot.reindex(columns=category_order, fill_value=0)
    count_pivot = count_pivot.reindex(columns=category_order, fill_value=0)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    segment_pivot.plot(
        kind='bar',
        stacked=True,
        colormap='viridis',
        width=0.8,
        ax=ax
    )

    # Add text annotations inside each segment
    bottoms = np.zeros(len(segment_pivot))
    for col in segment_pivot.columns:
        heights = segment_pivot[col].fillna(0)
        accuracies = accuracy_pivot[col].fillna(0)
        counts = count_pivot[col].fillna(0)
        
        for j, (height, accuracy, count) in enumerate(zip(heights, accuracies, counts)):
            if height > 0.01:  # Only annotate non-trivial segments
                y_pos = bottoms[j] + height / 2
                label = f"{accuracy:.2f}\nn={int(count)}"
                fontsize = 8 if height < 0.05 else 9
                ax.text(j, y_pos, label, ha='center', va='center', color='white', fontsize=fontsize, weight='bold')
        
        bottoms += heights

    # Add text for total accuracy on top of each bar
    total_accuracies = segment_pivot.sum(axis=1)
    total_counts_per_bin = count_pivot.sum(axis=1)
    for j, bin_range in enumerate(total_accuracies.index):
        total_acc = total_accuracies.loc[bin_range]
        total_in_bin = total_counts_per_bin.loc[bin_range]
        percent_of_all = (total_in_bin / total_examples) * 100
        if total_acc > 0:
            label = f"{total_acc:.2f}\n(n={int(total_in_bin)}, {percent_of_all:.1f}%)"
            ax.text(j, total_acc + 0.015, label, ha='center', va='bottom', color='black', weight='bold', fontsize=9)

    plt.title(f'Stacked Accuracy vs. Reasoning Length (Segment Size Proportional to Contribution)')
    plt.xlabel('Range of Number of Chunks')
    plt.ylabel('Overall Accuracy')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, 1.1)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Num Changes', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()

    plot_filename = f"binned_accuracy_vs_length_by_changes_{source_table_name}_{model_name.replace('/', '_')}.png"
    plot_path = os.path.join(out_dir, plot_filename)
    plt.savefig(plot_path)
    logger.info(f"Binned plot saved to {plot_path}")
    plt.show()


if __name__ == "__main__":
    fire.Fire(run)
