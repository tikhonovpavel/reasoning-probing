import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import fire
from tqdm import tqdm
from matplotlib.ticker import MaxNLocator

from rc_utils import extract_think_content, split_reasoning_chain

# Apply a style for better plot aesthetics
plt.style.use('seaborn-v0_8-whitegrid')

def analyze_chunk_distribution(db_path: str = "reasoning_traces.sqlite", output_dir: str = "analysis_results/chunk_distribution"):
    """
    Analyzes the distribution of reasoning chunks in prompts from a SQLite database
    and generates histograms for each model and dataset combination.

    Args:
        db_path: Path to the SQLite database file.
        output_dir: Directory to save the generated plots.
    """
    if not os.path.exists(db_path):
        print(f"Error: Database file not found at {db_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    conn = sqlite3.connect(db_path)

    try:
        # Find all relevant tables
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'reasoning_traces_%';")
        tables = [row[0] for row in cursor.fetchall()]
        
        if not tables:
            print("No tables matching 'reasoning_traces_%' found in the database.")
            return

        print(f"Found tables: {tables}")

        for table in tables:
            print(f"\nProcessing table: {table}...")
            # Load data into pandas
            try:
                df = pd.read_sql_query(f"SELECT model_path, full_prompt_text FROM {table}", conn)
            except pd.io.sql.DatabaseError:
                print(f"Could not read from table {table}. It might be empty or missing required columns.")
                continue

            # Calculate number of chunks
            tqdm.pandas(desc=f"Calculating chunks for {table}")
            df['num_chunks'] = df['full_prompt_text'].progress_apply(
                lambda x: len(split_reasoning_chain(extract_think_content(x) or ""))
            )

            # Generate plots for each model in the current table
            models = df['model_path'].unique()
            print(f"Found models in {table}: {models}")

            for model in models:
                model_df = df[df['model_path'] == model]
                
                plt.figure(figsize=(12, 7))
                
                max_chunks = int(model_df['num_chunks'].max())
                
                if max_chunks > 20:
                    # If the number of unique chunk counts is high, group them into 20 bins.
                    sns.histplot(data=model_df, x='num_chunks', bins=20, kde=False)
                else:
                    # For smaller ranges, use discrete bins for each chunk count.
                    bins = range(max_chunks + 2)
                    sns.histplot(data=model_df, x='num_chunks', bins=bins, kde=False, discrete=True)
                    if max_chunks > 0:
                        plt.xticks(bins[:-1])

                plt.title(f'Distribution of Reasoning Chunks\nModel: {model} | Dataset: {table}')
                plt.xlabel('Number of Chunks')
                plt.ylabel('Frequency (Count)')

                # Add text annotations for mean, median, etc.
                mean_chunks = model_df['num_chunks'].mean()
                median_chunks = model_df['num_chunks'].median()
                std_chunks = model_df['num_chunks'].std()
                total_samples = len(model_df)

                stats_text = (
                    f"Total Samples: {total_samples}\n"
                    f"Mean: {mean_chunks:.2f}\n"
                    f"Median: {median_chunks:.0f}\n"
                    f"Std Dev: {std_chunks:.2f}\n"
                    f"Max: {max_chunks:.0f}"
                )
                
                plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
                         fontsize=10, verticalalignment='top', horizontalalignment='right',
                         bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))


                # Sanitize filename
                safe_model_name = model.replace('/', '_')
                plot_filename = f"{table}_{safe_model_name}.png"
                plot_path = os.path.join(output_dir, plot_filename)
                
                plt.tight_layout()
                plt.savefig(plot_path)
                plt.close()

                print(f"Saved plot to {plot_path}")

    finally:
        conn.close()

    print("\nAnalysis complete.")

if __name__ == "__main__":
    fire.Fire(analyze_chunk_distribution)
