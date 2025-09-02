import json
import os
from itertools import islice

def migrate_results(master_file="results/all_experiments_results.jsonl", output_dir="results"):
    """
    Migrates a single master JSONL result file into separate, run-specific directories.

    Reads the master file, groups lines by 3 (assuming each run has 3 experiments),
    and creates a new directory for each run containing its own results.jsonl file.

    Args:
        master_file (str): Path to the master all_experiments_results.jsonl file.
        output_dir (str): The root directory where new migrated run folders will be created.
    """
    if not os.path.exists(master_file):
        print(f"Error: Master file not found at '{master_file}'. Nothing to migrate.")
        return

    print(f"Starting migration for '{master_file}'...")
    
    run_counter = 0
    lines_migrated = 0
    
    with open(master_file, 'r') as f:
        while True:
            # Read 3 lines at a time
            chunk = list(islice(f, 3))
            if not chunk:
                break

            run_counter += 1
            
            # Sanity check the chunk
            if len(chunk) < 3:
                print(f"Warning: Found a partial chunk of {len(chunk)} lines at the end of the file. Skipping.")
                continue

            try:
                # Parse the first line to get metadata for the directory name
                first_line_data = json.loads(chunk[0])
                model_name = first_line_data.get("model_name", "unknown_model")
                model_name_safe = os.path.basename(model_name) # Handle names like "Qwen/QwQ-32B"
                
                # Create a new directory for this run
                new_run_dir_name = f"migrated_{model_name_safe}_run_{run_counter}"
                new_run_dir_path = os.path.join(output_dir, new_run_dir_name)
                os.makedirs(new_run_dir_path, exist_ok=True)

                # Write the 3 lines to the new results file
                new_results_file_path = os.path.join(new_run_dir_path, "results.jsonl")
                with open(new_results_file_path, 'w') as out_f:
                    for line in chunk:
                        out_f.write(line)
                
                lines_migrated += len(chunk)
                print(f"  - Created '{new_results_file_path}' for run {run_counter}.")

            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse JSON in a chunk for run {run_counter}. Error: {e}. Skipping.")
                continue
            except Exception as e:
                print(f"An unexpected error occurred processing a chunk for run {run_counter}: {e}. Skipping.")
                continue

    print(f"\nMigration complete. Migrated {lines_migrated} lines from {run_counter} runs.")
    print(f"New directories with results.jsonl files have been created in '{output_dir}'.")
    print(f"You may now want to review the new directories and safely delete the old '{master_file}' file.")


if __name__ == "__main__":
    import fire
    fire.Fire(migrate_results)
