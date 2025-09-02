import os
import json
from collections import defaultdict

def migrate_results():
    """
    Migrates data from a single all_experiments_results.jsonl file into
    separate results.jsonl files within each experiment's timestamped directory.
    """
    results_dir = "results"
    master_file_path = os.path.join(results_dir, "all_experiments_results.jsonl")

    if not os.path.exists(master_file_path):
        print(f"Master file not found at {master_file_path}. Nothing to migrate.")
        return

    # 1. Find all existing run directories and sort them chronologically
    try:
        run_dirs = sorted([
            d for d in os.listdir(results_dir)
            if os.path.isdir(os.path.join(results_dir, d))
        ])
    except FileNotFoundError:
        print(f"Results directory '{results_dir}' not found. Nothing to migrate.")
        return
        
    if not run_dirs:
        print("No run directories found in 'results/'. Nothing to migrate.")
        return

    print(f"Found {len(run_dirs)} experiment directories.")

    # 2. Read the master file and group results by run
    runs = []
    current_run = []
    with open(master_file_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                # Heuristic: A new run starts with 'exp1' or if a run has no name (older format)
                is_start_of_new_run = (
                    "experiment_name" in data and data["experiment_name"].startswith("exp1")
                ) or "experiment_name" not in data
                
                if is_start_of_new_run and current_run:
                    runs.append(current_run)
                    current_run = []
                
                current_run.append(data)
            except json.JSONDecodeError:
                print(f"Warning: Skipping malformed line in {master_file_path}")

    if current_run:
        runs.append(current_run)

    print(f"Found {len(runs)} distinct runs in {master_file_path}.")

    # 3. Match runs to directories and write the new files
    if len(runs) != len(run_dirs):
        print("\nWARNING: The number of detected runs does not match the number of directories.")
        print(f"Runs: {len(runs)}, Directories: {len(run_dirs)}")
        print("Proceeding with migration for the smaller of the two counts.")

    num_to_migrate = min(len(runs), len(run_dirs))
    migrated_count = 0
    for i in range(num_to_migrate):
        run_dir = run_dirs[i]
        run_data = runs[i]
        
        output_path = os.path.join(results_dir, run_dir, "results.jsonl")
        
        with open(output_path, 'w') as f_out:
            for record in run_data:
                f_out.write(json.dumps(record) + "\n")
        
        print(f"  -> Migrated {len(run_data)} records to {output_path}")
        migrated_count += 1
        
    print(f"\nMigration complete. {migrated_count} directories now have a results.jsonl file.")

    # 4. Suggest renaming the old file
    backup_path = master_file_path + ".bak"
    print(f"\nIt is recommended to rename the old master file to avoid conflicts:")
    print(f"  mv {master_file_path} {backup_path}")


if __name__ == "__main__":
    migrate_results()
