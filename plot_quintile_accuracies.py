import neptune
import matplotlib.pyplot as plt
import fire
import os
import numpy as np
from matplotlib.ticker import MaxNLocator

def plot_quintiles(
    run_id: str = "PROB-40",
    project: str = "pashocles/probing-reasoning-classifier",
    output_dir: str = "plots",
    max_epochs: int = None,
):
    """
    Fetches quintile accuracy data from a Neptune run and plots it on a single graph.

    Args:
        run_id (str): The ID of the Neptune run (e.g., "PROB-36").
        project (str): The Neptune project name in the format "workspace/project-name".
        output_dir (str): Directory to save the plot.
        max_epochs (int, optional): The maximum number of epochs to display on the x-axis. Defaults to None.
    """
    print(f"Connecting to Neptune to fetch run '{run_id}' from project '{project}'...")
    
    run = neptune.init_run(
        project=project,
        with_id=run_id,
        mode="read-only",
    )

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 5))
    
    max_step = 0
    all_plot_accuracies = [] # To store all accuracy values for ylim
    markers = ['o', 's', 'v', '^', 'D'] # Different markers for each quintile
    
    # Fetch and plot each quintile
    for i in range(1, 6):
        metric_name = f"val/accuracy_quintile_{i}"
        try:
            # fetch_values() returns a pandas DataFrame
            series = run[metric_name].fetch_values()
            if not series.empty:
                # The training script logs pre-training validation at epoch -1 (step 0)
                # and then for each epoch from 0 to N-1 (steps 1 to N).
                # We label x-axis as "Epoch", where "Epoch 0" is pre-training.
                steps = series['step']
                accuracies = series['value']

                if max_epochs is not None:
                    mask = steps <= max_epochs
                    steps = steps[mask]
                    accuracies = accuracies[mask]
                
                if steps.empty:
                    continue
                
                all_plot_accuracies.extend(accuracies)
                
                # Use a custom label for step 0
                step_labels = [f"Epoch {int(s)}" for s in steps]
                step_labels[0] = "Pre-train" if steps.iloc[0] == 0 else step_labels[0]

                ax.plot(steps, accuracies, marker=markers[i-1], linestyle='-', label=f"Quintile {i} ({(i-1)*20}-{(i)*20}%)")
                
                if steps.max() > max_step:
                    max_step = steps.max()

            else:
                 print(f"Metric '{metric_name}' is empty or not found in the run.")

        except neptune.exceptions.NeptuneException as e:
            print(f"Could not fetch metric '{metric_name}': {e}")
            continue

    # Fetch tags for the title
    try:
        tags = run["sys/tags"].fetch()
        tags_str = f", Tags: {', '.join([t for t in tags if t != 'train_from_file'])}" if tags else ""
    except neptune.exceptions.NeptuneException as e:
        print(f"Could not fetch tags for run '{run_id}': {e}")
        tags_str = ""

    # Customize and show plot
    ax.set_title(f"Validation Accuracy Quintiles Over Epochs\n(Run: {run_id}{tags_str})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend(title="Reasoning Trace Position")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Ensure x-axis has integer ticks for epochs
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if max_epochs is not None:
        ax.set_xlim(-0.5, max_epochs + 0.5)

    # Dynamically set y-axis limits based on the data range
    if all_plot_accuracies:
        min_acc = min(all_plot_accuracies)
        max_acc = max(all_plot_accuracies)
        padding = (max_acc - min_acc) * 0.05  # 5% padding on top and bottom
        ax.set_ylim(min_acc - padding, max_acc + padding)
    else:
        ax.set_ylim(0, 1) # Default if no data was plotted

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    plot_filename = os.path.join(output_dir, f"quintile_accuracies_{run_id}.png")
    plt.tight_layout()
    plt.savefig(plot_filename)
    print(f"Plot saved successfully as {plot_filename}")
    
    # Stop the run object
    run.stop()
    print("Neptune run stopped.")

    # Show the plot
    plt.show()

if __name__ == "__main__":
    fire.Fire(plot_quintiles)
