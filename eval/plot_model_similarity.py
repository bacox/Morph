import os
import json
import glob
import matplotlib.pyplot as plt
import numpy as np

def plot_multiple_similarity_curves(parent_dir, output_path="combined_similarity_plot.pdf"):
    plt.figure(figsize=(8, 5))
    plotted = False

    for subdir in sorted(os.listdir(parent_dir)):
        full_path = os.path.join(parent_dir, subdir)
        if not os.path.isdir(full_path):
            continue

        degree = ''.join(filter(str.isdigit, subdir))
        label = f"Degree {degree}"

        all_runs = []
        all_iterations = set()

        for run_id in range(1, 6):
            run_path = os.path.join(full_path, f"run_{run_id}", "machine0")
            for file in glob.glob(os.path.join(run_path, "*_dissim_scores.json")):
                with open(file, "r") as f:
                    data = json.load(f)
                    if "local" in data and isinstance(data["local"], dict):
                        local_data = {int(k): v for k, v in data["local"].items()}
                        all_iterations.update(local_data.keys())
                        all_runs.append(local_data)

        if not all_runs:
            print(f"Warning: No valid data in {subdir}")
            continue

        all_iterations = sorted(all_iterations)
        iteration_index = {round_num: i for i, round_num in enumerate(all_iterations)}
        num_rounds = len(all_iterations)
        padded = np.full((len(all_runs), num_rounds), np.nan)

        for i, run in enumerate(all_runs):
            for round_num, val in run.items():
                idx = iteration_index[round_num]
                padded[i, idx] = val

        mean_vals = np.nanmean(padded, axis=0)
        std_vals = np.nanstd(padded, axis=0)

        plt.plot(all_iterations, mean_vals, label=label, linewidth=2)
        plt.fill_between(all_iterations, mean_vals - std_vals, mean_vals + std_vals, alpha=0.2)

        plotted = True

    if not plotted:
        raise RuntimeError("No valid similarity data found in any subdirectories.")

    plt.xlabel("Communication Rounds")
    plt.ylabel("Cosine Similarity to Global Model")
    plt.title("Similarity to Global Model for Different Degrees")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(parent_dir, output_path))
    print(f"Saved combined plot to {os.path.join(parent_dir, output_path)}")

plot_multiple_similarity_curves("data/experiments/similarity/model/")
