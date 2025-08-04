from pathlib import Path
import json
import numpy as np
import argparse


def load_result_files(algorithm_folder):
    all_runs = []

    for run_folder in algorithm_folder.glob("2025-*/machine0"):
        run_data = {"test_acc": {}, "test_loss": {}, "total_bytes": {}}

        result_files = list(run_folder.glob("*_results.json"))
        if not result_files:
            continue

        for file in result_files:
            node_id = file.stem.split("_")[0]
            with open(file, "r") as f:
                node_data = json.load(f)
                for metric in run_data:
                    if metric in node_data:
                        for round_str, value in node_data[metric].items():
                            round_int = int(round_str)
                            if round_int not in run_data[metric]:
                                run_data[metric][round_int] = []
                            if isinstance(value, dict):
                                run_data[metric][round_int].extend(value.values())
                            else:
                                run_data[metric][round_int].append(value)
        all_runs.append(run_data)
    return all_runs


def compute_avg_metric_over_runs(runs, metric):
    avg_per_round = {}
    for run in runs:
        for rnd, values in run[metric].items():
            if rnd not in avg_per_round:
                avg_per_round[rnd] = []
            avg_per_round[rnd].append(np.mean(values))
    return {rnd: np.mean(vals) for rnd, vals in avg_per_round.items()}


def find_round_to_reach_target(metric_dict, target, mode="max"):
    for rnd in sorted(metric_dict.keys()):
        if (mode == "max" and metric_dict[rnd] >= target) or (mode == "min" and metric_dict[rnd] <= target):
            return rnd
    return None


def analyze_folder(base_path_str):
    base_path = Path(base_path_str)
    algorithms = [f for f in base_path.iterdir() if f.is_dir()]
    metrics_summary = {}
    top_accuracy_epidemic = 0
    top_loss_epidemic = float("inf")

    # First pass: find top acc/loss for each algorithm
    for algo in algorithms:
        print(f"Analyzing {algo.name}...")

        runs = load_result_files(algo)
        if not runs:
            print(f"No runs found for {algo.name}. Skipping...")
            continue

        avg_acc = compute_avg_metric_over_runs(runs, "test_acc")
        avg_loss = compute_avg_metric_over_runs(runs, "test_loss")

        max_acc = max(avg_acc.values())
        min_loss = min(avg_loss.values())

        metrics_summary[algo.name] = {
            "top_accuracy": max_acc,
            "top_loss": min_loss,
            "avg_acc_per_round": avg_acc,
            "avg_loss_per_round": avg_loss,
            "avg_bytes_per_round": compute_avg_metric_over_runs(runs, "total_bytes"),
        }

        if algo.name in ["epidemic", "EL"]:
            print(f"Top accuracy for {algo.name}: {max_acc}")
            print(f"Top loss for {algo.name}: {min_loss}")
            top_accuracy_epidemic = max_acc
            top_loss_epidemic = min_loss

    # Second pass: determine when others reach epidemic's top accuracy/loss
    comparison_targets = {}

    algos = [alg.name for alg in algorithms]
    print(f"Finding comparison targets... for algos: {algos}")

    # for algo in ["fully_connected", "diss_dl", "epidemic"]:
    for algo in algos:
        if algo not in metrics_summary:
            continue

        acc_round = find_round_to_reach_target(
            metrics_summary[algo]["avg_acc_per_round"], top_accuracy_epidemic, mode="max"
        )
        loss_round = find_round_to_reach_target(
            metrics_summary[algo]["avg_loss_per_round"], top_loss_epidemic, mode="min"
        )

        acc_bytes = metrics_summary[algo]["avg_bytes_per_round"].get(acc_round, 0) if acc_round is not None else None
        loss_bytes = metrics_summary[algo]["avg_bytes_per_round"].get(loss_round, 0) if loss_round is not None else None

        comparison_targets[algo] = {
            "round_to_reach_top_epidemic_acc": acc_round,
            "bytes_to_reach_top_epidemic_acc": acc_bytes,
            "round_to_reach_top_epidemic_loss": loss_round,
            "bytes_to_reach_top_epidemic_loss": loss_bytes,
        }

    output = {
        "top_accuracy_and_loss": {
            algo: {"top_accuracy": metrics_summary[algo]["top_accuracy"], "top_loss": metrics_summary[algo]["top_loss"]}
            for algo in metrics_summary
        },
        "comparison_to_epidemic": comparison_targets,
    }

    output_file = base_path / "evaluation_summary.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    return output_file


if __name__ == "__main__":
    # Change folder for different experiment
    parser = argparse.ArgumentParser(description="Analyze experiment results.")
    parser.add_argument(
        "--path", type=str, default="data/experiments/cifar/degree_3", help="Path to the experiment folder to analyze."
    )
    args = parser.parse_args()

    analyze_folder(args.path)
