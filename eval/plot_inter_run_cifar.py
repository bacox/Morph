from datetime import datetime
from pathlib import Path
import json
from typing import Callable, List, Union
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# base_dirs = [
#     Path("../eval_remote/st1/data/nca/CIFAR10/regular_100_14"),
#     Path("../eval_remote/st3/data/nca/CIFAR10/regular_100_14"),
# ]

# style_map = {
#     "diss_dl": {"label": "DissDL", "color": "#d62728", "linestyle": "-"},
#     "epidemic": {"label": "EL-Local", "color": "#004D4D", "linestyle": "--"},
#     "fully_connected": {"label": "Fully Connected", "color": "#ff7f0e", "linestyle": "-."},
#     "static_mh": {"label": "Static MH", "color": "#1f77b4", "linestyle": ":"},
# }

# algs = 'gms', 'gds', 'EL', 'static'
style_map = {
    "gms": {"label": "GMS", "color": "#d62728", "linestyle": "-"},
    "gds": {"label": "GDS", "color": "#004D4D", "linestyle": "--"},
    "EL": {"label": "EL-Local", "color": "#ff7f0e", "linestyle": "-."},
    "static": {"label": "Static MH", "color": "#1f77b4", "linestyle": ":"},
}

# name_map = {"diss_dl": "DissDL", "epidemic": "EL-Local", "fully_connected": "Fully Connected", "static_mh": "Static MH"}
# name_map = {"diss_dl": "DissDL", "epidemic": "EL-Local", "fully_connected": "Fully Connected", "static_mh": "Static MH"}
name_map = {"gms": "GMS", "gds": "GDS", "EL": "EL-Local", "static": "Static MH"}


def interpolate_round_metrics_to_time(mean_dict, std_dict, time_mapping):
    times = []
    means = []
    stds = []
    for r in sorted(mean_dict.keys()):
        if str(r) in time_mapping:
            times.append(time_mapping[str(r)])
            means.append(mean_dict[r])
            stds.append(std_dict[r])
    return np.array(times), np.array(means), np.array(stds)


def load_run_metrics(result_file):
    with result_file.open("r") as f:
        return json.load(f)


def aggregate_algorithm_runs(algorithm_folder):
    all_runs = []
    for run_folder in algorithm_folder.glob("2025-*/machine0"):
        run_data = defaultdict(lambda: defaultdict(dict))
        result_files = list(run_folder.glob("*_results.json"))
        if not result_files:
            continue
        for file in result_files:
            node_id = file.stem.split("_")[0]
            node_data = load_run_metrics(file)
            for metric, rounds in node_data.items():
                for round_str, value in rounds.items():
                    run_data[metric][int(round_str)][node_id] = value
        all_runs.append(run_data)
    return all_runs


def compute_mean_std_across_runs(run_metrics_list, metric_name):
    round_values = defaultdict(list)
    for run_metrics in run_metrics_list:
        if metric_name not in run_metrics:
            continue
        for round_str, node_vals in run_metrics[metric_name].items():
            if isinstance(node_vals, dict):
                values = list(node_vals.values())
                avg_value = np.mean(values)
            else:
                avg_value = node_vals
            round_values[int(round_str)].append(avg_value)
    mean = {r: np.mean(vals) for r, vals in round_values.items()}
    std = {r: np.std(vals) for r, vals in round_values.items()}
    return mean, std


def compute_stability(run_metrics_list, metric_name):
    round_variances = defaultdict(list)
    for run_metrics in run_metrics_list:
        if metric_name not in run_metrics:
            continue
        for round_str, node_vals in run_metrics[metric_name].items():
            if isinstance(node_vals, dict):
                values = list(node_vals.values())
                variance = np.var(values)
                round_variances[int(round_str)].append(variance)
    stability = {r: np.mean(vars) for r, vars in round_variances.items()}
    return stability


def format_round_ticks(ax, fontsize):
    ticks = ax.get_xticks()
    ticks = [t for t in ticks if t >= 0]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(t)//1000}" for t in ticks], fontsize=fontsize, weight="bold")


def plot_metric(metric_data, metric_name, output_path, time_mappings, plot_over_time, export_as_pdf):
    plt.figure(figsize=(8, 6))
    font_size = 28 / 2
    for label, (mean_dict, std_dict) in metric_data.items():
        config = style_map.get(label, {"label": label, "color": "gray", "linestyle": "-"})
        label_clean = config["label"]
        color = config["color"]
        linestyle = config["linestyle"]

        if plot_over_time and label in time_mappings:
            time_mapping = time_mappings[label]
            x_vals, means, stds = interpolate_round_metrics_to_time(mean_dict, std_dict, time_mapping)
            xlabel = "Elapsed Time (s)"
        else:
            rounds = sorted(mean_dict.keys())
            x_vals = rounds
            means = np.array([mean_dict[r] for r in rounds])
            stds = np.array([std_dict[r] for r in rounds])
            xlabel = "Communication Rounds"

        plt.plot(x_vals, means, label=label_clean, color=color, linestyle=linestyle, linewidth=2.5)
        plt.fill_between(x_vals, means - stds, means + stds, color=color, alpha=0.2)
    assert len(metric_data), f"xlabel must be defined for plotting; {len(metric_data)=}"
    if xlabel == "Communication Rounds":
        plt.xlabel("Communication Rounds ($\\times 10^3$)", fontsize=font_size, weight="bold")
        format_round_ticks(plt.gca(), font_size)
    else:
        plt.xlabel(xlabel, fontsize=font_size, weight="bold")

    plt.ylabel(metric_name.replace("_", " ").title(), fontsize=font_size, weight="bold")
    plt.title(f"{metric_name.replace('_', ' ').title()} (Mean ± Std)", fontsize=font_size + 2, weight="bold")
    plt.legend(fontsize=font_size - 6)
    plt.grid(True, color="#cccccc", linewidth=0.8)
    # plt.tight_layout(pad=3)
    plt.tight_layout()
    plt.tick_params(labelsize=font_size - 6, width=2)

    fname = f"{metric_name}_time" if plot_over_time else metric_name
    ext = ".pdf" if export_as_pdf else ".png"
    plt.savefig(output_path / f"{fname}{ext}", bbox_inches="tight")
    plt.close()


def plot_stability(stability_data, output_path, export_as_pdf):
    plt.figure(figsize=(8, 6))
    font_size = 28 / 2
    ax = plt.gca()
    for label, stability in stability_data.items():
        rounds = sorted(stability.keys())
        values = [stability[r] for r in rounds]
        config = style_map.get(label, {"label": label, "color": "gray", "linestyle": "-"})
        ax.plot(
            rounds, values, label=config["label"], color=config["color"], linestyle=config["linestyle"], linewidth=2.5
        )

    ax.set_xlabel("Communication Rounds ($\\times 10^3$)", fontsize=font_size, weight="bold")
    format_round_ticks(ax, font_size)
    ax.set_ylabel("Inter-node Variance", fontsize=font_size, weight="bold")
    ax.set_title("Stability (Average Inter-node Variance)", fontsize=font_size + 2, weight="bold")
    ax.legend(fontsize=font_size - 6)
    ax.grid(True, color="#cccccc", linewidth=0.8)
    ax.tick_params(labelsize=font_size - 6, width=2)
    plt.tight_layout()
    ext = ".pdf" if export_as_pdf else ".png"
    plt.savefig(output_path / f"stability{ext}", bbox_inches="tight")
    plt.close()


def plot_bytes(metric_data, output_path, time_mappings, plot_over_time, export_as_pdf):
    plt.figure(figsize=(8, 6))
    font_size = 28 / 2
    for label, (mean_dict, std_dict) in metric_data.items():
        config = style_map.get(label, {"label": label, "color": "gray", "linestyle": "-"})
        label_clean = config["label"]
        color = config["color"]
        linestyle = config["linestyle"]

        if plot_over_time and label in time_mappings:
            time_mapping = time_mappings[label]
            x_vals, means, stds = interpolate_round_metrics_to_time(mean_dict, std_dict, time_mapping)
            xlabel = "Elapsed Time (s)"
        else:
            rounds = sorted(mean_dict.keys())
            x_vals = rounds
            means = np.array([mean_dict[r] for r in rounds])
            stds = np.array([std_dict[r] for r in rounds])
            xlabel = "Communication Rounds"

        plt.plot(x_vals, means, label=label_clean, color=color, linestyle=linestyle, linewidth=2.5)
        plt.fill_between(x_vals, means - stds, means + stds, color=color, alpha=0.2)

    if xlabel == "Communication Rounds":
        plt.xlabel("Communication Rounds ($\\times 10^3$)", fontsize=font_size, weight="bold")
        format_round_ticks(plt.gca(), font_size)
    else:
        plt.xlabel(xlabel, fontsize=font_size, weight="bold")

    plt.ylabel("Bytes Sent", fontsize=font_size, weight="bold")
    # plt.title("Mean Communication Cost", fontsize=font_size + 2, weight="bold")
    plt.legend(fontsize=font_size - 6)
    plt.grid(True, color="#cccccc", linewidth=0.8)
    plt.tight_layout()
    plt.tick_params(labelsize=font_size - 6, width=2)

    ext = ".pdf" if export_as_pdf else ".png"
    plt.savefig(output_path / f"communication_cost{ext}", bbox_inches="tight")
    plt.close()


def plot_combined_metrics(metric_data, stability_data, output_path, time_mappings, plot_over_time, export_as_pdf):
    fig, axes = plt.subplots(1, 3, figsize=(30, 8))  # Removed 4th plot
    titles = {"test_acc": "Test Accuracy", "test_loss": "Test Loss", "stability": "Inter-node Variance"}
    ylabels = {"test_acc": "Accuracy", "test_loss": "Loss", "stability": "Variance"}
    font_size = 28
    for idx, (metric, ax) in enumerate(zip(["test_acc", "test_loss", "stability"], axes.flat)):
        if metric == "stability":
            for label, stability in stability_data.items():
                config = style_map.get(label, {"label": label, "color": "gray", "linestyle": "-"})
                rounds = sorted(stability.keys())
                values = [stability[r] for r in rounds]
                ax.plot(
                    rounds,
                    values,
                    label=config["label"],
                    color=config["color"],
                    linestyle=config["linestyle"],
                    linewidth=2.5,
                )
            xlabel = "Communication Rounds"
        else:
            for label, (mean_dict, std_dict) in metric_data[metric].items():
                config = style_map.get(label, {"label": label, "color": "gray", "linestyle": "-"})
                label_clean = config["label"]
                color = config["color"]
                linestyle = config["linestyle"]
                if plot_over_time and label in time_mappings:
                    time_mapping = time_mappings[label]
                    x_vals, means, stds = interpolate_round_metrics_to_time(mean_dict, std_dict, time_mapping)
                    xlabel = "Elapsed Time (s)"
                else:
                    rounds = sorted(mean_dict.keys())
                    x_vals = rounds
                    means = np.array([mean_dict[r] for r in rounds])
                    stds = np.array([std_dict[r] for r in rounds])
                    xlabel = "Communication Rounds"
                ax.plot(x_vals, means, label=label_clean, color=color, linestyle=linestyle, linewidth=2.5)
                ax.fill_between(x_vals, means - stds, means + stds, color=color, alpha=0.2)

        if xlabel == "Communication Rounds":
            ax.set_xlabel("Communication Rounds ($\\times 10^3$)", fontsize=font_size, weight="bold")
            format_round_ticks(ax, font_size)
        else:
            ax.set_xlabel(xlabel, fontsize=font_size, weight="bold")

        ax.set_title(titles[metric], fontsize=font_size + 2, weight="bold")
        ax.set_ylabel(ylabels[metric], fontsize=font_size, weight="bold")
        ax.tick_params(labelsize=font_size - 6, width=2)
        ax.grid(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05), fontsize=font_size, ncol=len(labels))
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    # plt.tight_layout()
    ext = ".pdf" if export_as_pdf else ".png"
    plt.savefig(output_path / f"combined_metrics{ext}", bbox_inches="tight")
    plt.close()


def extract_experiment_name(path: Path, name_callable: Callable = lambda x: "") -> str:
    return name_callable(path.__str__())


def main(
    base_dirs: Union[Path, List[Path]],
    plot_over_time: bool = False,
    export_as_pdf: bool = True,
    name_callable: Callable = lambda x: "",
):

    if isinstance(base_dirs, Path):
        base_dirs = [base_dirs]

    assert isinstance(base_dirs, list), "base_dirs must be a list"
    assert all(isinstance(d, Path) for d in base_dirs), "All base directories must be Path objects."

    exp_name = extract_experiment_name(base_dirs[0], name_callable)

    print(f"Experiment Name: {exp_name}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path("graphs") / exp_name / timestamp
    output_path.mkdir(parents=True, exist_ok=True)

    algorithms = []

    for base_dir in base_dirs:
        if not base_dir.exists():
            print(f"Base directory {base_dir} does not exist. Skipping...")
            continue
        algorithms.extend([f for f in base_dir.iterdir() if f.is_dir()])

    print(f"Found {len(algorithms)} algorithms in base directories: {[algo.name for algo in algorithms]}")

    # directories  = [f for f in base_dir if f.is_dir() ]

    # algorithms = [f for f in base_dir.iterdir() if f.is_dir()]
    metrics_to_plot = ["test_acc", "test_loss"]

    all_metric_data = {m: {} for m in metrics_to_plot}
    stability_data = {}
    total_bytes_data = {}
    time_mappings = {}

    for algo_dir in algorithms:
        label = algo_dir.name
        print(f"Processing {label}... and alg_dir: {algo_dir}")
        run_metrics_list = aggregate_algorithm_runs(algo_dir)
        if not run_metrics_list:
            # print(f"No runs found for {label}. Skipping...")
            continue
        if plot_over_time:
            time_path = algo_dir / "average_time_mapping.json"
            if time_path.exists():
                with open(time_path, "r") as f:
                    time_mappings[label] = json.load(f)
            else:
                # print(f"No time mapping found for {label}. Skipping...")
                continue
        for metric in metrics_to_plot:
            mean, std = compute_mean_std_across_runs(run_metrics_list, metric)
            all_metric_data[metric][label] = (mean, std)
        mean_bytes, std_bytes = compute_mean_std_across_runs(run_metrics_list, "total_bytes")
        total_bytes_data[label] = (mean_bytes, std_bytes)
        stability_data[label] = compute_stability(run_metrics_list, "test_acc")

        # print(f"{stability_data[label]=}")
        # exit()
    export_options = [True, False]
    plot_skipped = False
    for export_as_pdf in export_options:
        for metric, data in all_metric_data.items():
            # assert len(data) > 0, f"No metric data found to plot for metric {metric}."
            plot_metric(data, metric, output_path, time_mappings, plot_over_time, export_as_pdf)
        plot_stability(stability_data, output_path, export_as_pdf)
        plot_combined_metrics(
            all_metric_data, stability_data, output_path, time_mappings, plot_over_time, export_as_pdf
        )
        plot_bytes(total_bytes_data, output_path, time_mappings, plot_over_time, export_as_pdf)

    print(f"[DONE] All plots saved in {output_path.resolve()}")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Plot inter-run metrics for CIFAR experiments.")

    # App positional argument for base directories
    parser.add_argument(
        "base_dir",
        type=Path,
        help="List of base directories containing the experiment data.",
    )

    # Set optional arguments for the locations. It can be multiple directories
    parser.add_argument(
        "--locations",
        nargs="+",
        type=Path,
        default=[Path("../eval_remote/st1"), Path("../eval_remote/st3")],
        help="List of base directories containing the experiment data.",
    )
    args = parser.parse_args()
    # print(f"Base directories provided: {args.base_dir}")
    # print(f"{args=}")
    # exit()

    plot_over_time = False
    # Change folder for different experiment or set export_as_pdf to False for exporting in png
    export_as_pdf = True
    # base_dir = Path("data/experiments/cifar/degree_3")
    # base_dir = Path("../eval_remote/st1/data/nca/CIFAR10/regular_100_3")
    # base_dirs = [
    #     Path("../eval_remote/st1/data/nca/CIFAR10/fully-connected_100"),
    #     Path("../eval_remote/st3/data/nca/CIFAR10/fully-connected_100"),
    # ]

    # base_dirs = [
    #     Path("../eval_remote/st1/data/nca/CIFAR10/regular_100_7"),
    #     Path("../eval_remote/st3/data/nca/CIFAR10/regular_100_7"),
    # ]

    base_dirs = [x / "data" / args.base_dir for x in args.locations]

    # print(f"Base directories: {base_dirs}")

    # exit()

    # Lambda to split string by '/' and take the last 3 parts and join by '_'
    lambda_split = lambda x: "_".join(x.split("/")[-3:])

    main(base_dirs, plot_over_time, export_as_pdf, lambda_split)
