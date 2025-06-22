import os
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

def load_global_distribution(run_path, num_classes):
    global_dist = np.zeros(num_classes)
    for fname in os.listdir(run_path):
        if fname.startswith("data_dist_node_") and fname.endswith(".csv"):
            df = pd.read_csv(os.path.join(run_path, fname))
            global_dist += df["count"].values
    return global_dist / global_dist.sum()

def load_node_distributions(run_path, num_nodes, num_classes):
    dists = {}
    for i in range(num_nodes):
        df = pd.read_csv(os.path.join(run_path, f"data_dist_node_{i}.csv"))
        dists[i] = df.set_index("class")["count"].reindex(range(num_classes), fill_value=0).values
    return dists

def load_peer_counts(run_path, nid):
    counts_by_round = {}
    with open(os.path.join(run_path, f"{nid}_peer_model_counts_log.txt")) as f:
        prev = defaultdict(int)
        for line in f:
            round_str, rest = line.strip().split(": ", 1)
            round_id = int(round_str)
            delta_counts = {}
            for pair in rest.split(","):
                pid, cnt = pair.strip().split(":")
                pid = int(pid)
                cnt = int(cnt)
                delta = cnt - prev[pid]
                if delta > 0:
                    delta_counts[pid] = delta
                prev[pid] = cnt
            counts_by_round[round_id] = delta_counts
    return counts_by_round

def load_neighbors(run_path, nid):
    neighbors_by_round = {}
    with open(os.path.join(run_path, f"neighbors_{nid}.txt")) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            round_str, peers_str = line.strip().split(": ", 1)
            round_id = int(round_str)
            peers = list(map(int, peers_str.strip().split(", "))) if peers_str.strip() else []
            neighbors_by_round[round_id] = peers
    return neighbors_by_round

def process_run(run_path, num_nodes, num_classes):
    global_distribution = load_global_distribution(run_path, num_classes).reshape(1, -1)
    node_data = load_node_distributions(run_path, num_nodes, num_classes)

    all_similarities = []
    improvements = []

    for nid in range(num_nodes):
        local = node_data[nid].copy()
        cumulative = np.zeros_like(local, dtype=float)
        peer_counts = load_peer_counts(run_path, nid)
        neighbors = load_neighbors(run_path, nid)

        sim_per_round = []

        for rnd in sorted(peer_counts.keys()):
            deltas = peer_counts[rnd]
            cumulative += local.copy()
            for peer in neighbors.get(rnd, []):
                if peer in deltas:
                    peer_dist = node_data[peer]
                    cumulative += deltas[peer] * peer_dist
            sim = cosine_similarity(cumulative.reshape(1, -1), global_distribution)[0, 0]
            sim_per_round.append(sim)

        all_similarities.append(sim_per_round)
        if sim_per_round:
            improvements.append(sim_per_round[-1] - sim_per_round[0])

    return all_similarities, improvements

def analyze_multiple_runs(parent_dir, num_runs, num_nodes=16, num_classes=10):
    all_runs_similarities = []
    run_stats = []

    for run_id in range(1, num_runs + 1):
        run_path = os.path.join(parent_dir, f"run_{run_id}/machine0")
        sim_curves, improvements = process_run(run_path, num_nodes, num_classes)

        max_len = max(len(sim) for sim in sim_curves)
        for sim in sim_curves:
            sim += [sim[-1]] * (max_len - len(sim))

        arr = np.array(sim_curves)
        all_runs_similarities.append(arr)

        start_sims = arr[:, 0]
        final_sims = arr[:, -1]
        run_stats.append({
            "Run ID": f"run_{run_id}",
            "Mean Start Sim": np.mean(start_sims),
            "Mean Final Sim": np.mean(final_sims),
            "Avg Improvement": np.mean(improvements),
            "Std Improvement": np.std(improvements),
            "Min Final Sim": np.min(final_sims),
            "Max Final Sim": np.max(final_sims)
        })

    # Aggregate across runs
    avg_sim_per_round = []
    std_sim_per_round = []

    stacked = np.stack(all_runs_similarities)  # shape: (num_runs, num_nodes, num_rounds)
    num_rounds = stacked.shape[2]
    for rnd in range(num_rounds):
        vals = stacked[:, :, rnd].flatten()
        avg_sim_per_round.append(np.mean(vals))
        std_sim_per_round.append(np.std(vals))

    # Plot
    rounds = list(range(num_rounds))
    plt.figure(figsize=(12, 6))
    plt.plot(rounds, avg_sim_per_round, label="Average Similarity", color="blue", linewidth=3)
    plt.fill_between(
        rounds,
        np.array(avg_sim_per_round) - np.array(std_sim_per_round),
        np.array(avg_sim_per_round) + np.array(std_sim_per_round),
        color="blue", alpha=0.3, label="±1 Std Dev"
    )

    plt.title("Average Similarity to Global Distribution Over Time", fontsize=20)
    plt.xlabel("Round", fontsize=18)
    plt.ylabel("Cosine Similarity", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(parent_dir, "avg_similarity_all_runs.pdf"))
    plt.close()

    stats_df = pd.DataFrame(run_stats)
    stats_df.to_csv(os.path.join(parent_dir, "per_run_similarity_stats.csv"), index=False)

# Change folder for different experiment
analyze_multiple_runs("data/experiments/similarity/data_distribution/cifar_7", num_runs=5)
