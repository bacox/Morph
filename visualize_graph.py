import argparse
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path


def read_edges(file_path: Path):
    edges = []
    with file_path.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                edges.append(tuple(parts))
    return edges


def compute_graph_stats(G: nx.Graph):
    num_nodes = G.number_of_nodes()
    degrees = [d for _, d in G.degree()]
    min_conn = min(degrees) if degrees else 0
    max_conn = max(degrees) if degrees else 0
    avg_conn = sum(degrees) / len(degrees) if degrees else 0.0
    return num_nodes, min_conn, max_conn, avg_conn


def visualize_graph(edges, output_path: Path):
    G = nx.Graph()
    G.add_edges_from(edges)

    num_nodes, min_conn, max_conn, avg_conn = compute_graph_stats(G)

    plt.figure(figsize=(10, 8))
    # pos = nx.spring_layout(G, seed=42)
    pos = nx.kamada_kawai_layout(
        G,
    )  # Use random layout for better distribution

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="skyblue",
        edge_color="gray",
        node_size=500,
        font_size=10,
    )

    stats_text = (
        f"Num nodes: {num_nodes}\n" f"Min conn: {min_conn}\n" f"Max conn: {max_conn}\n" f"Avg conn: {avg_conn:.2f}"
    )

    plt.gca().text(
        1.02,
        0.5,
        stats_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Graph saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize a graph from a .edges file and save it as a PNG.")
    parser.add_argument("input_file", type=Path, help="Path to the input .edges file")

    args = parser.parse_args()
    input_path = args.input_file

    if not input_path.exists():
        print(f"Error: {input_path} does not exist.")
        return

    output_path = input_path.with_suffix(".png")
    edges = read_edges(input_path)
    visualize_graph(edges, output_path)


if __name__ == "__main__":
    main()
