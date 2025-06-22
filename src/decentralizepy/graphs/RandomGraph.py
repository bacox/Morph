import random
import networkx as nx
from decentralizepy.graphs.Graph import Graph

class RandomGraph(Graph):
    """
    Generates a random graph where each node has at least one neighbor.
    """

    def __init__(self, n_procs=None, edge_prob=0.1, seed=None):
        super().__init__(n_procs=n_procs)
        self.edge_prob = edge_prob
        self.seed = seed
        if n_procs is not None:
            self.generate_random_graph()

    def generate_random_graph(self):
        random.seed(self.seed)
        G = nx.erdos_renyi_graph(self.n_procs, self.edge_prob, seed=self.seed)

        # Ensure no isolated nodes
        for node in range(self.n_procs):
            if G.degree[node] == 0:
                # Connect to a random *different* node
                target = random.choice([i for i in range(self.n_procs) if i != node])
                G.add_edge(node, target)

        self.adj_list = [set() for _ in range(self.n_procs)]
        for u, v in G.edges():
            self.__insert_edge__(u, v)

