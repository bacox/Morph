import networkx as nx

from decentralizepy.graphs.Graph import Graph


class Regular(Graph):
    """
    The class for generating a Regular topology

    """

    def __init__(self, n_procs, degree, seed=None):
        """
        Constructor. Generates a Ring graph

        Parameters
        ----------
        n_procs : int
            total number of nodes in the graph
        degree : int
            Neighbors of each node

        """
        super().__init__(n_procs)
        print(f"Generating Regular graph with {n_procs} nodes and degree {degree} and seed {seed}")
        G = nx.random_regular_graph(degree, n_procs, seed)
        adj = G.adjacency()
        for i, l in adj:
            self.adj_list[i] = set()  # a_new set
            for k in l:
                self.adj_list[i].add(k)
                self.__insert_edge__(i, k)
        if not nx.is_connected(G):
            # raise ValueError("The generated graph is not connected.")
            self.connect_graph()
