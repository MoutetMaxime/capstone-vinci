import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class Model:
    def __init__(
        self,
        adjacency: np.array,
        costs: np.array,
        demands: np.array,
        capacities: np.array,
        D: int,
    ):
        self.adjacency = adjacency
        self.costs = costs
        self.demands = demands
        self.capacities = capacities

        self.nodes = len(adjacency)
        self.D = D

        assert (
            len(costs) == self.nodes
            and len(demands) == self.nodes
            and len(capacities) == self.nodes
        )

        self.graph = nx.from_numpy_array(adjacency)

        shortest_path_matrix = np.full((self.nodes, self.nodes), np.inf)
        for source, paths in nx.all_pairs_dijkstra_path_length(
            self.graph, weight="weight"
        ):
            for target, length in paths.items():
                shortest_path_matrix[source, target] = length

        self.shortest_paths = shortest_path_matrix

    def get_close_indices(self, i, alpha):
        close_nodes = set()
        for j in range(self.nodes):
            if self.shortest_paths[i, j] <= alpha * self.D:
                close_nodes.add(j)
        return close_nodes

    def is_demand_satisfied(self, solution, alpha):
        for i in range(self.nodes):
            if (
                sum(
                    [
                        self.capacities[node] * solution[node]
                        for node in self.get_close_indices(i, alpha)
                    ]
                )
                < self.demands[i]
            ):
                return False
        return True

    @staticmethod
    def find_removable_nodes(g, solution):
        active_nodes = [i for i in range(len(solution)) if solution[i] == 1]
        removable_nodes = []

        for node in active_nodes:
            temp_nodes = active_nodes.copy()
            temp_nodes.remove(node)

            if temp_nodes and nx.is_connected(g.subgraph(temp_nodes)):
                removable_nodes.append(node)

        return removable_nodes

    def greedy_algorithm(self, alpha):
        x = np.ones(len(self.costs))  # Initialisation de tous les nœuds actifs

        while True:
            # Create G^
            g = nx.Graph()
            g.add_nodes_from([i for i in range(self.nodes) if x[i] == 1])
            for i in range(self.nodes):
                for j in range(i + 1, self.nodes):
                    if self.shortest_paths[i, j] <= self.D:
                        g.add_edge(i, j)

            active_nodes = self.find_removable_nodes(g, x)
            if len(active_nodes) == 0:
                return x

            x_prime = x.copy()
            flag = 0

            while True:
                j = max(active_nodes, key=lambda n: self.costs[n])
                x_prime[j] = 0

                if self.is_demand_satisfied(x_prime, alpha):
                    x = x_prime.copy()
                    flag = 1
                else:
                    x_prime[j] = 1
                    active_nodes.remove(j)

                if flag == 1 or not active_nodes:
                    break

    def is_solution(self, solution, alpha):
        isSolution = True

        # Objective (6a)
        objective = np.sum(self.costs * solution)

        # (6b, 6c)
        for i in range(self.nodes):
            closed = self.get_close_indices(i, alpha)
            if np.sum(self.capacities[closed] * solution[closed]) < self.demands[i]:
                isSolution = False
            if solution[i] not in [0, 1]:
                isSolution = False

        # (6c)
        # for i in range(self.nodes):

    def display_graph(self):
        plt.figure(figsize=(5, 5))
        nx.draw(
            self.graph,
            with_labels=True,
            node_color="lightblue",
            edge_color="gray",
            node_size=500,
            font_size=12,
        )
        plt.show()

    def display_charging_position(self, solution):
        node_colors = [
            "red" if solution[i] == 1 else "lightblue" for i in range(len(solution))
        ]

        plt.figure(figsize=(5, 5))
        nx.draw(
            self.graph,
            with_labels=True,
            node_color=node_colors,
            edge_color="gray",
            node_size=500,
            font_size=12,
        )
        plt.show()


if __name__ == "__main__":
    construction_costs = [0.11, 0.41, 0.30, 0.1, 0.2, 0.5, 0.35, 0.4, 0.2]
    capacities = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    demands = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    adjacency_matrix = np.array(
        [
            [0, 1, 5, 1, 0, 1, 0, 0, 0],
            [1, 0, 0, 1, 1, 0, 1, 0, 0],
            [5, 0, 0, 1, 0, 0, 0, 1, 0],
            [1, 1, 1, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
        ]
    )
    d = 1

    solution = Model(adjacency_matrix, construction_costs, demands, capacities, d)

    # print(solution.display_graph())
    # print(solution.get_close_indices(0, 0.5))
    print(solution.greedy_algorithm(1))
