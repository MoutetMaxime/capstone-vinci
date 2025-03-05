import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class EVCSPP:
    """
    Class to solve the Electric Vehicle Charging Station Placement Problem (EVCSPP).
    """

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
        self.selected_nodes = set(range(self.nodes))

    def get_close_indices(self, i: int, alpha: float):
        """
        Returns the indices of the nodes that are close (distance <= alpha*D) to node i.
        """
        return set(np.where(self.shortest_paths[i] <= alpha * self.D)[0])

    def is_demand_satisfied(self, solution: np.array, alpha: float):
        """
        Checks if the demand constraint is satisfied for the given solution.
        """
        for i in range(self.nodes):
            if (
                sum(
                    [
                        self.capacities[node] * solution[node]
                        for node in self.get_close_indices(i, alpha)
                    ]
                )
                < self.demands[i]  # Demand is not satisfied
            ):
                return False
        return True

    @staticmethod
    def find_removable_nodes(g: nx.graph, solution: np.array):
        """
        Finds the removable nodes in the solution that can be removed while keeping the graph connected.
        """
        active_nodes = [i for i in range(len(solution)) if solution[i] >= 1]
        # shuffle active nodes to avoid always removing the same node
        # np.random.shuffle(active_nodes)
        removable_nodes = []

        for node in active_nodes:
            temp_nodes = active_nodes.copy()
            temp_nodes.remove(node)

            if temp_nodes and nx.is_connected(g.subgraph(temp_nodes)):
                removable_nodes.append(node)
        print(removable_nodes)
        return removable_nodes

    def greedy_algorithm(self, alpha: float, k: int, verbose=False):
        """
        Executes the greedy algorithm to solve the EVCSPP.
        Starts with all nodes set to 0 and adds nodes until the demand is satisfied.
        The maximum number of nodes that can be set to 1 is `k`.
        """
        # Initialize all nodes to 0
        solution = np.zeros(self.nodes, dtype=int)

        # Continue adding nodes until the demand is satisfied or we reach the maximum number of nodes
        while np.sum(solution) < k:
            # Find the best node to add
            best_node = None
            best_improvement = -np.inf

            for node in range(self.nodes):
                if solution[node] == 0:
                    # Temporarily set the node to 1
                    solution[node] = 1

                    # Calculate the improvement in demand satisfaction
                    improvement = self.calculate_demand_satisfaction(solution, alpha)

                    # Revert the change
                    solution[node] = 0

                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_node = node

            # If no node improves the solution, break
            if best_node is None:
                break

            # Add the best node to the solution
            solution[best_node] = 1

            # Check if the demand is satisfied
            if self.is_demand_satisfied(solution, alpha):
                break

        return solution

    def calculate_demand_satisfaction(self, solution: np.array, alpha: float):
        """
        Calculates the total demand satisfaction for the given solution.
        """
        total_satisfaction = 0
        for i in range(self.nodes):
            total_satisfaction += min(
                sum(
                    [
                        self.capacities[node] * solution[node]
                        for node in self.get_close_indices(i, alpha)
                    ]
                ),
                self.demands[i],
            )
        return total_satisfaction

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

    def display_charging_position(self, solution, labels=None):
        if labels is None:
            labels = list(
                range(len(solution))
            )  # Utilise les indices si aucun label n'est fourni

        # Création d'un mapping entre indices et labels
        label_mapping = {i: labels[i] for i in range(len(labels))}

        # Coloration des nœuds
        node_colors = [
            "red" if solution[i] == 1 else "lightblue" for i in range(len(solution))
        ]

        plt.figure(figsize=(5, 5))
        pos = nx.spring_layout(self.graph)  # Définit la disposition des nœuds

        # Dessiner le graphe avec des nœuds renommés par les labels
        nx.draw(
            self.graph,
            pos,
            labels=label_mapping,  # Appliquer les labels aux nœuds
            with_labels=True,
            node_color=node_colors,
            edge_color="gray",
            node_size=500,
            font_size=12,
        )

        plt.show()
