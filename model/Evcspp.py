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
        active_nodes = [i for i in range(len(solution)) if solution[i] == 1]
        removable_nodes = []

        for node in active_nodes:
            temp_nodes = active_nodes.copy()
            temp_nodes.remove(node)

            if temp_nodes and nx.is_connected(g.subgraph(temp_nodes)):
                removable_nodes.append(node)

        return removable_nodes

    def greedy_algorithm(self, alpha: float, verbose=False):
        """
        Executes the greedy algorithm to solve the EVCSPP
        """
        # Initialisation de tous les nœuds actifs (tous les nœuds sont sélectionnés)
        x = np.ones(len(self.costs))

        # Vérifier si la solution initiale est valide
        if not self.is_demand_satisfied(x, alpha):
            print(
                "Le problème est infaisable : la solution initiale (1, 1, ..., 1) ne satisfait pas les demandes."
            )
            return np.zeros(len(self.costs))

        while True:
            # Création du graphe induit G^
            g = nx.Graph()
            active_nodes = [i for i in range(self.nodes) if x[i] == 1]
            g.add_nodes_from(active_nodes)
            for i in active_nodes:
                for j in active_nodes:
                    if self.adjacency[i, j] > 0 and self.shortest_paths[i, j] <= self.D:
                        g.add_edge(i, j)

            # Trouver les nœuds amovibles
            removable_nodes = self.find_removable_nodes(g, x)
            if verbose:
                print(f"Nœuds amovibles : {removable_nodes}")

            if not removable_nodes:
                if verbose:
                    print("Aucun nœud amovible trouvé, sortie de la boucle.")
                break

            # Sélectionner le nœud à supprimer en fonction du coût
            node_to_remove = max(removable_nodes, key=lambda node: self.costs[node])
            if verbose:
                print(f"Suppression du nœud : {node_to_remove}")
            x[node_to_remove] = 0

            # Vérifier si la nouvelle solution est valide
            if not self.is_demand_satisfied(x, alpha):
                if verbose:
                    print(
                        f"La solution n'est plus valide après la suppression du nœud {node_to_remove}, annulation de la suppression."
                    )
                x[node_to_remove] = 1
                break

        if verbose:
            print("Solution finale trouvée.")
        return x

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
