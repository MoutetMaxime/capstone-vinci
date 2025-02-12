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
        self.selected_nodes = set(range(self.nodes))  # Tous les nœuds sont sélectionnés au départ
    
    def print_distances(self):
        print("Distances entre chaque nœud :")
        for i in range(self.nodes):
            for j in range(self.nodes):
                print(f"Distance de {i} à {j} : {self.shortest_paths[i, j]}")


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
        # Initialisation de tous les nœuds actifs (tous les nœuds sont sélectionnés)
        x = np.ones(len(self.costs))

        # Vérifier si la solution initiale est valide
        if not self.is_demand_satisfied(x, alpha):
            print("Le problème est infaisable : la solution initiale (1, 1, ..., 1) ne satisfait pas les demandes.")
            return None

        while True:
            # Création du graphe induit G^
            g = nx.Graph()
            active_nodes = [i for i in range(self.nodes) if x[i] == 1]
            g.add_nodes_from(active_nodes)
            for i in active_nodes:
                for j in active_nodes:
                    if self.adjacency[i, j] > 0:
                        g.add_edge(i, j, weight=self.adjacency[i, j])

            # Trouver les nœuds amovibles
            removable_nodes = self.find_removable_nodes(g, x)
            print(f"Nœuds amovibles : {removable_nodes}")

            if not removable_nodes:
                print("Aucun nœud amovible trouvé, sortie de la boucle.")
                break

            # Sélectionner le nœud à supprimer en fonction du coût
            node_to_remove = max(removable_nodes, key=lambda node: self.costs[node])
            print(f"Suppression du nœud : {node_to_remove}")
            x[node_to_remove] = 0

            # Vérifier si la nouvelle solution est valide
            if not self.is_demand_satisfied(x, alpha):
                print(f"La solution n'est plus valide après la suppression du nœud {node_to_remove}, annulation de la suppression.")
                x[node_to_remove] = 1
                break

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
    adjacency_matrix2 = np.array(
        [
            [0, 1, 1, 1, 0, 1, 0, 0, 0],
            [1, 0, 0, 1, 1, 0, 1, 0, 0],
            [1, 0, 0, 1, 0, 0, 0, 1, 0],
            [1, 1, 1, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
        ]
    )
    d = 2

    solution = Model(adjacency_matrix2, construction_costs, demands, capacities, d)
    solution.print_distances()
    # Afficher le graphe
    solution.display_graph()

    # Exécuter l'algorithme glouton
    result = solution.greedy_algorithm(1)
    print("Solution finale :", result)

    # Afficher les positions des stations de recharge
    if result is not None:
        solution.display_charging_position(result)
