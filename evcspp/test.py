import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

nb_nodes = 5
construction_costs = [1, 2, 3, 1, 1]
demands = [3, 2, 4, 2, 1]
adjacency_matrix = np.array(
    [
        [0, 1, 1, 1, 0],
        [1, 0, 0, 1, 1],
        [1, 0, 0, 1, 0],
        [1, 1, 1, 0, 1],
        [0, 1, 0, 1, 0],
    ]
)
G = nx.from_numpy_array(np.array(adjacency_matrix))


def get_shortest_paths(adjacency_matrix: np.array):
    num_nodes = len(adjacency_matrix)
    shortest_path_matrix = np.full((num_nodes, num_nodes), np.inf)
    for source, paths in nx.all_pairs_shortest_path_length(G):
        for target, length in paths.items():
            shortest_path_matrix[source, target] = length
    return shortest_path_matrix


def demand_satisfied(active_nodes, demands, capacities, solution):
    """
    Constraint (6b)

    Args:

    """
    sum([capacities[node] * solution[node] for node in close_nodes])


def greedy_algorithm(G, construction_costs):
    x = np.ones(len(construction_costs))  # Initialisation de tous les nœuds actifs

    while np.any(x == 1):
        # Construire l'ensemble N contenant les nœuds actifs
        active_nodes = [i for i in range(len(x)) if x[i] == 1]
        subgraph = G.subgraph(active_nodes)

        x_prime = x.copy()
        flag = 0

        while True:
            # Sélectionner le nœud j avec le plus grand coût de construction
            j = max(active_nodes, key=lambda n: construction_costs[n])

            # Supprimer j temporairement
            x_prime[j] = 0

            # Vérifier si le graphe reste connecté après suppression de j
            if nx.is_connected(
                G.subgraph([i for i in range(len(x)) if x_prime[i] == 1])
            ):
                x = x_prime.copy()
                flag = 1
            else:
                x_prime[j] = 1  # Réactiver j si la suppression casse la connectivité

            active_nodes.remove(j)

            if flag == 1 or not active_nodes:
                break

    return x


greedy_solution = greedy_algorithm(G, construction_costs)
print("Solution de l'algorithme glouton:", greedy_solution)


# plt.figure(figsize=(5, 5))
# nx.draw(
#     G,
#     with_labels=True,
#     node_color="lightblue",
#     edge_color="gray",
#     node_size=500,
#     font_size=12,
# )
# plt.show()
