import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# NOTES

# On suppose que toutes les bornes  ont la même capacité (les bornes existantes et celles qu'on veut placer)

# Dans le calcul de la demande servie: on compte plusieurs fois chaque borne comme source de capacité sans prendre en compte ce qu'elle a déjà délivré
# Ex: borne 0 peut servir la zone 0 avec une capacité C, puis elle peut servir la zone 1 avec une capacité C... => C'est faux en réalité

# Dans l'algo, tant qu'on peut placer une borne qui augmentera la demande totale captée on la place
# Problème: on peut ajouter une borne alors qu'elle coute + cher que ce qu'elle ne rapporte 
# Potentielle modif: n'ajouter une borne que si ce qu'elle rapporte est > threshold

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
        existing_cs: np.array,
        D: int,
        profit_per_kWh = 0.10
    ):
        self.adjacency = adjacency
        self.costs = costs
        self.demands = demands
        self.capacities = capacities
        self.n_nodes = len(adjacency)
        self.existing_cs = existing_cs
        self.D = D
        self.profit_per_kWh = profit_per_kWh

        assert (
            len(costs) == self.n_nodes
            and len(demands) == self.n_nodes
            and len(capacities) == self.n_nodes
        )

        self.graph = nx.from_numpy_array(adjacency)        
        self.shortest_paths = adjacency
        self.selected_nodes = set(range(self.n_nodes))

    def get_close_indices(self, i: int, alpha: float):
        """
        Returns the indices of the nodes that are close (distance <= alpha*D) to node i.
        """
        return set(np.where(self.shortest_paths[i] <= alpha * self.D)[0])

    def is_demand_satisfied(self, solution: np.array, existing_cs: np.array, alpha: float):
        """
        Checks if the demand constraint is satisfied for the given solution and the existing charging stations.
        """
        for i in range(self.n_nodes):
            if (
                sum(
                    [
                        self.capacities[idx_node] * (solution[idx_node] + existing_cs[idx_node])
                        for idx_node in self.get_close_indices(i, alpha)
                    ]
                )
                < self.demands[i]  # Demand is not satisfied
            ):
                return False
        return True

    def greedy_algorithm(self, alpha: float, k: int, verbose=False):
        """
        Executes the greedy algorithm to solve the EVCSPP.
        Starts with all nodes set to 0 and adds nodes until the demand is satisfied.
        The maximum number of charging points that we can put is k.
        """

        # Initialize all nodes to 0
        solution = np.zeros(self.n_nodes, dtype=int)
        
        if self.is_demand_satisfied(solution, self.existing_cs, alpha):
            print('The demand is already satisfied by the existing stations')
            return solution

        # Continue adding nodes until the demand is satisfied or we reach the maximum number of nodes
        while np.sum(solution) < k:
            
            print(f"Itération {np.sum(solution)}")

            # Find the best node to add
            best_node = None
            best_improvement = -np.inf
            
            for node in range(self.n_nodes):
               
                # Temporarily add one charging point
                solution[node] += 1
                
                # Calculate the improvement in demand satisfaction
                improvement = self.calculate_reached_demand(solution, self.existing_cs, alpha)
                
                # Revert the change
                solution[node] -= 1
                
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_node = node

            # If no node improves the solution, break
            if best_node is None:
                print("No new charging point could improve results")
                break

            # Add one charging point to the node
            solution[best_node] += 1

            # Check if the demand is satisfied
            if self.is_demand_satisfied(solution, self.existing_cs, alpha):
                print("The total demand is satisfied")
                break
        
        if np.sum(solution) == k:
            print("Maximum number of charging points was placed")
        
        return solution
    
    # Critère supplémentaire : on ajoute une nouvelle borne seulement si elle améliore le profit 
    def greedy_algorithm_profitable(self, alpha: float, k: int, verbose=False):
        """
        Executes the greedy algorithm to solve the EVCSPP.
        Starts with all nodes set to 0 and adds nodes until the profit stops increasing.
        The maximum number of charging points that we can put is k.
        """

        # Initialize all nodes to 0
        solution = np.zeros(self.n_nodes, dtype=int)
        best_profit = 0

        # Stop if the demand is already satisfied (as defined in HK) : not necessarly interesting
        # if self.is_demand_satisfied(solution, self.existing_cs, alpha):
        #     print('The demand is already 'satisfied' by the existing stations')
        #     return solution
        
        # Continue adding nodes until the demand is satisfied or we reach the maximum number of nodes
        while np.sum(solution) < k:
            
            print(f"Itération {np.sum(solution)}")

            # Find the best node to add in order to improve profit
            best_node_idx = None
            # best_profit = current_profit
            
            for i_node in range(self.n_nodes):
               
                # Temporarily add one charging point
                solution[i_node] += 1
                
                # Calculate the profit that comes with this new configuration
                # demand_reached = self.calculate_reached_demand(solution, self.existing_cs, alpha)
                demand_reached = self.calculate_reached_demand_saturated(solution, self.existing_cs, alpha)
                temp_profit = demand_reached*self.profit_per_kWh - self.costs @ solution

                # Revert the change
                solution[i_node] -= 1
                
                if temp_profit > best_profit:
                    best_profit = temp_profit
                    best_node_idx = i_node

            # If no node improves the solution, break
            if best_node_idx is None:
                print("Adding a new charging points would not increase profit")
                break

            # Add one charging point to the node
            print(f"Best node at step {np.sum(solution)} : {best_node_idx}")
            # print(f"Profit : {best_profit}")
            solution[best_node_idx] += 1
            
            # Stop if the demand is already satisfied (as defined in the HK paper): not necessarly interesting for us
            if self.is_demand_satisfied(solution, self.existing_cs, alpha):
                print("The total demand is 'satisfied'")
            #     break
        
        if np.sum(solution) == k:
            print("Maximum number of charging points was placed")
    

        return solution

    
    def calculate_reached_demand(self, solution: np.array, existing_cs: np.array, alpha: float):
        """
        Calculates the total reached demand for the given solution and with respect to existing charging stations.
        """
        our_reached_demand = 0
        
        for i in range(self.n_nodes):
            # capacité totale de nos bornes et des bornes existantes pour servir ce noeud 
            our_capacity = sum([self.capacities[idx_node] * solution[idx_node] for idx_node in self.get_close_indices(i, alpha)])
            existing_cs_capacity = sum([self.capacities[idx_node] * existing_cs[idx_node] for idx_node in self.get_close_indices(i, alpha)])

            # si la demande n'est pas saturée, chacun peut y capter son maximum
            if our_capacity + existing_cs_capacity <= self.demands[i]:
                our_reached_demand += our_capacity
            
            # si la demande est saturée, on suppose que la demande est répartie équitablement
            else:
                total_capacity = our_capacity + existing_cs_capacity
                our_reached_demand += self.demands[i] * (our_capacity/total_capacity)

        return our_reached_demand
    
    def calculate_reached_demand_saturated(self, solution: np.array, existing_cs: np.array, alpha: float):
        """
        Calculates the total reached demand for the given solution and with respect to existing charging stations.
        """
        our_reached_demand = 0
        
        for i in range(self.n_nodes):
            # capacité totale de nos bornes et des bornes existantes pour servir ce noeud 
            our_capacity = sum([self.capacities[idx_node] * solution[idx_node] for idx_node in self.get_close_indices(i, alpha)])
            existing_cs_capacity = sum([self.capacities[idx_node] * existing_cs[idx_node] for idx_node in self.get_close_indices(i, alpha)])

            # si la demande n'est pas saturée, chacun peut y capter son maximum
            if our_capacity + existing_cs_capacity <= self.demands[i]:
                our_reached_demand += our_capacity
            
            # si la demande est saturée, on suppose que la demande est répartie équitablement
            else:
                total_capacity = our_capacity + existing_cs_capacity
                our_reached_demand += self.demands[i] * (our_capacity/total_capacity)

        # on ne peut pas capter + de demande que notre capacité totale
        our_full_capacity = solution @ self.capacities
        our_reached_demand = min(our_reached_demand, our_full_capacity)

        return our_reached_demand

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


    # A mettre à jour pour indiquer le nb de bornes et si la borne est nouvelle ou non
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
