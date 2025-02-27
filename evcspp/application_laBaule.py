from urllib.parse import urlencode

import geopandas as gpd
import numpy as np
import requests
from class2 import Model
from utils_algo import construct_adjacency, create_nodes


def get_demands(nodes):
    return np.array([node["pop"] for node in nodes])


def get_costs(nodes):
    implantation_cost = 60000 / (365 * 15)  # 60000€ / 15 years
    pi = 20  # €/charge
    demand = get_demands(nodes)
    costs = implantation_cost - pi * demand
    return costs


def get_capacities(nodes):
    return np.array([1 / node["density"] * 1e6 for node in nodes])


nodes = create_nodes("../data/inputs_1km.csv")
demands = get_demands(nodes)
costs = get_costs(nodes)
capacities = get_capacities(nodes)
adjacency = construct_adjacency(nodes)
with open("adjacency.npy", "wb") as f:
    np.save(f, adjacency)

# Paramètres
print(demands)
print(costs)
print(capacities)
print(adjacency)


Ds = [10, 20, 30, 40, 50]
alphas = [0.2, 0.4, 0.6, 0.8, 1]
Ds = [10]
alphas = [0.2]
results = []

for D in Ds:
    for alpha in alphas:
        print(f"D = {D}, alpha = {alpha}")
        solution = Model(adjacency, costs, demands, capacities, D)
        result = solution.greedy_algorithm(alpha)
        optimized_func = costs @ result
        print("Fonction objectif optimisée :", optimized_func)
        print("Nombre de stations :", np.sum(result))

        # Save
        results.append(
            {
                "D": D,
                "alpha": alpha,
                "result": result,
                "optimized_func": optimized_func,
                "nb_stations": np.sum(result),
            }
        )

# Save results
import pandas as pd

df = pd.DataFrame(results)
df.to_csv("results_profit.csv", index=False)
