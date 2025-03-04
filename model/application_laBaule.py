from urllib.parse import urlencode

import geopandas as gpd
import numpy as np
import requests
from Evcspp import EVCSPP
from utils import construct_adjacency, create_nodes, nodes_iris


def get_demands_from_pop(nodes):
    return np.array([node["pop"] for node in nodes])


def get_demands(nodes):
    return np.array([node["demand"] for node in nodes])


def get_costs_by_charge(nodes):
    implantation_cost = 60000 / (365 * 15)  # 60000€ / 15 years
    pi = 20  # €/kWh
    demand = get_demands_from_pop(nodes)
    costs = implantation_cost - pi * demand
    return costs


def get_costs_by_kWh(nodes):
    implantation_cost = 60000 / (365 * 15)  # 60000€ / 15 years
    pi = 0.40  # €/kWh
    demand = get_demands(nodes)
    costs = implantation_cost + pi * demand
    return costs


def get_capacities(nodes):
    return np.array([1 / node["density"] * 1e6 for node in nodes])


# nodes = create_nodes("../data/inputs_1km.csv")
nodes = nodes_iris
demands = get_demands_from_pop(nodes)
costs = get_costs_by_charge(nodes)
capacities = get_capacities(nodes)
# adjacency = np.load("adjacency_iris.npy")
adjacency = construct_adjacency(nodes)
with open("adjacency_iris.npy", "wb") as f:
    np.save(f, adjacency)

# Paramètres
print(demands)
print(costs)
print(capacities)
print(adjacency)


# Ds = np.linspace(10, 60, 10)
# alphas = np.linspace(0.1, 0.6, 20)
# results = []

# for D in Ds:
#     for alpha in alphas:
#         print(f"D = {D}, alpha = {alpha}")
#         solution = EVCSPP(adjacency, costs, demands, capacities, D)
#         result = solution.greedy_algorithm(alpha)
#         optimized_func = -costs @ result
#         print("Fonction objectif optimisée :", optimized_func)
#         print("Nombre de stations :", np.sum(result))

#         # Save
#         results.append(
#             {
#                 "D": D,
#                 "alpha": alpha,
#                 "result": result,
#                 "optimized_func": optimized_func,
#                 "nb_stations": np.sum(result),
#             }
#         )

# # Save results
# import pandas as pd

# df = pd.DataFrame(results)
# df.to_csv("results_1km_profit.csv", index=False)
