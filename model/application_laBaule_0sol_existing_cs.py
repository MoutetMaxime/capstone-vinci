from urllib.parse import urlencode
import geopandas as gpd
import numpy as np
import requests
from Evcspp_0sol_existing_cs import EVCSPP
from utils import create_nodes


def get_demands_from_pop(gdf):
    return np.array(gdf["population"])

def get_demands_kWh(gdf):
    return np.array(gdf["total_demand_kWh"])

# Problème ici on suppose que la demande de la zone est complètement captée par la borne
# On pourrait plutôt mettre un cout fixe et maximiser la demande captée

# def get_costs_by_charge(nodes):
#     implantation_cost = 60000 / (365 * 15)  # 60000€ / 15 years
#     pi = 20  # €/personne
#     demand = get_demands_from_pop(nodes)
#     costs = implantation_cost - pi * demand
#     return costs

# def get_costs_by_kWh(nodes):
#     implantation_cost = 60000 / (365 * 15)  # 60000€ / 15 years
#     pi = 0.25  # €/kWh à fournir
#     demand = get_demands(nodes)
#     costs = implantation_cost - pi * demand
#     return costs

def get_fixed_cost(gdf):
    implantation_cost = np.array([60000 / (365 * 15) for i in range(len(gdf))])  # 60000€ / 15 years
    return implantation_cost

# 1e6 proportionality constant used in the article (see table)
def get_capacities(gdf):
    return np.array(1 / gdf["density_km2"] * 1e6)

# 12h de service continu avec 22kW
def get_capacity_kWh(gdf, puissance=22):
    C = puissance * 12
    return np.array([C for i in range(len(gdf))])

def get_existing_cs(gdf):
    return np.array(gdf["nb_pdc"])

input_file = "../data/processed/44055/200/gdf_city_with_dist.gpkg"
adjacency_file = "../data/processed/44055/200/adjacency_matrix.npy"
export_file = "results_eliott/existing_stations.csv"

# nodes est une list de dictionnaire avec les attributs qui permettent de récupérer la demande, la capacité, ...
gdf = gpd.read_file(input_file)

# Demande en population ou en kWh
# demands = get_demands_from_pop(gdf)
demands = get_demands_kWh(gdf)


# Coûts définis de plusieurs manières
costs = get_fixed_cost(gdf)
# costs = get_costs_by_charge(nodes)
# costs = get_costs_by_kWh(nodes)


# Capacité définie de plusieurs manières
# capacities = get_capacities(gdf)
capacities = get_capacity_kWh(gdf)

# Prendre en compte ou non les bornes existantes
# existing_cs = get_existing_cs(gdf)
existing_cs = np.array([0 for i in range(len(gdf))])

# Profit par kWh délivré
profit_per_kWh = 0.10 # 22kWh
# profit_per_kWh = 0.15 # 60kWh
# profit_per_kWh = 0.20 # 120kWh

adjacency = np.load(adjacency_file)

# Paramètres
# print(demands)
# print(costs)
# print(capacities)
# print(adjacency)

k = 10
Ds = np.linspace(10, 60, 10)
alphas = np.linspace(0.1, 0.6, 20)
results = []
Ds = [15]
alphas = [0.1]

print(existing_cs)

for D in Ds:
    for alpha in alphas:
        print(f"D = {D}, alpha = {alpha}, k = {k}")
        solution = EVCSPP(adjacency, costs, demands, capacities, existing_cs, D, profit_per_kWh)
        # result = solution.greedy_algorithm(alpha, k)
        result = solution.greedy_algorithm_profitable(alpha, k)
        reached_demand = solution.calculate_reached_demand(result, existing_cs, alpha)
        optimized_func = -costs @ result
        final_profit = reached_demand * profit_per_kWh - costs @ result 

        print("Solution: ", result)
        print("Reached demand: ", reached_demand)
        print("Fonction objectif optimisée :", optimized_func)
        print(f"Profit final : {final_profit}€/jour")
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
df.to_csv(export_file, index=False)
