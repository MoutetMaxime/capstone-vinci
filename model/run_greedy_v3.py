import geopandas as gpd
import numpy as np
from model.greedy_v3 import EVCSPP
import os

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

zone = "200"
if zone=="200":
    input_file = "../data/processed/44055/200/gdf_city_with_dist.gpkg"
    adjacency_file = "../data/processed/44055/200/adjacency_matrix.npy"
elif zone =="1km":
    input_file = "../data/processed/44055/1km/gdf_city_with_dist.gpkg"
    adjacency_file = "../data/processed/44055/1km/adjacency_matrix.npy"
else:
    input_file = "../data/processed/44055/iris/gdf_city_with_dist.gpkg"
    adjacency_file = "../data/processed/44055/iris/adjacency_matrix.npy"



# nodes est une list de dictionnaire avec les attributs qui permettent de récupérer la demande, la capacité, ...
gdf = gpd.read_file(input_file)

# Coûts définis de plusieurs manières
costs = get_fixed_cost(gdf)

use_kWh = True
use_existing_cs = False
# use_existing_cs = True

if use_kWh:
    # Demande en kWh
    demands = get_demands_kWh(gdf)
    # costs = get_costs_by_kWh(nodes) # not used here
    capacities = get_capacity_kWh(gdf)

else:
    # Demande en population ou en kWh
    demands = get_demands_from_pop(gdf)
    # costs = get_costs_by_charge(nodes)
    capacities = get_capacities(gdf)


if use_existing_cs:
    # Prendre en compte les bornes existantes
    existing_cs = get_existing_cs(gdf)
else:
    # Ne pas prendre en compte les bornes existantes
    existing_cs = np.array([0 for i in range(len(gdf))])

# Profit par kWh délivré
profit_per_kWh = 0.10 # 22kWh
# profit_per_kWh = 0.15 # 60kWh
# profit_per_kWh = 0.20 # 120kWh

# Profit par recharge
profit_per_charge = 20

adjacency = np.load(adjacency_file)



k = 20
# Ds = np.linspace(10, 60, 10)
# alphas = np.linspace(0.1, 0.6, 20)

# Pour avoir alpha*D = [1km, 3km, 5km]
Ds = [15]
alphas = [1/15, 1/5, 1/3]

export_file = f"results_with_cs/profitable/{zone}_with_cs_{use_existing_cs}_{k}max_kWh_{use_kWh}.csv"
os.makedirs(os.path.dirname(export_file), exist_ok=True)

# Paramètres
# print(demands)
# print(costs)
# print(capacities)
# print(adjacency)

print(f"Use existing cs {use_existing_cs}")
print(f"Use kWh: {use_kWh}")

results = []
for D in Ds:
    for alpha in alphas:
        print(f"\n\nD = {D}, alpha = {alpha}, k = {k}")
        solution = EVCSPP(adjacency, costs, demands, capacities, existing_cs, D, profit_per_kWh)
        # result = solution.greedy_algorithm(alpha, k)
        result = solution.greedy_algorithm_profitable(alpha, k)
        # reached_demand = solution.calculate_reached_demand(result, existing_cs, alpha)
        reached_demand = solution.calculate_reached_demand_saturated(result, existing_cs, alpha)
        optimized_func = - costs @ result
        
        # Ici on calcule toujours pour kWh, à généraliser pour une demande en population
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
                 "k": k,
                 "result": result,
                 "optimized_func": optimized_func,
                 "nb_stations": np.sum(result),
                 "profit": final_profit, 
                 "existing_cs": existing_cs
             }
         )


# Save results
import pandas as pd

df = pd.DataFrame(results)
df.to_csv(export_file, index=False)
