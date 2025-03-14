# import libraries
import os
import pandas as pd
import numpy as np
import geopandas as gpd
import os
from shapely.geometry import LineString, Point

from utils_data import get_contours_city_ign, get_iris_contours_city, add_population_iris, get_vehicles_city 
from utils_data import get_adjacency_from_gdf, get_centroids_communes, get_contours_city_simplifie, get_df_OD_city, restrict_area, get_routing, get_traffic_demand_per_iris
from utils_data import merge_iris_pop_and_traffic, add_revenus_iris, add_area_gdf, compute_routes_OD

data_dir = '../data'

# Input files
file_ign_region = f'{data_dir}/BDCARTO/44_Loire_Atlantique/data.gpkg'
file_iris_contours = f"{data_dir}/IRIS/contours-iris.gpkg"
file_iris_population = f"{data_dir}/IRIS/base-ic-logement-2021.xlsx"
file_iris_revenus = f"{data_dir}/IRIS/iris_revenus.csv"

file_OD = f'{data_dir}/RP2020_MOBPRO_csv/FD_MOBPRO_2020.csv'
file_contours_simplifies = f'{data_dir}/communes-version-simplifiee.geojson'
file_bornes = f'{data_dir}/consolidation-etalab-schema-irve-statique-v-2.3.1-20250129.csv'

dir_ev = f"{data_dir}/vehicules"
file_vp = f"{dir_ev}/parc_vp_com_2011_2024.xlsx"
file_vul = f"{dir_ev}/parc_vul_com_2011_2024.xlsx"
file_pl = f"{dir_ev}/parc_pl_com2011_2024.xlsx"
file_tcp = f"{dir_ev}/parc_tcp_commune_2024.xlsx"
files_vehicules = {'vp': file_vp, 'vul': file_vul, 'pl': file_pl, 'tcp': file_tcp}

# Nos parametres
code_insee = 44055
crs_angles='EPSG:4326'
crs_meters='EPSG:2154'
waiting_time_api=0.1

avg_dist_hab = 30
conso_km = 0.2
avg_demand_hab = avg_dist_hab * conso_km

restricted_distance_km = 150
ratio_ev_france = 0.03 
conso_kwh_km = 0.2

# Pour les bornes 
vinci_names_operateur = ['Easy Charge | FR*ECH', 'Easycharge services']

# A parametrer selon le code de la ville
processed_dir = f'{data_dir}/processed/{code_insee}'
preprocessed_dir = f'{data_dir}/preprocessed/{code_insee}'
os.makedirs(processed_dir, exist_ok=True)
os.makedirs(preprocessed_dir, exist_ok=True)

# Vehicles files
preprocessed_file_vp = f"{preprocessed_dir}/vp.xlsx"
preprocessed_file_vul = f"{preprocessed_dir}/vul.xlsx"
preprocessed_file_pl = f"{preprocessed_dir}/pl.xlsx"
preprocessed_file_tcp = f"{preprocessed_dir}/tcp.xlsx"
processed_files_vehicules = {'vp': preprocessed_file_vp, 'vul': preprocessed_file_vul, 'pl': preprocessed_file_pl, 'tcp': preprocessed_file_tcp}
export_file_vehicles = f"{preprocessed_dir}/df_vehicles_2024.csv"

# Preprocessed OD file
export_file_OD_routes = f"{preprocessed_dir}/processed_OD.parquet"

# Export file for IRIS
export_dir_iris = f"{processed_dir}/iris"
os.makedirs(export_dir_iris, exist_ok=True)

export_file_iris_pop_demand = f"{export_dir_iris}/demand_per_hab.geojson"
export_file_iris_traffic_demand = f"{export_dir_iris}/traffic_demand.geojson"
export_file_iris_all_demand = f"{export_dir_iris}/all_demand.geojson"
export_file_df_adjacency=f"{export_dir_iris}/gdf_city_iris_with_dist.csv"
export_file_df_adjacency_gpkg = f"{export_dir_iris}/gdf_city_with_dist.gpkg"
export_file_np_adjacency=f"{export_dir_iris}/adjacency_matrix.npy"
export_file_cs_iris = f"{export_dir_iris}/cs.gpkg" 

# Creation des fichiers
gdf_ign_city, geometry_ign_city = get_contours_city_ign(file_ign_region, code_insee)
gdf_iris_raw = get_iris_contours_city(file_iris_contours, code_insee)

# ADD POPULATION DEMAND
if not os.path.exists(export_file_iris_pop_demand):
    print(f"Computing demand related to population...")
    gdf_iris = add_area_gdf(gdf_iris_raw, crs_meters=crs_meters, crs_angles=crs_angles)
    gdf_iris = add_revenus_iris(file_iris_revenus, gdf_iris)
    gdf_iris_population = add_population_iris(file_iris_population, gdf_iris)
    gdf_iris_population['density_km2'] = gdf_iris_population['population'] / gdf_iris_population['area_km2']

    pop_city = gdf_iris_population.population.sum() # population de la ville à partir de IRIS
    df_vehicules = get_vehicles_city(code_insee, files_vehicules, processed_files_vehicules, export_file_vehicles)
    ratio_ev_pop = ((df_vehicules['NB_EL']+df_vehicules['NB_RECHARGEABLE'])/pop_city).iloc[0]
    gdf_iris_population['demand_pop_kWh'] = gdf_iris_population['population'] * ratio_ev_pop * gdf_iris_population['ratio_appart'] * avg_demand_hab
    gdf_iris_population.drop(columns=['ratio_appart', 'area_km2'], inplace=True)
    gdf_iris_population.to_file(export_file_iris_pop_demand, driver="GeoJSON")
    print(f"Demand related to population saved at : {export_file_iris_pop_demand}")
else:
    gdf_iris_population = gpd.read_file(export_file_iris_pop_demand)
    print(f"Demand related to pop loaded")

# ADD TRAFFIC DEMAND
if not os.path.exists(export_file_iris_traffic_demand):
    print(f"Computing demand related to traffic...")
    df_traffic_city = get_df_OD_city(file_OD, code_insee)
    gdf_contours_city, geometry_contours_city = get_contours_city_simplifie(file_contours_simplifies, code_insee)

    gdf_contours_city = gdf_contours_city.to_crs(crs_meters)
    gdf_contours_city['centroid'] = gdf_contours_city.geometry.centroid
    gdf_contours_city = gdf_contours_city.to_crs(crs_angles)
    gdf_contours_city['centroid'] = gdf_contours_city['centroid'].to_crs(crs_angles)
    
    centroid_city = gdf_contours_city.iloc[0].centroid
    gdf_full = get_centroids_communes(file_contours_simplifies, df_traffic_city, centroid_city)
    gdf_filtered, restricted_area = restrict_area(gdf_full, restricted_distance_km)

    # Get routes and distances using API
    if not os.path.exists(export_file_OD_routes):
        gdf_filtered = compute_routes_OD(gdf_filtered, export_file_OD_routes, waiting_time_api=0.1, crs_angles='EPSG:4326')
    else:
        gdf_filtered = gpd.read_parquet(export_file_OD_routes)

    gdf_iris_traffic = get_traffic_demand_per_iris(gdf_filtered, gdf_contours_city, gdf_iris, ratio_ev_france, conso_kwh_km)
    gdf_iris_traffic.to_file(export_file_iris_traffic_demand, driver="GeoJSON")
    
    print(f"Demand related to traffic saved at : {export_file_iris_traffic_demand}")
else:
    gdf_iris_traffic = gpd.read_file(export_file_iris_traffic_demand)
    print(f"Demand related to traffic loaded")

# COMBINE BOTH DEMANDS 
if not os.path.exists(export_file_iris_all_demand):    
    print(f"Adding demand related to traffic and population")
    gdf_iris_all_demands = merge_iris_pop_and_traffic(gdf_iris_population, gdf_iris_traffic)
    print(f"Demand related to traffic+pop saved at : {export_file_iris_all_demand}")
    gdf_iris_all_demands.to_file(export_file_iris_all_demand, driver="GeoJSON")
else:
    gdf_iris_all_demands = gpd.read_file(export_file_iris_all_demand)
    print(f"Demand related to traffic+pop loaded")


# GET NUMBER OF STATIONS PER IRIS AND ADD TO THE DATASET
if not os.path.exists(export_file_cs_iris):    
    
    # Load bornes dataset
    df_bornes = pd.read_csv(file_bornes)

    # Convert latitude & longitude to a geometry column
    df_bornes['geometry'] = df_bornes.apply(lambda row: Point(row['consolidated_longitude'], row['consolidated_latitude']), axis=1)
    gdf_bornes = gpd.GeoDataFrame(df_bornes, geometry='geometry')
    gdf_bornes.set_crs(crs_angles, inplace=True)

    # Filter to keep the bornes inside the city
    gdf_bornes_city = gdf_bornes[gdf_bornes.geometry.intersects(geometry_contours_city)]

    # Ajout d'un booléen qui indique si la borne appartient à Vinci ou non
    gdf_bornes_city['bornes_vinci'] = gdf_bornes_city['nom_operateur'].isin(vinci_names_operateur).astype(int)
    gdf_bornes_city = gdf_bornes_city[['bornes_vinci', 'puissance_nominale', 'geometry']]
    gdf_bornes_city = gdf_bornes_city.reset_index(drop=True) 

    # Récupère les bornes qui sont dans la ville
    gdf_bornes_iris = gpd.sjoin(gdf_bornes_city, gdf_iris_raw, how="inner", predicate="within")

    # Group by IRIS code and Vinci status, counting the number of rows instead of summing 'nbre_pdc'
    summary_iris_df = (
        gdf_bornes_iris.groupby(["code_iris", "bornes_vinci"])
        .size()  # Count occurrences instead of summing 'nbre_pdc'
        .unstack(fill_value=0)  # Fill missing values with 0
        .reset_index()
    )

    # Rename columns to match the required format
    summary_iris_df = summary_iris_df.rename(columns={0: "concu_nb_pdc", 1: "vinci_nb_pdc"})

    # Merge the counts back with the original 'gdf_iris_raw' dataframe
    final_df_cs = pd.merge(gdf_iris_raw, summary_iris_df, how="left", on="code_iris")

    # Fill NaN values in the merged DataFrame with 0 (for the charging station counts)
    final_df_cs.fillna({'concu_nb_pdc': 0, 'vinci_nb_pdc': 0}, inplace=True)

    # Add a column for the total number of charging stations (not charging points)
    final_df_cs['nb_pdc'] = final_df_cs['concu_nb_pdc'] + final_df_cs['vinci_nb_pdc']

    # Ensure the final DataFrame remains a GeoDataFrame
    final_df_cs = gpd.GeoDataFrame(final_df_cs, geometry='geometry', crs=gdf_iris.crs)

    final_df_cs.to_file(export_file_cs_iris, driver="GPKG")
else:
    final_df_cs = gpd.read_file(export_file_cs_iris)

final_df_cs = final_df_cs[['code_iris', 'concu_nb_pdc', 'vinci_nb_pdc', 'nb_pdc']]
gdf_iris_all_demands = gdf_iris_all_demands.merge(final_df_cs, on="code_iris", how="left")


# ADD ADJACENCY MATRIX, CLEAN AND SAVE
if not os.path.exists(export_file_np_adjacency) or not os.path.exists(export_file_df_adjacency_gpkg):
    print(f"Getting distance between IRIS centroids...")
    gdf_city_iris_with_dist, adjacency_matrix = get_adjacency_from_gdf(gdf_iris_all_demands, delay=waiting_time_api, crs_angles=crs_angles, crs_meters=crs_meters)
    
    # Clean columns
    cols = ['nom_iris', 'geometry', 'population', 'density_km2', 'revenues', 'demand_pop_kWh', 'demand_traffic_kWh', 'total_demand_kWh']
    cols += ['concu_nb_pdc', 'vinci_nb_pdc', 'nb_pdc']
    cols += [f'dist_{i}' for i in range(len(gdf_city_iris_with_dist))]

    gdf_city_iris_with_dist = gdf_city_iris_with_dist[cols]
    # gdf_city_iris_with_dist.to_csv(export_file_df_adjacency, index=False)
    
    gdf_city_iris_with_dist.to_file(export_file_df_adjacency_gpkg, driver="GPKG")
    print(f"Dataframe saved at {export_file_df_adjacency_gpkg}")
    
    np.save(export_file_np_adjacency, adjacency_matrix)
    print(f"Adjacency matrix alone saved at {export_file_np_adjacency}")
else:
    print(f"Final file already available at {export_file_df_adjacency_gpkg}")