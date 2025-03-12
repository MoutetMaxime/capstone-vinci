# import libraries
import os
import pandas as pd
import numpy as np
import geopandas as gpd
import os
from shapely.geometry import LineString

from utils_data import get_contours_city_ign, get_iris_contours_city, add_population_iris, get_vehicles_city 
from utils_data import get_adjacency_from_gdf, get_centroids_communes, get_contours_city_simplifie, get_df_OD_city, restrict_area, get_routing, get_traffic_demand_per_iris
from utils_data import merge_iris_pop_and_traffic, add_revenus_iris, add_area_gdf

data_dir = '../data'

# Input files
file_ign_region = f'{data_dir}/BDCARTO/44_Loire_Atlantique/data.gpkg'
file_iris_contours = f"{data_dir}/IRIS/contours-iris.gpkg"
file_iris_population = f"{data_dir}/IRIS/base-ic-logement-2021.xlsx"
file_iris_revenus = f"{data_dir}/IRIS/iris_revenus.csv"

file_OD = f'{data_dir}/RP2020_MOBPRO_csv/FD_MOBPRO_2020.csv'
file_contours_simplifies = f'{data_dir}/communes-version-simplifiee.geojson'

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


# A parametrer selon le code de la ville
processed_dir = f'{data_dir}/processed'
preprocessed_dir = f'{data_dir}/preprocessed'
os.makedirs(processed_dir, exist_ok=True)
os.makedirs(preprocessed_dir, exist_ok=True)

preprocessed_file_vp = f"{preprocessed_dir}/vp_{code_insee}.xlsx"
preprocessed_file_vul = f"{preprocessed_dir}/vul_{code_insee}.xlsx"
preprocessed_file_pl = f"{preprocessed_dir}/pl_{code_insee}.xlsx"
preprocessed_file_tcp = f"{preprocessed_dir}/tcp_{code_insee}.xlsx"
processed_files_vehicules = {'vp': preprocessed_file_vp, 'vul': preprocessed_file_vul, 'pl': preprocessed_file_pl, 'tcp': preprocessed_file_tcp}

export_file_vehicles = f"{processed_dir}/df_vehicles_{code_insee}_2024.csv"
export_file_iris_pop_demand = f"{processed_dir}/demand_per_iris_hab_{code_insee}.geojson"
export_file_iris_pop_demand = f"{processed_dir}/demand_per_iris_hab_{code_insee}.geojson"
export_file_iris_traffic_demand = f"{processed_dir}/traffic_demand_per_iris_{code_insee}.geojson"
export_file_iris_all_demand = f"{processed_dir}/all_demand_per_iris_{code_insee}.geojson"
export_file_df_adjacency=f"{processed_dir}/gdf_city_iris_with_dist_{code_insee}.csv"
export_file_np_adjacency=f"{processed_dir}/adjacency_matrix_iris_{code_insee}.npy"

# Creation des fichiers
gdf_ign_city, geometry_ign_city = get_contours_city_ign(file_ign_region, code_insee)
gdf_iris = get_iris_contours_city(file_iris_contours, code_insee)

    

# Load or compute population demand
if not os.path.exists(export_file_iris_pop_demand):
    print(f"Computing demand related to population...")
    gdf_iris = add_area_gdf(gdf_iris, crs_meters=crs_meters, crs_angles=crs_angles)
    gdf_iris = add_revenus_iris(file_iris_revenus, gdf_iris)
    gdf_iris_population = add_population_iris(file_iris_population, gdf_iris)
    gdf_iris_population['density_km2'] = gdf_iris_population['population'] / gdf_iris_population['area_km2']

    pop_city = gdf_iris_population.population.sum() # population de la ville à partir de IRIS
    df_vehicules = get_vehicles_city(code_insee, files_vehicules, processed_files_vehicules, export_file_vehicles)
    ratio_ev_pop = ((df_vehicules['NB_EL']+df_vehicules['NB_RECHARGEABLE'])/pop_city).iloc[0]
    gdf_iris_population['demand_pop_kWh'] = gdf_iris_population['population'] * ratio_ev_pop * gdf_iris_population['ratio_appart'] * avg_demand_hab
    # gdf_iris_population.drop(columns=['P21_PMEN', 'ratio_appart'], inplace=True)
    # gdf_iris_population.rename(columns={'P21_PMEN':'population'})
    gdf_iris_population.drop(columns=['ratio_appart', 'area_km2'], inplace=True)
    gdf_iris_population.to_file(export_file_iris_pop_demand, driver="GeoJSON")
    print(f"Demand related to population saved at : {export_file_iris_pop_demand}")
else:
    gdf_iris_population = gpd.read_file(export_file_iris_pop_demand)
    print(f"Demand related to pop loaded")

# Load of compute traffic demand
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

    # Get routes and distances between 
    gdf_filtered["routing"] = gdf_filtered.apply(lambda row: get_routing(row["centroid_hab"].coords[0], row["centroid_lt"].coords[0], delay=waiting_time_api), axis=1)
    gdf_filtered["route"] = gdf_filtered["routing"].apply(lambda row: LineString(row["geometry"]["coordinates"]))
    gdf_filtered["route"] = gpd.GeoSeries(gdf_filtered["route"], crs=crs_angles)
    gdf_filtered["distance_km"] = gdf_filtered["routing"].apply(lambda row: row["distance"])
    gdf_filtered.drop(columns=['routing'], inplace=True)
    gdf_filtered = gdf_filtered.set_geometry('route')

    gdf_iris_traffic, gdf_final_od = get_traffic_demand_per_iris(gdf_filtered, gdf_contours_city, gdf_iris, ratio_ev_france, conso_kwh_km)
    gdf_iris_traffic.to_file(export_file_iris_traffic_demand, driver="GeoJSON")
    
    print(f"Demand related to traffic saved at : {export_file_iris_traffic_demand}")
else:
    gdf_iris_traffic = gpd.read_file(export_file_iris_traffic_demand)
    print(f"Demand related to traffic loaded")

if not os.path.exists(export_file_iris_all_demand):    
    print(f"Adding demand related to traffic and population")
    gdf_iris_all_demands = merge_iris_pop_and_traffic(gdf_iris_population, gdf_iris_traffic)
    print(f"Demand related to traffic+pop saved at : {export_file_iris_all_demand}")
    gdf_iris_all_demands.to_file(export_file_iris_all_demand, driver="GeoJSON")
else:
    gdf_iris_all_demands = gpd.read_file(export_file_iris_all_demand)
    print(f"Demand related to traffic+pop loaded")

if not os.path.exists(export_file_df_adjacency):
    print(f"Getting distance between IRIS centroids...")
    gdf_city_iris_with_dist, adjacency_matrix = get_adjacency_from_gdf(gdf_iris_all_demands, delay=waiting_time_api, crs_angles=crs_angles, crs_meters=crs_meters)
    
    # Clean columns
    cols = ['nom_iris', 'geometry', 'population', 'density_km2', 'revenues', 'demand_pop_kWh', 'demand_traffic_kWh', 'total_demand_kWh']
    cols += [f'dist_{i}' for i in range(len(gdf_city_iris_with_dist))]

    gdf_city_iris_with_dist = gdf_city_iris_with_dist[cols]
    gdf_city_iris_with_dist.to_csv(export_file_df_adjacency, index=False)
    print(f"Dataframe saved at {export_file_df_adjacency}")
    np.save(export_file_np_adjacency, adjacency_matrix)
    print(f"Adjacency matrix alone saved at {export_file_np_adjacency}")