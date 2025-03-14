import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
import json
import folium
import geopandas as gpd
import pyogrio
from shapely.ops import split
from shapely.geometry import MultiLineString, LineString, Point, Polygon, MultiPolygon
import pandas as pd
import numpy as np
from folium.plugins import MarkerCluster
import os
import requests
import json
from utils_data import get_contours_city_ign, add_area_gdf, get_data_carroyee_city 
from utils_data import get_euclidian_adjacency_from_df, get_adjacency_from_gdf, get_vehicles_city, get_df_OD_city, get_contours_city_simplifie, get_centroids_communes, restrict_area, get_routing, get_traffic_demand_per_carreau, compute_routes_OD

code_insee=44055
crs_meters = 'EPSG:2154'
crs_angles = 'EPSG:4326'
waiting_time_api=0.1
restricted_distance_km = 150

avg_dist_hab = 30
conso_kwh_km = 0.2
avg_demand_hab = avg_dist_hab * conso_kwh_km
ratio_ev_france = 0.03 

data_dir = '../data'
dir_ev = f"{data_dir}/vehicules"
file_vp = f"{dir_ev}/parc_vp_com_2011_2024.xlsx"
file_vul = f"{dir_ev}/parc_vul_com_2011_2024.xlsx"
file_pl = f"{dir_ev}/parc_pl_com2011_2024.xlsx"
file_tcp = f"{dir_ev}/parc_tcp_commune_2024.xlsx"
files_vehicules = {'vp': file_vp, 'vul': file_vul, 'pl': file_pl, 'tcp': file_tcp}

processed_dir = f'{data_dir}/processed/{code_insee}'
preprocessed_dir = f'{data_dir}/preprocessed/{code_insee}'
export_dir_200 = f'{processed_dir}/200'
os.makedirs(processed_dir, exist_ok=True)
os.makedirs(preprocessed_dir, exist_ok=True)
os.makedirs(export_dir_200, exist_ok=True)

preprocessed_file_vp = f"{preprocessed_dir}/vp.xlsx"
preprocessed_file_vul = f"{preprocessed_dir}/vul.xlsx"
preprocessed_file_pl = f"{preprocessed_dir}/pl.xlsx"
preprocessed_file_tcp = f"{preprocessed_dir}/tcp.xlsx"
processed_files_vehicules = {'vp': preprocessed_file_vp, 'vul': preprocessed_file_vul, 'pl': preprocessed_file_pl, 'tcp': preprocessed_file_tcp}
export_file_vehicles = f"{preprocessed_dir}/df_vehicles_2024.csv"

file_ign = f'{data_dir}/BDCARTO/44_Loire_Atlantique/data.gpkg'
file_carroye = f'{data_dir}/Filosofi2017_carreaux_200m_met.gpkg'
file_OD = f'{data_dir}/RP2020_MOBPRO_csv/FD_MOBPRO_2020.csv'
file_contours_simplifies = f'{data_dir}/communes-version-simplifiee.geojson'
file_bornes = f'{data_dir}/consolidation-etalab-schema-irve-statique-v-2.3.1-20250129.csv'

export_file_OD_routes = f"{preprocessed_dir}/processed_OD.parquet"
export_file_df_adjacency_gpkg=f"{export_dir_200}/gdf_city_with_dist.gpkg"
export_file_np_adjacency=f"{export_dir_200}/adjacency_matrix.npy"
export_file_cs_200 = f"{export_dir_200}/cs.gpkg"

vinci_names_operateur = ['Easy Charge | FR*ECH', 'Easycharge services']

# PART  0: load data and preprocess

print("Loading data...")
gdf_contours_city, geometry_contours_city = get_contours_city_ign(file_ign, code_insee, crs=crs_angles)
center_lon, center_lat = geometry_contours_city.centroid.coords[0]
gdf_carroye_city = get_data_carroyee_city(file_carroye, geometry_contours_city, crs=crs_angles)

cols = ['Idcar_200m', 'I_est_200', 'Ind', 'geometry', 'Men_coll', 'Men_mais', 'Ind_snv']
gdf_carroye_city = gdf_carroye_city[cols]
gdf_carroye_city = gdf_carroye_city.rename(columns={'Idcar_200m':'id', 'I_est_200':'estimated', 'Ind':'population', 'Ind_snv':'niveau_de_vie'}) 
gdf_carroye_city = add_area_gdf(gdf_carroye_city, crs_meters=crs_meters, crs_angles=crs_angles)

gdf_carroye_city['density_km2'] = gdf_carroye_city['population'] / gdf_carroye_city['area_km2']
gdf_carroye_city['ratio_appart'] = gdf_carroye_city['Men_coll'] / (gdf_carroye_city['Men_coll'] + gdf_carroye_city['Men_mais'])
gdf_carroye_city['niveau_de_vie_moyen'] = gdf_carroye_city['niveau_de_vie'] / gdf_carroye_city['population']

gdf_carroye_city.drop(columns=['area_km2', 'Men_coll', 'Men_mais', 'niveau_de_vie'], inplace=True)

### PART 1 : Get demand related to population ###

print("Computing  population demand...")
pop_city = gdf_carroye_city.population.sum() # population de la ville à partir des carreaux
df_vehicules = get_vehicles_city(code_insee, files_vehicules, processed_files_vehicules, export_file_vehicles)
ratio_ev_pop = ((df_vehicules['NB_EL']+df_vehicules['NB_RECHARGEABLE'])/pop_city).iloc[0]
gdf_carroye_city['demand_pop_kWh'] = gdf_carroye_city['population'] * ratio_ev_pop * gdf_carroye_city['ratio_appart'] * avg_demand_hab
gdf_carroye_city.drop(columns=['ratio_appart'], inplace=True) 


### PART 2 : Get demand related to the OD matrices ###
print("Computing  traffic demand...")
df_traffic_city = get_df_OD_city(file_OD, code_insee)
gdf_contours_city, geometry_contours_city = get_contours_city_simplifie(file_contours_simplifies, code_insee)

# Commute centroid in meters
gdf_contours_city = gdf_contours_city.to_crs(crs_meters)
gdf_contours_city['centroid'] = gdf_contours_city.geometry.centroid

# Put back in angles
gdf_contours_city = gdf_contours_city.to_crs(crs_angles)
gdf_contours_city['centroid'] = gdf_contours_city['centroid'].to_crs(crs_angles)
centroid_city = gdf_contours_city.iloc[0].centroid

# Add centroids features in each row
gdf_od_full = get_centroids_communes(file_contours_simplifies, df_traffic_city, centroid_city)

# Keep only trips within a limited area
gdf_od_filtered, restricted_area = restrict_area(gdf_od_full, restricted_distance_km)

# Get routes and distances using API
if not os.path.exists(export_file_OD_routes):
    gdf_od_filtered = compute_routes_OD(gdf_od_filtered, export_file_OD_routes, waiting_time_api=0.1, crs_angles='EPSG:4326')
else:
    # gdf_od_filtered = gpd.read_file(export_file_OD_routes)
    gdf_od_filtered = gpd.read_parquet(export_file_OD_routes)

gdf_carroye_city = get_traffic_demand_per_carreau(gdf_od_filtered, gdf_carroye_city, ratio_ev_france, conso_kwh_km)

### PART 3 : Add the two types of demand ###

gdf_carroye_city['total_demand_kWh'] = gdf_carroye_city['demand_pop_kWh'] + gdf_carroye_city['demand_traffic_kWh']

### PART 4 : Get the number of current stations ###
    
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
gdf_bornes_iris = gpd.sjoin(gdf_bornes_city, gdf_carroye_city, how="inner", predicate="within")

# Group by IRIS code and Vinci status, counting the number of rows instead of summing 'nbre_pdc'
summary_iris_df = (
    gdf_bornes_iris.groupby(["id", "bornes_vinci"])
    .size()  # Count occurrences instead of summing 'nbre_pdc'
    .unstack(fill_value=0)  # Fill missing values with 0
    .reset_index()
)

# Rename columns to match the required format
summary_iris_df = summary_iris_df.rename(columns={0: "concu_nb_pdc", 1: "vinci_nb_pdc"})

# Merge the counts back with the original 'gdf_iris_raw' dataframe
gdf_carroye_city = pd.merge(gdf_carroye_city, summary_iris_df, how="left", on="id")

# Fill NaN values in the merged DataFrame with 0 (for the charging station counts)
gdf_carroye_city.fillna({'concu_nb_pdc': 0, 'vinci_nb_pdc': 0}, inplace=True)

# Add a column for the total number of charging stations (not charging points)
gdf_carroye_city['nb_pdc'] = gdf_carroye_city['concu_nb_pdc'] + gdf_carroye_city['vinci_nb_pdc']

# Ensure the final DataFrame remains a GeoDataFrame
gdf_carroye_city = gpd.GeoDataFrame(gdf_carroye_city, geometry='geometry', crs=gdf_carroye_city.crs)

gdf_carroye_city.to_file(export_file_cs_200, driver="GPKG")

# gdf_carroye_city = gdf_carroye_city[['id', 'concu_nb_pdc', 'vinci_nb_pdc', 'nb_pdc']]
# gdf_carroye_city = gdf_carroye_city.merge(gdf_carroye_city, on="id", how="left")

### PART 5 : Add the adjacency matrix for the carreaux ###

print("Computing  adjacency matrix...")

# gdf_city_200_with_dist, adjacency_matrix = get_adjacency_from_gdf(gdf_carroye_city_kept, delay=waiting_time_api, crs_angles=crs_angles, crs_meters=crs_meters)
gdf_city_200_with_dist, adjacency_matrix = get_euclidian_adjacency_from_df(gdf_carroye_city, crs_angles=crs_angles, crs_meters=crs_meters)

# Clean columns 			
cols = ['id', 'geometry', 'population', 'density_km2', 'niveau_de_vie_moyen', 'demand_pop_kWh', 'demand_traffic_kWh', 'total_demand_kWh']
cols += ['concu_nb_pdc', 'vinci_nb_pdc', 'nb_pdc']
cols += [f'dist_{i}' for i in range(len(gdf_city_200_with_dist))]

gdf_city_with_dist = gdf_city_200_with_dist[cols]
gdf_city_with_dist.to_file(export_file_df_adjacency_gpkg, driver="GPKG")
print(f"Dataframe saved at {export_file_df_adjacency_gpkg}")

np.save(export_file_np_adjacency, adjacency_matrix)
print(f"Adjacency matrix alone saved at {export_file_np_adjacency}")