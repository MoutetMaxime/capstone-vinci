import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
import json
import geopandas as gpd
from shapely.ops import split
from shapely.geometry import MultiLineString, LineString, Point, Polygon, MultiPolygon
import pandas as pd
import numpy as np
import random
import requests
import json
from urllib.parse import urlencode
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from itertools import combinations
import os
from scipy.spatial.distance import euclidean
from itertools import combinations


def get_routing(start_point, end_point, crs='EPSG:4326', delay=0):
    
    # Wait to avoid API error
    time.sleep(delay)
    
    # Base URL
    base_url = "https://data.geopf.fr/navigation/itineraire"

    # Parameters as a dictionary
    params = {
        "resource": "bdtopo-osrm",
        "start": f"{start_point[0]},{start_point[1]}",  # New start coordinates
        "end": f"{end_point[0]},{end_point[1]}",  # New end coordinates
        "profile": "car",
        "optimization": "fastest",
        "constraints": '{"constraintType":"banned","key":"wayType","operator":"=","value":"autoroute"}',
        "getSteps": "true",
        "getBbox": "true",
        "distanceUnit": "kilometer",
        # "timeUnit": "hour",
        "timeUnit": "minute",
        "crs": crs,
    }

    # Encode parameters properly and create full URL
    query_string = urlencode(params)
    full_url = f"{base_url}?{query_string}"

    # print("Generated URL:", full_url)

    # Make the request
    response = requests.get(full_url, headers={"Accept": "application/json"})

    # Check response
    if response.status_code == 200:
        data = response.json()
        # print("Response:", data)
        return data
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

def get_random_point_in_geometry(geometry):
    """Generate one random point inside the given geometry 
       and return a GeoDataFrame with its coordinates."""
    
    min_x, min_y, max_x, max_y = geometry.bounds
    while True:
        random_point = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
        if geometry.contains(random_point):
            # Create a GeoDataFrame with the point
            gdf = gpd.GeoDataFrame(pd.DataFrame({'x': [random_point.x], 'y': [random_point.y]}),
                                   geometry=[random_point], crs="EPSG:4326")
            return gdf
        
def get_isochrone(
    point, resource="bdtopo-valhalla", cost_value=300, cost_type="time",
    profile="car", direction="departure", constraints=None,
    distance_unit="meter", time_unit="second", crs="EPSG:4326"
):
    """
    Sends a request to the GeoPF Isochrone API.

    Parameters:
        point (tuple): (longitude, latitude) of the starting location.
        resource (str): Routing engine to use (default: "bdtopo-valhalla").
        cost_value (int): Value for the cost function (e.g., time in seconds).
        cost_type (str): Cost type ("time" or "distance").
        profile (str): Transport mode ("car", "bike", "foot", etc.).
        direction (str): "departure" (outward) or "arrival" (inward).
        constraints (dict or None): Constraints on the route.
        distance_unit (str): Distance unit ("meter", "kilometer", etc.).
        time_unit (str): Time unit ("second", "minute", etc.).
        crs (str): Coordinate reference system (default is EPSG:4326).

    Returns:
        dict: The JSON response from the API.
    """

    base_url = "https://data.geopf.fr/navigation/isochrone"

    # Format constraints as a JSON string if provided
    constraints_str = json.dumps(constraints) if constraints else None

    # Define query parameters
    params = {
        "point": f"{point[0]},{point[1]}",  # Format as "longitude,latitude"
        "resource": resource,
        "costValue": cost_value,
        "costType": cost_type,
        "profile": profile,
        "direction": direction,
        "constraints": constraints_str,  # JSON-encoded constraints
        "distanceUnit": distance_unit,
        "timeUnit": time_unit,
        "crs": crs
    }

    # Send request
    response = requests.get(base_url, params=params, headers={"accept": "application/json"})

    # Check response status
    if response.status_code == 200:
        return response.json()  # Return parsed JSON response
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None  # Return None if the request fails

def get_distance(start_point, end_point, crs='EPSG:4326', delay=0.1):
    routing = get_routing(start_point, end_point, crs=crs, delay=delay)
    distance = routing.get("distance")
    return distance

def preprocess_vehicles_data(df_vehicles):
    # Create an empty list to store the processed dataframes
    processed_dfs = []

    for type_vhcl, df in tqdm(df_vehicles.items()):

        # Add a new column to indicate the type of vehicle
        df["type_vhcl"] = type_vhcl

        # Assuming 'code commune de résidence' and 'libellé commune de résidence' are the identifiers
        df = df.drop(
            columns=[
                col
                for col in [
                    "Crit'Air",
                    "crit_air",
                    "Catégorie de véhicules",
                ]
                if col in list(df.columns)
            ]
        )

        # Étape 1: Regrouper les colonnes d'années en une seule colonne 'Year'
        df_melted = df.melt(
            id_vars=[
                "Code commune de résidence",
                "libellé commune de résidence",
                "Carburant",
                "statut",
                "type_vhcl",  # Include the new column in the id_vars
            ],
            var_name="Year",
            value_name="Value",
        )

        # Étape 2: Créer des colonnes distinctes pour chaque type de carburant
        df_pivot = df_melted.pivot_table(
            index=[
                "Code commune de résidence",
                "libellé commune de résidence",
                "Year",
                "statut",
                "type_vhcl",  # Include the new column in the index
            ],
            columns="Carburant",
            values="Value",
            fill_value=0,
            aggfunc="sum",
        ).reset_index()

        df_pivot["Year"] = df_pivot["Year"].astype(str)
        
        df_pivot["statut"] = df_pivot["statut"].apply(
            lambda x: (
                "Professionnel"
                if x == "PRO"
                else ("Particulier" if x == "PAR" else x)
            )
        )

        df_pivot["NB_THERMIQUE"] = (
            df_pivot.get("Diesel thermique", 0)
            + df_pivot.get("Essence thermique", 0)
            + df_pivot.get("Diesel hybride non rechargeable", 0)
            + df_pivot.get("Essence hybride non rechargeable", 0)
        )

        df_pivot["NB_RECHARGEABLE"] = (
            df_pivot.get("Diesel hybride rechargeable", 0)
            + df_pivot.get("Essence hybride rechargeable", 0)
            )

        df_pivot["NB_EL"] = df_pivot.get("Electrique et hydrogène", 0)

        df_pivot = df_pivot.rename(
            columns={
                "Code commune de résidence": "Code_Commune",
                "libellé commune de résidence": "Nom_Commune",
                "statut": "Statut",
                "type_vhcl": "Type_Vhcl",
                "Year": "Year",
            }
        )

        df_pivot = df_pivot[
            [
                "Code_Commune",
                "Nom_Commune",
                "Year",
                "Statut",
                "Type_Vhcl",
                "NB_THERMIQUE",
                "NB_RECHARGEABLE",
                "NB_EL",
            ]
        ]

        # Append the processed dataframe to the list
        processed_dfs.append(df_pivot)

    # Concatenate all processed dataframes
    global_df = pd.concat(processed_dfs, ignore_index=True)

    # global_df.to_pickle(file_export)

    return global_df

def get_contours_city_ign(file_ign, code_insee, crs='EPSG:4326'):
    
    # Charger un fichier GeoPackage
    gdf_contours_all_communes = gpd.read_file(file_ign, layer="commune")
    
    gdf_contours_all_communes['code_insee'] = gdf_contours_all_communes['code_insee'].astype(str)
    code_insee = str(code_insee)  # Convertir également l'entrée utilisateur en string
    
    gdf_contours_city = gdf_contours_all_communes[gdf_contours_all_communes['code_insee'] == code_insee].to_crs(crs)
    # gdf_contours_city.drop('date_du_recensement', inplace=True)
    
    gdf_contours_city = gdf_contours_city[['code_insee', 'code_insee_du_departement', 'code_insee_de_la_region', 'population', 'surface_en_ha', 'code_postal', 'nom_officiel', 'geometry']]
    geometry_contours_city = gdf_contours_city.iloc[0].geometry  # Extract the polygon geometry
    
    return gdf_contours_city, geometry_contours_city

def get_contours_city_simplifie(file_contours_simplifie, code_insee, crs='EPSG:4326'):
    
    gdf_communes_contours = gpd.read_file(file_contours_simplifie)

    # robustness
    gdf_communes_contours['code'] = gdf_communes_contours['code'].astype(str)
    code_insee = str(code_insee)

    gdf_city = gdf_communes_contours[gdf_communes_contours['code'] == code_insee]
    city_geometry = gdf_city.iloc[0].geometry
    
    return gdf_city, city_geometry

def get_iris_contours_city(file_iris, code_insee, crs = 'EPSG:4326'):
    gdf_iris_contours = gpd.read_file(file_iris)

    # Convertir 'code_insee' en chaîne de caractères pour assurer la compatibilité
    gdf_iris_contours['code_insee'] = gdf_iris_contours['code_insee'].astype(str)
    code_insee = str(code_insee)  # Convertir également l'entrée utilisateur en string
    
    # Keep dataframe for La Baule
    gdf_city_iris = gdf_iris_contours[gdf_iris_contours['code_insee'] ==  code_insee]
    gdf_city_iris = gdf_city_iris.to_crs(crs)

    gdf_city_iris.drop(columns=['cleabs', 'iris', 'type_iris', 'code_insee', 'nom_commune'], inplace=True)
    gdf_city_iris = gdf_city_iris.reset_index(drop=True)

    return gdf_city_iris

def add_population_iris(file_iris, gdf_iris):
    df_iris_hab = pd.read_excel(file_iris, header=5)  # Load first sheet
    
    # Ensure both columns are the same type (string is safer for joins)
    gdf_iris["code_iris"] = gdf_iris["code_iris"].astype(str)
    df_iris_hab["IRIS"] = df_iris_hab["IRIS"].astype(str)

    # Perform the left join
    # cols_to_keep = ["IRIS", "P21_MEN", "P21_PMEN"]
    cols_to_keep = ["IRIS", "P21_PMEN", "P21_LOG", "P21_APPART"]
    gdf_merged = gdf_iris.merge(df_iris_hab[cols_to_keep], left_on="code_iris", right_on="IRIS", how="left")
    
    gdf_merged['ratio_appart'] = gdf_merged['P21_APPART'] / gdf_merged['P21_LOG']
    gdf_merged.drop(columns=['P21_APPART', 'P21_LOG'], inplace=True)

    # Drop redundant IRIS column if needed
    gdf_merged.drop(columns=["IRIS"], inplace=True)
    gdf_merged = gdf_merged.rename(columns={'P21_PMEN':'population'})

    return gdf_merged

def add_area_gdf(gdf, crs_meters='EPSG:2154', crs_angles='EPSG:4326'):
    gdf.set_crs(crs_meters, inplace=True, allow_override=True)  # Example: Setting to WGS84 (latitude/longitude)
    gdf['area'] = gdf.geometry.area
    gdf.set_crs(crs_angles, inplace=True, allow_override=True)
    return gdf

def merge_iris_pop_and_traffic(df_iris_pop, df_iris_traffic):
    df_iris_traffic["code_iris"] = df_iris_traffic["code_iris"].astype(str)
    df_iris_pop["code_iris"] = df_iris_pop["code_iris"].astype(str)

    # Perform the left join
    gdf_iris_all_demands = df_iris_pop.merge(df_iris_traffic[["code_iris", "demand_traffic_kWh"]], on="code_iris", how="left")
    gdf_iris_all_demands['total_demand_kWh'] = gdf_iris_all_demands['demand_pop_kWh'] + gdf_iris_all_demands['demand_traffic_kWh']
    # gdf_iris_all_demands.head()

    return gdf_iris_all_demands

def get_adjacency_from_gdf(gdf, delay=0.1, crs_angles='EPSG:4326', crs_meters='EPSG:2154'):
    
    gdf = gdf.reset_index(drop=True)

    # Create an empty DataFrame to store distances
    distances_df = pd.DataFrame(np.nan, index=gdf.index, columns=gdf.index)

    # add information for dataset
    gdf['geometry'] = gdf['geometry'].to_crs(crs_meters) 
    gdf['centroid'] = gdf.geometry.centroid
    gdf['geometry'] = gdf['geometry'].to_crs(crs_angles)
    gdf['centroid'] = gdf['centroid'].to_crs(crs_angles)
    gdf['longitude'], gdf['latitude'] = zip(*gdf['centroid'].apply(lambda p: (p.x, p.y)))

    # Compute distances efficiently (only once per pair)
    for i, j in combinations(gdf.index, 2):
        dist = get_distance(gdf.at[i, "centroid"].coords[0], gdf.at[j, "centroid"].coords[0], crs=crs_angles, delay=delay)
        distances_df.at[i, j] = dist
        distances_df.at[j, i] = dist  # Use symmetry

    # Set diagonal to 0 (distance to itself)
    np.fill_diagonal(distances_df.values, 0)

    # Merge the distance DataFrame back into the GeoDataFrame
    gdf = gdf.join(distances_df)
    gdf = gdf.rename(columns={col: f"dist_{col}" for col in gdf.columns if col in gdf.index})

    adjacency_matrix = distances_df.values

    return gdf, adjacency_matrix

def get_vehicles_city(code_insee, files_vehicles, preprocessed_files_vehicles, export_file_name):
    
    code_insee = str(code_insee)

    if os.path.exists(export_file_name):
        print("File already exists")
        df_2024 = pd.read_csv(export_file_name)
        return df_2024

    print("Processing vehicule dataset (it may take around 10min)...")
    file_vp = files_vehicles['vp'] 
    file_vul = files_vehicles['vul'] 
    file_pl = files_vehicles['pl'] 
    file_tcp = files_vehicles['tcp'] 
    preprocessed_file_vp = preprocessed_files_vehicles['vp'] 
    preprocessed_file_vul = preprocessed_files_vehicles['vul'] 
    preprocessed_file_pl = preprocessed_files_vehicles['pl'] 
    preprocessed_file_tcp = preprocessed_files_vehicles['tcp'] 

    print("Processing vp...")
    if os.path.exists(preprocessed_file_vp):
        vp_city = pd.read_excel(preprocessed_file_vp)
    else:
        vp = pd.read_excel(file_vp, header=3,)
        vp = vp.rename(columns={'2 024': 2024})
        vp_city = vp[vp['Code commune de résidence'] == code_insee]
        vp_city.to_excel(preprocessed_file_vp, index=False)

    print("Processing vul...")
    if os.path.exists(preprocessed_file_vul):
        vul_city = pd.read_excel(preprocessed_file_vul)
    else:
        vul = pd.read_excel(file_vul, header=3,)
        vul_city = vul[vul['Code commune de résidence'] == code_insee]
        vul_city.to_excel(preprocessed_file_vul, index=False)

    print("Processing pl...")
    if os.path.exists(preprocessed_file_pl):
        pl_city = pd.read_excel(preprocessed_file_pl)
    else:
        pl = pd.read_excel(file_pl, header=3,)
        pl_city = pl[pl['Code commune de résidence'] == code_insee]
        pl_city.to_excel(preprocessed_file_pl, index=False)

    print("Processing tcp...")
    if os.path.exists(preprocessed_file_tcp):
        tcp_city = pd.read_excel(preprocessed_file_tcp)
    else:
        tcp = pd.read_excel(file_tcp, header=3)
        tcp_city = tcp[tcp['Code commune de résidence'] == code_insee]
        tcp_city.to_excel(preprocessed_file_tcp, index=False)

    print("Merging data...")
    raw_vehicules = {
        'vp': vp_city,
        'vul': vul_city, 
        'pl': pl_city,
        'tcp': tcp_city
    }
    process_vehicles = preprocess_vehicles_data(raw_vehicules)

    df_grouped = process_vehicles.groupby("Year", as_index=False)[
    ["NB_THERMIQUE", "NB_RECHARGEABLE", "NB_EL"]
    ].sum()

    df_2024 = df_grouped[df_grouped['Year'] == '2024']
    df_2024.to_csv(export_file_name, index=False)

    print("Data saved!")
    
    return df_2024

def get_traffic_demand_per_iris(df_od, gdf_city_contours, gdf_city_iris, ratio_ev, conso_kwh_km, crs_meters='EPSG:2154', crs_angles='EPSG:4326'):

    # On passe tout en metrique
    gdf_city_iris = gdf_city_iris.to_crs(crs_meters).reset_index(drop=True)
    city_geometry_metric = gdf_city_contours.to_crs(crs_meters).iloc[0]
    df_od = df_od.to_crs(crs_meters)
    df_od['route'] = df_od['route'].to_crs(crs_meters)

    liste_demand_ev = [0] * len(gdf_city_iris)

    for idx, quartier in gdf_city_iris.iterrows():
        # Initialiser la longueur pour ce quartier
        # longueur_total = 0
        longueur_total_ev = 0
        # print(idx)

        # Pour chaque route, calculer l'intersection avec le quartier
        for _, trip in df_od.iterrows():
            # Intersecter la route avec le quartier
            intersection_quartier = quartier['geometry'].intersection(trip['route'])
            intersection_city = city_geometry_metric['geometry'].intersection(trip['route'])
            # print(intersection_city)

            # Si l'intersection n'est pas vide, ajouter la longueur de l'intersection
            if not intersection_quartier.is_empty:
                # Ajouter la longueur de l'intersection (en km)
                proportion_intercepted_demand = intersection_quartier.length / intersection_city.length
                longueur_total_ev += (trip['distance_km'] * trip['cnt'] * ratio_ev * proportion_intercepted_demand) # déja en km
        
        # liste_demand[idx] = longueur_total
        liste_demand_ev[idx] = longueur_total_ev

    gdf_city_iris['demand_km_ev'] = liste_demand_ev
    gdf_city_iris['demand_traffic_kWh'] = gdf_city_iris['demand_km_ev'] * conso_kwh_km
    gdf_city_iris.drop(columns=['demand_km_ev'], inplace=True)

    # crs en angles
    gdf_city_iris = gdf_city_iris.to_crs(crs_angles)  # Reprojection en CRS projeté (en mètres)
    df_od = df_od.to_crs(crs_angles)
    df_od['route'] = df_od['route'].to_crs(crs_angles)

    return gdf_city_iris

# Généralisation de la fonction pour les IRIS (car ici les carreaux ne couvrent pas toute la ville)
# Au lieu de normaliser par la distance de route qui passe par la ville, on normalise par la distance de route qui passe par les carreaux
def get_traffic_demand_per_carreau(df_od, gdf_city_carreau, ratio_ev, conso_kwh_km, crs_meters='EPSG:2154', crs_angles='EPSG:4326'):

    # On passe tout en metrique
    gdf_city_carreau = gdf_city_carreau.to_crs(crs_meters).reset_index(drop=True)
    df_od = df_od.to_crs(crs_meters)
    df_od['route'] = df_od['route'].to_crs(crs_meters)

    # On initialise à 0 les demandes pour chaque carreau
    gdf_city_carreau['demand_km_ev'] = 0.

    # Pour chaque voyage, on calcule la longueur interceptée par tous les carreaux + la longueur interceptée par carreau
    # On associe la distance (longueur_carreau / longueur_tous_les_carreaux) * distance_trajet comme demande en km pour chaque carreau
    for _, trip in df_od.iterrows():
        longueur_interceptee_carreaux = 0.
        
        # stocke les demandes temporaires (le temps de faire le calcul)
        gdf_city_carreau['demand_km_ev_temp'] = 0.
        
        for idx_carreau, carreau in gdf_city_carreau.iterrows():
            
            # Intersecter la route avec le carreau
            intersection_current_carreau = carreau['geometry'].intersection(trip['route'])
            
            # si le chemin passe par le carreau
            if not intersection_current_carreau.is_empty:
                
                longueur_current_carreau = intersection_current_carreau.length
                longueur_interceptee_carreaux += longueur_current_carreau 
                
                # gdf_city_carreau['demand_km_ev_temp'].iloc[idx_carreau] += (trip['distance_km'] * trip['cnt'] * ratio_ev * longueur_current_carreau) # déja en km
                gdf_city_carreau.loc[idx_carreau, 'demand_km_ev_temp'] += (trip['distance_km'] * trip['cnt'] * ratio_ev * longueur_current_carreau) # déja en km
        
        gdf_city_carreau['demand_km_ev_temp'] /= longueur_interceptee_carreaux
        gdf_city_carreau['demand_km_ev'] += gdf_city_carreau['demand_km_ev_temp']                

    gdf_city_carreau['demand_traffic_kWh'] = gdf_city_carreau['demand_km_ev'] * conso_kwh_km
    gdf_city_carreau.drop(columns=['demand_km_ev', 'demand_km_ev_temp'], inplace=True)

    # crs en angles
    gdf_city_carreau = gdf_city_carreau.to_crs(crs_angles)  # Reprojection en CRS projeté (en mètres)

    return gdf_city_carreau

def restrict_area(gdf_od, resticted_dist_km=150, crs_meters="EPSG:2154", crs_angles="EPSG:4326"):
    
    # Passer en crs metriques pour calculer des longueurs / surfaces 
    gdf_restricted = gdf_od.set_geometry("centroid_hab")
    gdf_restricted['centroid_hab'] = gdf_restricted['centroid_hab'].to_crs(crs_meters)
    gdf_restricted = gdf_restricted.set_geometry("centroid_lt")
    gdf_restricted['centroid_lt'] = gdf_restricted['centroid_lt'].to_crs(crs_meters)
    gdf_restricted = gdf_restricted.to_crs(crs_meters)
    
    # Create a 100 km buffer (100,000 meters)
    buffer_m = resticted_dist_km*1000
    df_restricted_areas = gdf_restricted['centroid_lt'].buffer(buffer_m)
    restricted_area_metric = df_restricted_areas.iloc[0]
    restricted_area_angles = df_restricted_areas.to_crs(crs_angles).iloc[0]
    
    gdf_restricted = gdf_restricted[gdf_restricted["centroid_hab"].within(restricted_area_metric)]

    # remettre en crs angulaire
    gdf_restricted = gdf_restricted.to_crs(crs_angles)
    gdf_restricted['centroid_hab'] = gdf_restricted['centroid_hab'].to_crs(crs_angles)
    gdf_restricted['centroid_lt'] = gdf_restricted['centroid_lt'].to_crs(crs_angles)
    
    return gdf_restricted, restricted_area_angles

def get_centroid_gdf(gdf, crs_meters='EPSG:2154', crs_angles='EPSG:4326'):
    
    gdf = gdf.to_crs(crs_meters)
    gdf['centroid'] = gdf['geometry'].centroid
    gdf = gdf.drop(columns=['geometry']) 
    gdf = gdf.set_geometry('centroid')
    gdf['centroid'] = gdf['centroid'].to_crs(crs_angles)
    
    return gdf

def get_centroids_communes(file_contours, df_traffic_city, centroid_city, crs_angles='EPSG:4326', crs_meters='EPSG:2154'):

    gdf_communes_contours = gpd.read_file(file_contours)
    gdf_communes_contours["code"] = pd.to_numeric(gdf_communes_contours["code"], errors="coerce").astype("Int64")
    gdf_communes_centroids = get_centroid_gdf(gdf_communes_contours, crs_meters=crs_meters, crs_angles=crs_angles)
    
    # ajouter le merge avec la ville sinon cette fonction sert à rien
    df_merged = df_traffic_city.merge(gdf_communes_centroids[['code', 'centroid']], 
              left_on='code_commune', 
              right_on='code', 
              how='left')
    
    # Renommer la colonne pour éviter la confusion
    df_merged.rename(columns={'centroid': 'centroid_hab'}, inplace=True)
    df_merged.drop(columns=['code'], inplace=True)

    df_merged = gpd.GeoDataFrame(df_merged, geometry="centroid_hab", crs=crs_angles)
    df_merged["centroid_lt"] = gpd.GeoSeries([centroid_city] * len(df_merged), crs=crs_angles)

    return df_merged

def get_df_OD_city(file_OD, code_insee):
    
    df_mob_full = pd.read_csv(file_OD, sep=';')
    code_insee = int(code_insee)

    dict_transport = {1: 'Pas de transport', 
                  2: 'Pied', 
                  3: 'Vélo', 
                  4: 'Deux roues motorisées', 
                  5: 'Voiture, camion, fourgonettes', 
                  6: 'Transport en commun'}

    col_mob = ['COMMUNE', 'DCLT', 'TRANS'] 
    df_mob = df_mob_full[col_mob]
    df_mob['transport_name'] = df_mob['TRANS'].apply(lambda x: dict_transport.get(x, 'default_value'))
    df_mob = df_mob.drop(columns = ['TRANS'])
    df_mob = df_mob.rename(columns={'COMMUNE': 'code_commune', 'DCLT': 'code_lieu_travail'})

    df_mob["code_commune"] = pd.to_numeric(df_mob["code_commune"], errors="coerce").astype("Int64")
    df_mob["code_lieu_travail"] = pd.to_numeric(df_mob["code_lieu_travail"], errors="coerce").astype("Int64")

    # keep only people working in the city but leaving outside
    df_city = df_mob[(df_mob['code_lieu_travail'] == code_insee) & (df_mob['code_commune'] != code_insee)]

    # keep only people coming to work in car and get their count
    df_city_voit = df_city[df_city['transport_name'] == 'Voiture, camion, fourgonettes']
    df_city_voit = df_city_voit.drop(columns = ['transport_name'])
    df_city_voit_unique = df_city_voit.groupby(["code_commune", "code_lieu_travail"], as_index=False).size()
    df_city_voit_unique = df_city_voit_unique.rename(columns={'size': 'cnt'})

    return df_city_voit_unique

def compute_routes_OD(gdf_od, export_file, waiting_time_api=0.1, crs_angles='EPSG:4326'):

    # Compute the route and distance for the od matrices of the city
    gdf_od["routing"] = gdf_od.apply(lambda row: get_routing(row["centroid_hab"].coords[0], row["centroid_lt"].coords[0], delay=waiting_time_api), axis=1)
    gdf_od["route"] = gdf_od["routing"].apply(lambda row: LineString(row["geometry"]["coordinates"]))
    gdf_od["route"] = gpd.GeoSeries(gdf_od["route"], crs=crs_angles)
    gdf_od["distance_km"] = gdf_od["routing"].apply(lambda row: row["distance"])
    gdf_od.drop(columns=['routing'], inplace=True)
    gdf_od = gdf_od.set_geometry('route')
    
    # gdf_od.to_file(export_file, driver="GeoJSON")
    gdf_od.to_parquet(export_file)

    return gdf_od

def add_area_gdf(gdf, crs_meters='EPSG:2154', crs_angles='EPSG:4326'):
    gdf = gdf.to_crs(crs_meters)  # Example: Setting to WGS84 (latitude/longitude)
    gdf['area_km2'] = gdf.geometry.area / 10**6
    gdf = gdf.to_crs(crs_angles)
    return gdf

def add_revenus_iris(file_iris_revenus, gdf_iris):
    
    df_iris_revenus = pd.read_csv(file_iris_revenus, sep = ";")  # Load first sheet
    
    # Ensure both columns are the same type (string is safer for joins)
    gdf_iris["code_iris"] = gdf_iris["code_iris"].astype(str)
    df_iris_revenus["IRIS"] = df_iris_revenus["IRIS"].astype(str)

    # Perform the left join
    cols_to_keep = ["IRIS", "DISP_MED20"]
    gdf_merged = gdf_iris.merge(df_iris_revenus[cols_to_keep], left_on="code_iris", right_on="IRIS", how="left")

    # Drop redundant IRIS column if needed
    gdf_merged.drop(columns=["IRIS"], inplace=True)

    gdf_merged['DISP_MED20'] = pd.to_numeric(gdf_merged['DISP_MED20'], errors='coerce')
    
    gdf_merged = gdf_merged.rename(columns={'DISP_MED20': 'revenues'})

    return gdf_merged

def get_data_carroyee_city(file_carreaux, geometry_city, crs='EPSG:4326'):
    gdf_carroye = gpd.read_file(file_carreaux)
    gdf_carroye = gdf_carroye.to_crs(crs)
    gdf_carroye_city = gdf_carroye[gdf_carroye.geometry.intersects(geometry_city)]
    gdf_carroye_city = gdf_carroye_city.reset_index(drop=True)
    return gdf_carroye_city


def get_euclidian_adjacency_from_df(gdf, crs_angles='EPSG:4326', crs_meters='EPSG:2154'):
    
    gdf = gdf.reset_index(drop=True)

    # Create an empty DataFrame to store distances
    distances_df = pd.DataFrame(np.nan, index=gdf.index, columns=gdf.index)

    # add information for dataset
    gdf['geometry'] = gdf['geometry'].to_crs(crs_meters) 
    gdf['centroid'] = gdf.geometry.centroid

    # Compute distances efficiently (only once per pair)
    for i, j in combinations(gdf.index, 2):
        # Get the distance in km
        dist = euclidean(gdf.at[i, "centroid"].coords[0], gdf.at[j, "centroid"].coords[0]) / 1_000
        distances_df.at[i, j] = dist
        distances_df.at[j, i] = dist  # Use symmetry

    # Set diagonal to 0 (distance to itself)
    np.fill_diagonal(distances_df.values, 0)

    # Merge the distance DataFrame back into the GeoDataFrame
    gdf = gdf.join(distances_df)
    gdf = gdf.rename(columns={col: f"dist_{col}" for col in gdf.columns if col in gdf.index})

    gdf['geometry'] = gdf['geometry'].to_crs(crs_angles)
    gdf['centroid'] = gdf['centroid'].to_crs(crs_angles)
    gdf['longitude'], gdf['latitude'] = zip(*gdf['centroid'].apply(lambda p: (p.x, p.y)))

    adjacency_matrix = distances_df.values

    return gdf, adjacency_matrix