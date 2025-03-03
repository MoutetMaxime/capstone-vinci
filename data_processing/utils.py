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
import random
import requests
import json
from urllib.parse import urlencode
import pandas as pd
import numpy as np
from tqdm import tqdm
import time

def get_routing(start_point, end_point, delay=0):
    
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
        "crs": "EPSG:4326",
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

def get_distance(start_point, end_point):
    routing = get_routing(start_point, end_point)
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

    global_df.to_pickle("../data/EV/processed/donnees_EV.pkl")

    return global_df