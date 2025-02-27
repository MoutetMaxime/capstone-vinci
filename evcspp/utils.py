import time
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests


def create_nodes(file_path):
    nodes = []

    df = pd.read_csv(file_path)

    for i in range(len(df)):
        node = {
            "id": i,
            "latitude": df["latitude"][i],
            "longitude": df["longitude"][i],
            "pop": df["Ind"][i],
            "rev_med": df["Ind_snv"][i],
            "density": df["Ind"][i],
        }
        nodes.append(node)

    return nodes


def get_routing_between_points(starting_point, ending_point):
    # Base URL

    # Base URL
    base_url = "https://data.geopf.fr/navigation/itineraire"

    # Parameters as a dictionary
    params = {
        "resource": "bdtopo-osrm",
        "start": f"{starting_point[1]},{starting_point[0]}",  # New start coordinates
        "end": f"{ending_point[1]},{ending_point[0]}",  # New end coordinates
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
    else:
        print(f"Error: {response.status_code}")

    return data.get("distance")


def construct_adjacency(nodes):
    adjacency_matrix = np.zeros((len(nodes), len(nodes)))
    for i, node_i in enumerate(nodes):
        for j, node_j in enumerate(nodes):
            if i == j:
                continue
            adjacency_matrix[i, j] = get_routing_between_points(
                (node_i["latitude"], node_i["longitude"]),
                (node_j["latitude"], node_j["longitude"]),
            )

    # Symetrize for now
    adjacency_matrix = (adjacency_matrix + adjacency_matrix.T) / 2

    return adjacency_matrix


nodes = [
    {
        "id": 0,
        "name": "Escoublac",
        "latitude": 47.301024,
        "longitude": -2.359042,
        "pop": 3471.348097,
        "density": 416.703,
        "rev_med": 27.110,
    },
    {
        "id": 1,
        "name": "Le Guézy",
        "latitude": 47.286831,
        "longitude": -2.328533,
        "pop": 4374.168414,
        "density": 385.999,
        "rev_med": 27.800,
    },
    {
        "id": 2,
        "name": "Beslon",
        "latitude": 47.292557,
        "longitude": -2.382274,
        "pop": 2308.248737,
        "density": 1013.604,
        "rev_med": 23.160,
    },
    {
        "id": 3,
        "name": "Centre-Benoît",
        "latitude": 47.281696,
        "longitude": -2.401234,
        "pop": 1669.696548,
        "density": 1063.247,
        "rev_med": 31.940,
    },
    {
        "id": 4,
        "name": "La Baule les Pins",
        "latitude": 47.280523,
        "longitude": -2.367825,
        "pop": 2394.211088,
        "density": 1316.247,
        "rev_med": 31.760,
    },
    {
        "id": 5,
        "name": "Gare-Grand Clos",
        "latitude": 47.284849,
        "longitude": -2.403604,
        "pop": 1942.327115,
        "density": 1646.44,
        "rev_med": 26.750,
    },
]
