from shapely.ops import split
import numpy as np
from utils_data import get_contours_city_ign, add_area_gdf, get_data_carroyee_city, get_adjacency_from_gdf
from utils_data import get_euclidian_adjacency_from_df
import os

file_ign = '../Data/BDCARTO/44_Loire_Atlantique/data.gpkg'
file_carroye = '../Data/Filosofi2017_carreaux_200m_met.gpkg'
code_insee=44055
crs_meters = 'EPSG:2154'
crs_angles = 'EPSG:4326'
waiting_time_api=0.1

processed_dir = '../data/processed'
os.makedirs(processed_dir, exist_ok=True)

export_file_df_adjacency=f"{processed_dir}/gdf_city_200_with_dist_{code_insee}.csv"
export_file_np_adjacency=f"{processed_dir}/adjacency_matrix_200_{code_insee}.npy"


if not os.path.exists(export_file_df_adjacency) or not os.path.exists(export_file_np_adjacency):
    print("Processing data...")
    gdf_contours_city, geometry_contours_city = get_contours_city_ign(file_ign, code_insee, crs=crs_angles)
    gdf_carroye_city = get_data_carroyee_city(file_carroye, geometry_contours_city, crs=crs_angles)

    cols = ['Idcar_200m', 'I_est_200', 'Ind', 'geometry', 'Men_coll', 'Men_mais', 'Ind_snv']
    gdf_carroye_city_kept = gdf_carroye_city[cols]
    gdf_carroye_city_kept = gdf_carroye_city_kept.rename(columns={'Idcar_200m':'id', 'I_est_200':'estimated', 'Ind':'population', 'Ind_snv':'niveau_de_vie'}) 
    gdf_carroye_city_kept = add_area_gdf(gdf_carroye_city_kept, crs_meters=crs_meters, crs_angles=crs_angles)

    gdf_carroye_city_kept['density_km2'] = gdf_carroye_city_kept['population'] / gdf_carroye_city_kept['area_km2']
    gdf_carroye_city_kept['ratio_appart'] = gdf_carroye_city_kept['Men_coll'] / (gdf_carroye_city_kept['Men_coll'] + gdf_carroye_city_kept['Men_mais'])
    gdf_carroye_city_kept['niveau_de_vie_moyen'] = gdf_carroye_city_kept['niveau_de_vie'] / gdf_carroye_city_kept['population']

    gdf_carroye_city_kept.drop(columns=['area_km2', 'Men_coll', 'Men_mais', 'niveau_de_vie'], inplace=True)

    gdf_city_200_with_dist, adjacency_matrix = get_euclidian_adjacency_from_df(gdf_carroye_city_kept, crs_angles=crs_angles, crs_meters=crs_meters)

    # Clean columns
    cols = ['id', 'geometry', 'population', 'density_km2', 'niveau_de_vie_moyen']
    cols += [f'dist_{i}' for i in range(len(gdf_city_200_with_dist))]

    gdf_city_with_dist = gdf_city_200_with_dist[cols]
    gdf_city_with_dist.to_csv(export_file_df_adjacency, index=False)
    print(f"Dataframe saved at {export_file_df_adjacency}")

    np.save(export_file_np_adjacency, adjacency_matrix)
    print(f"Adjacency matrix alone saved at {export_file_np_adjacency}")
else:
    print("Data already computed and saved")
    print(export_file_np_adjacency)
    print(export_file_df_adjacency)
