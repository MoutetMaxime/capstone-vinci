import pandas as pd
import streamlit as st
import tqdm as stqdm


def preprocess_vehicles_data(df_vehicles):
    # Create an empty list to store the processed dataframes
    processed_dfs = []

    for type_vhcl, df in stqdm(df_vehicles.items()):
        st.write(type_vhcl)
        st.dataframe(df.head(1000))

        # cumulate the columns to

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
                "Professionnel" if x == "PRO" else ("Particulier" if x == "PAR" else x)
            )
        )

        try:
            df_pivot["NB_VP_THERMIQUE"] = (
                df_pivot["Diesel thermique"]
                + df_pivot["Essence thermique"]
                + df_pivot["Diesel hybride non rechargeable"]
                + df_pivot["Essence hybride non rechargeable"]
            )
        except:
            df_pivot["NB_VP_THERMIQUE"] = (
                df_pivot["Diesel thermique"] + df_pivot["Essence thermique"]
            ) + df_pivot["Diesel hybride non rechargeable"]

        df_pivot["NB_VP_RECHARGEABLE"] = (
            df_pivot["Diesel hybride rechargeable"]
            + df_pivot["Essence hybride rechargeable"]
        )

        df_pivot["NB_VP_EL"] = df_pivot["Electrique et hydrogène"]

        st.dataframe(df_pivot.tail(1000))
        st.write(df_pivot.shape)

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
                "NB_VP_THERMIQUE",
                "NB_VP_RECHARGEABLE",
                "NB_VP_EL",
            ]
        ]

        # Append the processed dataframe to the list
        processed_dfs.append(df_pivot)

    # Concatenate all processed dataframes
    global_df = pd.concat(processed_dfs, ignore_index=True)
    st.dataframe(global_df.tail(1000))
    st.write(global_df.shape)

    global_df.to_pickle("data/1_Processed/Donnees_EV.pkl")


if not "raw_vehicles_data" in st.session_state["data"]:
    st.session_state["data"]["raw_vehicles_data"] = {
        type_vhcl: pd.read_excel(
            f"data/vehicules/parc_{type_vhcl}_com_2011_2024.xlsx",
            header=3,
        )
        for type_vhcl in ["vp", "vul", "pl", "tcp"]
    }
    st.write(st.session_state["data"]["raw_vehicles_data"].keys())

if st.button("Preprocess Vehicles Data"):
    preprocess_vehicles_data(st.session_state["data"]["raw_vehicles_data"])
