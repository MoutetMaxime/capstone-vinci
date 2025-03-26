import pandas as pd
import streamlit as st
import tqdm as stqdm


def preprocess_vehicles_data(df_vehicles):
    processed_dfs = []

    for type_vhcl, df in stqdm.tqdm(df_vehicles.items()):
        st.write(type_vhcl)
        st.dataframe(df.head(1000))

        # Ajouter une colonne indiquant le type de véhicule
        df["type_vhcl"] = type_vhcl

        # Supprimer les colonnes inutiles si elles existent
        df = df.drop(
            columns=[
                col
                for col in ["Crit'Air", "crit_air", "Catégorie de véhicules"]
                if col in df.columns
            ]
        )

        # Transformer les colonnes d'années en une seule colonne 'Year'
        df_melted = df.melt(
            id_vars=[
                "Code commune de résidence",
                "libellé commune de résidence",
                "Carburant",
                "statut",
                "type_vhcl",
            ],
            var_name="Year",
            value_name="Value",
        )

        # Créer des colonnes distinctes pour chaque type de carburant
        df_pivot = df_melted.pivot_table(
            index=[
                "Code commune de résidence",
                "libellé commune de résidence",
                "Year",
                "statut",
                "type_vhcl",
            ],
            columns="Carburant",
            values="Value",
            fill_value=0,
            aggfunc="sum",
        ).reset_index()

        df_pivot["Year"] = df_pivot["Year"].astype(str)
        df_pivot["statut"] = df_pivot["statut"].apply(
            lambda x: (
                "Professionnel" if x == "PRO" else "Particulier" if x == "PAR" else x
            )
        )

        # Calculer les types de véhicules
        df_pivot["NB_VP_THERMIQUE"] = (
            df_pivot.get("Diesel thermique", 0)
            + df_pivot.get("Essence thermique", 0)
            + df_pivot.get("Diesel hybride non rechargeable", 0)
            + df_pivot.get("Essence hybride non rechargeable", 0)
        )

        df_pivot["NB_VP_RECHARGEABLE"] = df_pivot.get(
            "Diesel hybride rechargeable", 0
        ) + df_pivot.get("Essence hybride rechargeable", 0)

        df_pivot["NB_VP_EL"] = df_pivot.get("Electrique et hydrogène", 0)

        st.dataframe(df_pivot.tail(1000))
        st.write(df_pivot.shape)

        # Renommer les colonnes
        df_pivot = df_pivot.rename(
            columns={
                "Code commune de résidence": "Code_Commune",
                "libellé commune de résidence": "Nom_Commune",
                "statut": "Statut",
                "type_vhcl": "Type_Vhcl",
                "Year": "Year",
            }
        )

        # Garder uniquement les colonnes pertinentes
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

        processed_dfs.append(df_pivot)

    global_df = pd.concat(processed_dfs, ignore_index=True)
    st.dataframe(global_df.tail(1000))
    st.write(global_df.shape)

    global_df.to_pickle("data/1_Processed/Donnees_EV.pkl")


# **Correction de l'erreur de session_state**
if "data" not in st.session_state:
    st.session_state["data"] = {}

if "raw_vehicles_data" not in st.session_state["data"]:
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
