# Capstone Project (Vinci): Optimizing the Positioning of Charging Points for Electric Vehicles

## Description

This project addresses the complex problem of locating electric vehicle (EV) charging stations. Focusing on La Baule-Escoublac, we pursued two parallel approaches:

1. **Demand Estimation**: We conducted an in-depth data analysis to estimate the demand for charging stations using two distinct methods:  
   - Based on the number of EV owners.  
   - Based on energy needs derived from population travel patterns.  

2. **Station Placement Optimization**: We tackled the NP-hard problem of optimally placing charging stations using two approaches:  
   - A **greedy algorithm**.  
   - A **genetic algorithm**.  

Our results align well with existing station locations and expected profitability.

## Repository Structure

- **`data_exploration/`**: Notebooks for initial data analysis.
- **`data_visualisation/`**: Notebooks for generating visuals for reports and presentations.
- **`demand_estimation/`**: Scripts and notebooks for demand estimation.
- **`model/`**: Implementation of optimization algorithms and execution scripts.
- **`data/`**: Raw, preprocessed, and processed datasets.

## Installation

1. Clone the repository:  
   ```bash
   git clone https://github.com/MoutetMaxime/capstone-vinci.git
   
2. Install dependencies:
   ```bash
   pip install -r requirements.txt 

3. Download important data 
- IGN BDCARTO (Department 44 in Geopackage format): https://geoservices.ign.fr/bdcarto
- Contours IRIS: https://geoservices.ign.fr/contoursiris
- Population data per 1km (Geopackage): https://www.insee.fr/fr/statistiques/7655464?sommaire=7655515 
- Population data per 200m (Geopackage): https://www.insee.fr/fr/statistiques/7655475?sommaire=7655515
- House-Work pattern: https://www.insee.fr/fr/statistiques/7637844?sommaire=7637890

4. Rename files or rename paths within the code

## Usage 

### Demand Estimation  

Run scripts from `demand_estimation/` to compute and save demand estimations:  

- **Scripts**:  
  - `get_demand_per_iris.py`  
  - `get_demand_per_200.py`  
  - `get_demand_per_1km.py`  

- **Visualization**:  
  - `nb_demo_final_df.ipynb` (main results visualization)  
  - Additional notebooks explaining intermediate steps.

### Model Execution  

After estimating demand, define the data source and run optimization algorithms from `model/`:

- **Greedy Algorithm**:  
  - `Greedy_vX`: Solver class and utility functions.  
  - `Run_greedy_vX`: Experiment setup (data, hyperparameters).  
  - `Visualisation_greedy_vX`: Notebooks for result visualization.  


## Details on the Model Directory

### Greedy Algorithm Versions  

- **V1**: Starts with charging points at every possible location and removes them iteratively until demand is no longer met.  
- **V2**: Starts with zero charging points and adds them until demand is satisfied.  
- **V3 (greedy_profitable)**: Starts with zero charging points and adds them until profitability stops increasing.  

## Potential Improvements

### **Existing Charging Stations**
- For 200×200 grids, we currently do not retrieve charging stations that do not fit into any grid cell.

### **Greedy Model & Demand Coverage**
- The calculation of reached demand for a given station configuration could be improved.
- In the script for the 200mx200m, route-based distances between zones should be used instead of Euclidean distances. However, this was not done due to excessive API usage and limited time to do it properly.

### **General Enhancements**

- Allow stations (both new and existing) to have different types.
- Better estimate the capacity of the stations.  
- Model demand more precisely within the city.
