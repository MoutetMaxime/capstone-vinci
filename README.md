# Capstone Project (Vinci) : Optimising the positioning of charging points for electric vehicles 

## Description
In this work, we tackled the complex problem of electrical vehicle charging stations
placement. Focusing on a specific city, La Baule-Escoublac, our approach followed two parallel
paths. In the first one, we tried to estimate the demand of charging stations from users. We
conducted a thorough data analysis, allowing us to calculate two distinct types of demand: the
first based on the number of electric vehicle owners, and the second on energy needs estimated
from the population’s travel patterns. In the second path, we implemented different algorithm to
solve the NP-hard optimization problem of placing the stations. We used two approach, a greedy
and a genetic one. Our final results are consistent with already existing stations and profit values.

## Repository Structure

- **data_exploration/**: Contains notebooks for initial data exploration.
- **data_visualisation/**: Contains notebooks dedicated to creating visuals used in reports and presentations.
- **demand_estimation/**: Contains scripts and notebooks for estimating demand and saving the results.
- **model/**: Holds the classes implementing the algorithms and the scripts to run them.
- **data/**: Stores raw, preprocessed, and processed data.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/username/repository.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

- **Demand Estimation**: Execute the scripts in `demand_estimation/` to compute and save demand estimation.
- **Modeling**: Define data source after demand estimation and run algorithms from `model/`.