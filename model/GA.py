import random
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


class ChargingStationGA:
    def __init__(
        self,
        locations: List[Tuple[float, float]],
        demands: List[float],
        prices: List[float],
        costs: List[float],
        max_distance: float,
        station_capacity: float,
        max_total_stations: int = None,
        existing_stations: Dict[int, int] = None,
        existing_station_capacity: float = None,
        max_stations_per_location: int = 5,
        population_size: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        elite_size: int = 10,
        max_generations: int = 100,
        adjacency_matrix: np.ndarray = None,
    ):
        """
        Initialize the genetic algorithm for charging station placement optimization.

        Parameters:
        -----------
        locations : List[Tuple[float, float]]
            List of (x, y) coordinates for potential station locations
        demands : List[float]
            Daily demand in kWh for each location
        prices : List[float]
            Price per kWh at each location
        costs : List[float]
            Daily cost of operating a station at each location
        max_distance : float
            Maximum distance an EV can travel without recharging
        station_capacity : float
            Capacity of each station in kWh per day
        max_total_stations : int, optional
            Maximum total number of stations that can be placed across all locations
        existing_stations : Dict[int, int], optional
            Dictionary mapping location indices to number of existing competitor stations
        existing_station_capacity : float, optional
            Capacity of competitor stations in kWh per day (if different from our stations)
        max_stations_per_location : int, optional
            Maximum number of stations that can be placed at any location
        population_size : int, optional
            Size of the population in the genetic algorithm
        mutation_rate : float, optional
            Probability of mutation for each gene
        crossover_rate : float, optional
            Probability of crossover between two individuals
        elite_size : int, optional
            Number of top individuals to preserve in each generation
        max_generations : int, optional
            Maximum number of generations to run
        """
        self.locations = locations
        self.num_locations = len(locations)
        self.demands = demands
        self.prices = prices
        self.costs = costs
        self.max_distance = max_distance
        self.station_capacity = station_capacity
        self.max_total_stations = max_total_stations
        self.existing_stations = existing_stations or {}
        self.existing_station_capacity = existing_station_capacity or station_capacity
        self.max_stations_per_location = max_stations_per_location
        
        # GA parameters
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.max_generations = max_generations
        
        # Precompute distance matrix
        if adjacency_matrix is not None:
            self.distance_matrix = adjacency_matrix
        else:
            self.distance_matrix = self._compute_distance_matrix()
        
        # Initialize population
        self.population = self._initialize_population()
        self.best_solution = None
        self.best_fitness = float('-inf')
        self.fitness_history = []
    
    def _compute_distance_matrix(self) -> np.ndarray:
        """Calculate the distance matrix between all locations."""
        n = self.num_locations
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = self.locations[i]
                x2, y2 = self.locations[j]
                distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                dist_matrix[i, j] = distance
                dist_matrix[j, i] = distance
        return dist_matrix
    
    def _initialize_population(self) -> List[np.ndarray]:
        """Initialize random population of station configurations."""
        population = []
        for _ in range(self.population_size):
            # Create random chromosome: each gene represents number of stations at a location
            chromosome = np.random.randint(0, self.max_stations_per_location + 1, size=self.num_locations)
            
            # If max_total_stations is set, ensure the chromosome respects this constraint
            if self.max_total_stations is not None:
                self._enforce_max_stations_constraint(chromosome)
                
            population.append(chromosome)
        return population
    
    def _enforce_max_stations_constraint(self, chromosome: np.ndarray) -> None:
        """Enforce the maximum total stations constraint on a chromosome."""
        if self.max_total_stations is None:
            return
            
        # While we have too many stations, randomly reduce them
        total_stations = np.sum(chromosome)
        while total_stations > self.max_total_stations:
            # Find non-zero indices
            nonzero_indices = np.where(chromosome > 0)[0]
            if len(nonzero_indices) == 0:
                break
                
            # Randomly select an index to reduce
            idx = np.random.choice(nonzero_indices)
            chromosome[idx] -= 1
            total_stations -= 1
    
    def calculate_fitness(self, chromosome: np.ndarray) -> Tuple[float, float, float, List[float]]:
        """
        Calculate fitness of a chromosome.
        
        Returns:
        --------
        Tuple[float, float, float, List[float]]:
            (overall_fitness, profit, coverage_score, demand_satisfaction)
        """
        # Check if the solution violates the max_total_stations constraint
        if self.max_total_stations is not None and np.sum(chromosome) > self.max_total_stations:
            return float('-inf'), 0, 0, []
        
        profit, demand_satisfaction = self._calculate_profit(chromosome)
        coverage_score = self._calculate_coverage(chromosome)
        
        # Combined fitness - you can adjust weights to prioritize profit vs coverage
        fitness = profit + coverage_score
        
        return fitness, profit, coverage_score, demand_satisfaction
    
    def _calculate_profit(self, chromosome: np.ndarray) -> Tuple[float, List[float]]:
        """
        Calculate daily profit from the station configuration.
        
        Returns:
        --------
        Tuple[float, List[float]]:
            (total_profit, demand_satisfaction_percentages)
        """
        total_profit = 0
        demand_satisfaction = []
        
        for i in range(self.num_locations):
            num_stations = chromosome[i]
            location_demand = self.demands[i]
            
            if num_stations > 0 or i in self.existing_stations:
                # Calculate total capacity at this location
                our_capacity = num_stations * self.station_capacity
                competitor_capacity = self.existing_stations.get(i, 0) * self.existing_station_capacity
                total_capacity = our_capacity + competitor_capacity
                
                # Calculate what percentage of demand can be satisfied (capped at 100%)
                satisfaction_percentage = min(1.0, total_capacity / location_demand)
                
                # Calculate our market share based on capacity ratio
                if total_capacity > 0:
                    our_market_share = our_capacity / total_capacity
                else:
                    our_market_share = 0
                
                # Calculate the demand we can serve
                our_served_demand = location_demand * satisfaction_percentage * our_market_share
                
                # Calculate revenue and profit
                location_revenue = our_served_demand * self.prices[i]
                location_cost = num_stations * self.costs[i]
                location_profit = location_revenue - location_cost
                
                total_profit += location_profit
                demand_satisfaction.append(satisfaction_percentage * 100)  # Store as percentage
            else:
                demand_satisfaction.append(0)  # No station, no demand satisfied
            

        return total_profit, demand_satisfaction

    def _calculate_coverage(self, chromosome: np.ndarray) -> float:
        """Calculate coverage score based on reachability within max_distance."""
        # Identify locations with at least one station (ours or competitors)
        our_station_locations = np.where(chromosome > 0)[0]
        competitor_locations = list(self.existing_stations.keys())
        all_station_locations = list(set(our_station_locations) | set(competitor_locations))

        if len(all_station_locations) == 0:
            return 0

        # For each location, check if it's covered by any station
        weighted_coverage = 0
        total_demand = sum(self.demands)

        for i in range(self.num_locations):
            # Check if this location is within range of a station
            is_covered = False

            for station_idx in all_station_locations:
                if self.distance_matrix[i, station_idx] <= self.max_distance:
                    is_covered = True
                    break

            if is_covered:
                # Only count our coverage for our market share
                if i in our_station_locations:
                    num_stations = chromosome[i]
                    competitor_stations = self.existing_stations.get(i, 0)
                    total_stations = num_stations + competitor_stations
                    our_market_share = num_stations / total_stations if total_stations > 0 else 0
                    weighted_coverage += self.demands[i] * our_market_share
                elif i not in competitor_locations:  # Area covered by our network but no station here
                    weighted_coverage += self.demands[i]

        # Normalize coverage score
        normalized_coverage = weighted_coverage / total_demand * 100 if total_demand > 0 else 0
        return normalized_coverage

    def select_parents(self, fitness_results: List[Tuple[int, float]]) -> List[np.ndarray]:
        """Select parents for reproduction using tournament selection."""
        parents = []

        # Keep elite individuals
        for i in range(min(self.elite_size, len(fitness_results))):
            parents.append(self.population[fitness_results[i][0]])

        # Tournament selection for the rest
        while len(parents) < self.population_size:
            # Select random individuals for tournament
            tournament_size = min(3, len(fitness_results))
            tournament_indices = random.sample(range(len(fitness_results)), tournament_size)
            tournament_fitness = [(idx, fitness_results[idx][1]) for idx in tournament_indices]
            
            # Select the winner (highest fitness)
            winner_idx = max(tournament_fitness, key=lambda x: x[1])[0]
            parents.append(self.population[winner_idx])
        
        return parents
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Perform crossover between two parents."""
        if random.random() > self.crossover_rate:
            return parent1.copy()
        
        # One-point crossover
        crossover_point = random.randint(1, self.num_locations - 1)
        child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        
        # Ensure the child respects the max_total_stations constraint
        if self.max_total_stations is not None:
            self._enforce_max_stations_constraint(child)
            
        return child
    
    def mutate(self, chromosome: np.ndarray) -> np.ndarray:
        """Mutate the chromosome."""
        for i in range(self.num_locations):
            if random.random() < self.mutation_rate:
                # Randomly increase or decrease the number of stations at this location
                change = random.choice([-1, 1])
                new_value = max(0, min(self.max_stations_per_location, chromosome[i] + change))
                chromosome[i] = new_value
        
        # Ensure the chromosome respects the max_total_stations constraint after mutation
        if self.max_total_stations is not None:
            self._enforce_max_stations_constraint(chromosome)
            
        return chromosome
    
    def create_next_generation(self, fitness_results: List[Tuple[int, float]]) -> List[np.ndarray]:
        """Create the next generation through selection, crossover, and mutation."""
        parents = self.select_parents(fitness_results)
        next_generation = []
        
        # Preserve elite individuals
        for i in range(min(self.elite_size, len(parents))):
            next_generation.append(parents[i])
        
        # Create children through crossover and mutation
        while len(next_generation) < self.population_size:
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            
            next_generation.append(child)
        
        return next_generation
    
    def run(self) -> Tuple[np.ndarray, Dict]:
        """Run the genetic algorithm and return the best solution."""
        print("Starting genetic algorithm optimization...")
        print(f"Station capacity: {self.station_capacity} kWh/day")
        print(f"Maximum total stations: {self.max_total_stations if self.max_total_stations is not None else 'unlimited'}")
        
        best_profit = 0
        best_coverage = 0
        best_demand_satisfaction = []
        
        for generation in range(self.max_generations):
            # Evaluate fitness for current population
            fitness_values = []
            for i, chromosome in enumerate(self.population):
                fitness, profit, coverage, demand_satisfaction = self.calculate_fitness(chromosome)
                fitness_values.append((i, fitness, profit, coverage, demand_satisfaction))
            
            # Filter out infeasible solutions (those with -inf fitness)
            valid_fitness_values = [x for x in fitness_values if x[1] != float('-inf')]
            
            # If all solutions are infeasible, we have a problem with our constraints
            if not valid_fitness_values:
                print("Warning: All solutions in this generation are infeasible!")
                # Create some feasible solutions
                self.population = self._initialize_population()
                continue
            
            # Sort by fitness (descending)
            valid_fitness_values.sort(key=lambda x: x[1], reverse=True)
            
            # Record best fitness
            best_idx, best_gen_fitness, best_gen_profit, best_gen_coverage, best_gen_satisfaction = valid_fitness_values[0]
            self.fitness_history.append(best_gen_fitness)
            
            # Update best overall solution
            if best_gen_fitness > self.best_fitness:
                self.best_fitness = best_gen_fitness
                self.best_solution = self.population[best_idx].copy()
                best_profit = best_gen_profit
                best_coverage = best_gen_coverage
                best_demand_satisfaction = best_gen_satisfaction
            
            # Print progress
            if generation % 10 == 0:
                total_stations = np.sum(self.population[best_idx])
                avg_satisfaction = np.mean([s for s in best_gen_satisfaction if s > 0]) if best_gen_satisfaction else 0
                print(f"Generation {generation}: Fitness = {best_gen_fitness:.2f}, "
                      f"Profit: {best_gen_profit:.2f}, Coverage: {best_gen_coverage:.2f}, "
                      f"Avg Satisfaction: {avg_satisfaction:.1f}%, Stations: {total_stations}")
            
            # Create next generation
            fitness_results = [(x[0], x[1]) for x in valid_fitness_values]
            self.population = self.create_next_generation(fitness_results)
        
        # Final evaluation
        print(f"\nOptimization complete after {self.max_generations} generations")
        print(f"Best solution fitness: {self.best_fitness:.2f}")
        print(f"Best profit: {best_profit:.2f}")
        print(f"Total demand satisfied: {np.sum([d * s for d, s in zip(self.demands, best_demand_satisfaction)]) / sum(self.demands):.2f}%")
        print(f"Best coverage: {best_coverage:.2f}")
        print(f"Total stations: {np.sum(self.best_solution)}")
        
        # Calculate average demand satisfaction for locations with stations
        active_satisfaction = [s for s in best_demand_satisfaction if s > 0]
        avg_satisfaction = np.mean(active_satisfaction) if active_satisfaction else 0
        print(f"Average demand satisfaction at active locations: {avg_satisfaction:.1f}%")
        
        # Find unsatisfied locations (low satisfaction percentage)
        if best_demand_satisfaction:
            unsatisfied_locs = [(i, sat) for i, sat in enumerate(best_demand_satisfaction) 
                               if 0 < sat < 80 and self.best_solution[i] > 0]
            if unsatisfied_locs:
                print("\nLocations with demand satisfaction below 80%:")
                for loc_idx, sat in unsatisfied_locs:
                    print(f"  Location {loc_idx}: {sat:.1f}% satisfied (demand: {self.demands[loc_idx]:.1f} kWh, "
                          f"capacity: {self.best_solution[loc_idx] * self.station_capacity:.1f} kWh)")
        
        # Prepare result statistics
        stations_count = sum(self.best_solution)
        locations_used = np.sum(self.best_solution > 0)
        
        result_stats = {
            "fitness": self.best_fitness,
            "profit": best_profit,
            "coverage": best_coverage,
            "total_stations": stations_count,
            "locations_used": locations_used,
            "solution": self.best_solution,
            "fitness_history": self.fitness_history,
            "demand_satisfaction": np.sum([d * s for d, s in zip(self.demands, best_demand_satisfaction)]) / sum(self.demands)
        }
        
        return self.best_solution, result_stats
    
    def plot_results(self, result_stats: Dict):
        """Plot the optimization results."""
        fig, ax = plt.subplots(3, 1, figsize=(12, 16))
        
        # 1. Plot fitness history
        ax[0].plot(result_stats["fitness_history"])
        ax[0].set_title("Fitness Evolution")
        ax[0].set_xlabel("Generation")
        ax[0].set_ylabel("Fitness")
        ax[0].grid(True)
        
        # 2. Plot station placement
        solution = result_stats["solution"]
        x = [loc[0] for loc in self.locations]
        y = [loc[1] for loc in self.locations]
        
        # Size circles by number of stations
        sizes = [50 * (s + 1) for s in solution]
        
        # Color based on demand satisfaction
        demand_satisfaction = result_stats["demand_satisfaction"]
        colors = []
        for i, sat in enumerate(demand_satisfaction):
            if solution[i] == 0:
                colors.append('lightgray')  # No station
            elif sat < 50:
                colors.append('red')  # Poor satisfaction
            elif sat < 80:
                colors.append('orange')  # Medium satisfaction
            else:
                colors.append('green')  # Good satisfaction
        
        ax[1].scatter(x, y, s=sizes, c=colors, alpha=0.7)
        
        # Add labels with number of stations and satisfaction
        for i, (xi, yi) in enumerate(self.locations):
            if solution[i] > 0:
                sat_txt = f"{int(demand_satisfaction[i])}%"
                ax[1].text(xi, yi, f"{solution[i]}", 
                         fontsize=9, ha='center', va='center', color='white')
                ax[1].text(xi, yi+0.3, sat_txt, 
                         fontsize=8, ha='center', va='center', color='black',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Add competitor stations
        for loc_idx, count in self.existing_stations.items():
            if count > 0:
                xi, yi = self.locations[loc_idx]
                ax[1].scatter(xi, yi, s=50 * count, c='blue', alpha=0.2)
                ax[1].text(xi, yi-0.4, f"{count} comp", fontsize=8, ha='center')
        
        # Add coverage circles
        for i, (xi, yi) in enumerate(self.locations):
            if solution[i] > 0:
                coverage_circle = plt.Circle((xi, yi), self.max_distance, 
                                            color='blue', fill=False, linestyle='--', alpha=0.3)
                ax[1].add_patch(coverage_circle)
        
        ax[1].set_title("Charging Station Placement")
        ax[1].set_xlabel("X Coordinate")
        ax[1].set_ylabel("Y Coordinate")
        ax[1].grid(True)
        
        # Add legend for satisfaction colors
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='>80% Satisfied'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='50-80% Satisfied'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='<50% Satisfied'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', alpha=0.2, markersize=10, label='Competitor'),
        ]
        ax[1].legend(handles=legend_elements, loc='upper right')
        
        # 3. Plot demand satisfaction at each location with stations
        location_indices = [i for i in range(self.num_locations) if solution[i] > 0]
        if location_indices:
            location_labels = [f"Loc {i}" for i in location_indices]
            satisfaction_values = [demand_satisfaction[i] for i in location_indices]
            capacity_values = [solution[i] * self.station_capacity for i in location_indices]
            demand_values = [self.demands[i] for i in location_indices]
            
            x_pos = np.arange(len(location_indices))
            width = 0.35
            
            ax[2].bar(x_pos - width/2, demand_values, width, label='Demand (kWh)')
            ax[2].bar(x_pos + width/2, capacity_values, width, label='Capacity (kWh)')
            
            # Add satisfaction percentages as text
            for i, sat in enumerate(satisfaction_values):
                ax[2].text(x_pos[i], max(demand_values[i], capacity_values[i]) + 10, 
                          f"{sat:.1f}%", ha='center', va='bottom')
            
            ax[2].set_xlabel('Location')
            ax[2].set_ylabel('kWh')
            ax[2].set_title('Demand vs. Capacity at Each Station Location')
            ax[2].set_xticks(x_pos)
            ax[2].set_xticklabels(location_labels)
            ax[2].legend()
            ax[2].grid(True, linestyle='--', alpha=0.7)
        
        # Add some stats as text
        stations_text = (
            f"Total Stations: {result_stats['total_stations']}"
            f"{' (max: ' + str(self.max_total_stations) + ')' if self.max_total_stations is not None else ''}\n"
            f"Station Capacity: {self.station_capacity} kWh/day\n"
            f"Locations Used: {result_stats['locations_used']}\n"
            f"Daily Profit: ${result_stats['profit']:.2f}\n"
            f"Coverage: {result_stats['coverage']:.2f}%\n"
            f"Avg Satisfaction: {np.mean([s for s in demand_satisfaction if s > 0]):.1f}%"
        )
        ax[1].text(0.02, 0.02, stations_text, transform=ax[1].transAxes, 
                 fontsize=10, verticalalignment='bottom', 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        return fig


# Example usage
def run_example():
    # Define example data
    num_locations = 20
    
    # Generate random locations in a 10x10 grid
    locations = [(np.random.uniform(0, 10), np.random.uniform(0, 10)) for _ in range(num_locations)]
    
    # Generate random demand (kWh/day) for each location
    demands = np.random.uniform(100, 800, num_locations)
    
    # Price per kWh at each location
    prices = 0.25 * np.ones(num_locations)
    
    # Daily cost of operating a station at each location
    costs =  60000 / (365 * 15) * np.ones(num_locations)
    
    # Maximum distance an EV can travel (in same units as location coordinates)
    max_distance = 3.0
    
    # Station capacity (kWh/day)
    station_capacity = 200
    
    # Maximum total stations constraint
    max_total_stations = 20
    
    # Existing competitor stations
    existing_stations = {
        1: 2,  # 2 competitor stations at location 3
        8: 1,  # 1 competitor station at location 8
        15: 3  # 3 competitor stations at location 15
    }
    
    # Create and run the genetic algorithm
    ga = ChargingStationGA(
        locations=locations,
        demands=demands,
        prices=prices,
        costs=costs,
        max_distance=max_distance,
        station_capacity=station_capacity,
        max_total_stations=max_total_stations,
        existing_stations=existing_stations,
        existing_station_capacity=180,  # Competitor stations have slightly lower capacity
        max_stations_per_location=5,
        population_size=100,
        mutation_rate=0.1,
        crossover_rate=0.8,
        elite_size=10,
        max_generations=100
    )
    
    solution, stats = ga.run()
    ga.plot_results(stats)

    print(solution)
    print(solution + np.array([existing_stations.get(i, 0) for i in range(num_locations)]))
    
    return solution, stats


if __name__ == "__main__":
    solution, stats = run_example()