import os
import sys
import numpy as np

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.generate_dataset import generate_tou_tariff
from models.fuzzy_comfort import FuzzyComfortEvaluator

np.random.seed(42)


class Appliance:

    def __init__(self, name, duration, power_kw, pref_start):
        self.name = name
        self.duration = duration
        self.power_kw = power_kw
        self.pref_start = pref_start


class GeneticApplianceScheduler:

    def __init__(
        self,
        baseline_load,
        tou_tariff,
        pop_size=80,
        generations=80,
        mutation_rate=0.15,
    ):
        self.baseline_load = baseline_load  # 24-hour baseline (kW)
        self.tou_tariff = tou_tariff  # 24-hour pricing ($/kWh)
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.fuzzy_evaluator = FuzzyComfortEvaluator()

        self.appliances = [
            Appliance("Washing Machine", duration=2, power_kw=1.5, pref_start=8),
            Appliance("Dishwasher", duration=2, power_kw=1.2, pref_start=19),
            Appliance("Water Heater", duration=3, power_kw=3.0, pref_start=6),
            Appliance("EV Charger", duration=4, power_kw=7.0, pref_start=18),
        ]
        self.num_apps = len(self.appliances)

    def _random_chromosome(self):
        """Chromosome: [start_h1, start_h2, start_h3, start_h4, hvac_setback]"""
        start_hours = [
            np.random.randint(0, 24 - app.duration + 1)
            for app in self.appliances
        ]
        hvac_setback = np.random.uniform(0.0, 2.5)  # °C setback
        return start_hours + [hvac_setback]

    def _build_load_curve(self, chromosome):
        """Constructs total hourly load curve (kW)."""
        load_curve = self.baseline_load.copy()
        hvac_setback = chromosome[-1]

        # HVAC savings during peak hours (16:00 - 21:00) due to setback
        for h in range(16, 21):
            load_curve[h] = max(0.1, load_curve[h] - (hvac_setback * 0.15))

        # Add flexible appliance loads
        for idx, app in enumerate(self.appliances):
            start_h = int(chromosome[idx])
            for h in range(start_h, start_h + app.duration):
                if h < 24:
                    load_curve[h] += app.power_kw
        return load_curve

    def evaluate_chromosome(self, chromosome):
        """Calculates total energy cost ($) and comfort penalty."""
        load_curve = self._build_load_curve(chromosome)
        energy_cost = np.sum(load_curve * self.tou_tariff)

        # Comfort Penalty Computation
        hvac_setback = chromosome[-1]
        total_penalty = 0.0

        for idx, app in enumerate(self.appliances):
            start_h = int(chromosome[idx])
            delay = abs(start_h - app.pref_start)
            current_price = self.tou_tariff[start_h]

            # Use Fuzzy Inference System
            penalty = self.fuzzy_evaluator.evaluate_comfort_penalty(
                temp_error=hvac_setback if idx == 0 else 0.0,
                delay_hours=delay,
                price_usd=current_price,
            )
            total_penalty += penalty

        # Weighted Objective Function
        alpha = 2.5  # Weight of comfort penalty in USD equivalence
        objective_score = energy_cost + (alpha * total_penalty)
        fitness = 1.0 / (1.0 + objective_score)

        return fitness, energy_cost, total_penalty

    def optimize(self):
        """Executes Genetic Algorithm optimization loop."""
        population = [self._random_chromosome() for _ in range(self.pop_size)]

        best_chromosome = None
        best_fitness = -1

        for gen in range(self.generations):
            fitness_scores = []
            for chrom in population:
                fit, _, _ = self.evaluate_chromosome(chrom)
                fitness_scores.append(fit)

                if fit > best_fitness:
                    best_fitness = fit
                    best_chromosome = chrom

            # Selection: Tournament Selection
            new_population = []
            for _ in range(self.pop_size):
                i1, i2 = np.random.choice(self.pop_size, 2, replace=False)
                winner = (
                    population[i1]
                    if fitness_scores[i1] > fitness_scores[i2]
                    else population[i2]
                )
                new_population.append(winner.copy())

            # Crossover (Single-Point)
            for i in range(0, self.pop_size - 1, 2):
                if np.random.rand() < 0.7:
                    cpoint = np.random.randint(1, self.num_apps)
                    (
                        new_population[i][:cpoint],
                        new_population[i + 1][:cpoint],
                    ) = (
                        new_population[i + 1][:cpoint].copy(),
                        new_population[i][:cpoint].copy(),
                    )

            # Mutation
            for i in range(self.pop_size):
                if np.random.rand() < self.mutation_rate:
                    app_idx = np.random.randint(0, self.num_apps)
                    max_start = 24 - self.appliances[app_idx].duration
                    new_population[i][app_idx] = np.random.randint(
                        0, max_start + 1
                    )

                if np.random.rand() < self.mutation_rate:
                    new_population[i][-1] = np.random.uniform(0.0, 2.5)

            population = new_population

        best_fit, best_cost, best_penalty = self.evaluate_chromosome(
            best_chromosome
        )
        return best_chromosome, best_cost, best_penalty


if __name__ == "__main__":
    # Generate baseline 24h sample
    tou_rates = generate_tou_tariff()
    baseline_load = np.array(
        [
            0.6, 0.5, 0.5, 0.4, 0.4, 0.5, 0.8, 1.1,
            1.2, 0.9, 0.8, 0.7, 0.7, 0.8, 0.9, 1.1,
            1.5, 1.8, 2.0, 1.9, 1.6, 1.2, 0.9, 0.7,
        ]
    )

    # Habit-driven unoptimized start hours
    habit_chromosome = [8, 19, 6, 18, 0.0]  # Pref starts, no HVAC setback
    scheduler = GeneticApplianceScheduler(baseline_load, tou_rates)

    _, unopt_cost, unopt_penalty = scheduler.evaluate_chromosome(
        habit_chromosome
    )

    # Run GA Optimizer
    best_chrom, opt_cost, opt_penalty = scheduler.optimize()

    print("--- GA Load Scheduler Results ---")
    print(f"Unoptimized Schedule Energy Cost: ${unopt_cost:.2f}")
    print(
        f"Optimized Schedule Energy Cost:   ${opt_cost:.2f} (Savings: {((unopt_cost - opt_cost) / unopt_cost) * 100:.1f}%)\n"
    )

    print("Appliance Schedule Shift:")
    for idx, app in enumerate(scheduler.appliances):
        habit_h = habit_chromosome[idx]
        opt_h = int(best_chrom[idx])
        print(
            f"  - {app.name:15s}: Habitual Start = {habit_h:02d}:00 --> Optimized Start = {opt_h:02d}:00"
        )

    print(f"  - HVAC Temp Setback: {best_chrom[-1]:.2f} °C during Peak")