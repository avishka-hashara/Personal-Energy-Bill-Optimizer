import numpy as np


# --- Triangular & Trapezoidal Membership Functions ---
def trimf(x, a, b, c):
    """Triangular membership function."""
    return np.maximum(
        0, np.minimum((x - a) / (b - a + 1e-9), (c - x) / (c - b + 1e-9))
    )


def trapmf(x, a, b, c, d):
    """Trapezoidal membership function."""
    return np.maximum(
        0,
        np.minimum(
            np.minimum((x - a) / (b - a + 1e-9), 1), (d - x) / (d - c + 1e-9)
        ),
    )


class FuzzyComfortEvaluator:

    def __init__(self):
        pass

    # --- Fuzzification ---
    def _temp_error_mf(self, temp_err):
        small = trapmf(temp_err, 0, 0, 0.5, 1.2)
        moderate = trimf(temp_err, 0.8, 1.8, 2.8)
        large = trapmf(temp_err, 2.2, 3.2, 5.0, 5.0)
        return {"small": small, "moderate": moderate, "large": large}

    def _delay_mf(self, delay):
        short = trapmf(delay, 0, 0, 1.0, 2.0)
        medium = trimf(delay, 1.5, 2.5, 4.0)
        long_d = trapmf(delay, 3.0, 4.5, 8.0, 8.0)
        return {"short": short, "medium": medium, "long": long_d}

    def _price_mf(self, price):
        low = trapmf(price, 0, 0, 0.12, 0.15)
        med = trimf(price, 0.13, 0.18, 0.25)
        high = trapmf(price, 0.22, 0.35, 1.0, 1.0)
        return {"low": low, "medium": med, "high": high}

    # --- Rule Evaluation & Defuzzification (Centroid Method) ---
    def evaluate_comfort_penalty(
        self, temp_error: float, delay_hours: float, price_usd: float
    ) -> float:
        """Evaluates fuzzy rules and returns a scalar Comfort Penalty in [0, 1]."""
        t = self._temp_error_mf(temp_error)
        d = self._delay_mf(delay_hours)
        p = self._price_mf(price_usd)

        # Rule Base Activation Levels
        # Rule 1: Large temp error always causes High Penalty
        r1_high = t["large"]

        # Rule 2: Long delay during LOW price is unacceptable (High Penalty)
        r2_high = min(d["long"], p["low"])

        # Rule 3: Long delay during HIGH price is acceptable tradeoff (Medium Penalty)
        r3_med = min(d["long"], p["high"])

        # Rule 4: Moderate temp error during HIGH price is tolerated (Low Penalty)
        r4_low = min(t["moderate"], p["high"])

        # Rule 5: Small temp error AND Short delay is ideal (Low Penalty)
        r5_low = min(t["small"], d["short"])

        # Rule 6: Moderate temp error during LOW price causes Medium Penalty
        r6_med = min(t["moderate"], p["low"])

        # Aggregate Consequent Strengths
        w_low = max(r4_low, r5_low)
        w_med = max(r3_med, r6_med)
        w_high = max(r1_high, r2_high)

        # Defuzzification over discrete domain [0, 1]
        x_domain = np.linspace(0, 1, 101)
        # Output MFs: Low ~ 0.1, Medium ~ 0.5, High ~ 0.9
        mf_low = trimf(x_domain, 0.0, 0.1, 0.4)
        mf_med = trimf(x_domain, 0.3, 0.5, 0.7)
        mf_high = trimf(x_domain, 0.6, 0.9, 1.0)

        aggregated_mf = np.maximum(
            np.minimum(w_low, mf_low),
            np.maximum(
                np.minimum(w_med, mf_med), np.minimum(w_high, mf_high)
            ),
        )

        # Centroid Defuzzification
        sum_mf = np.sum(aggregated_mf)
        if sum_mf == 0:
            return 0.1  # Default low penalty

        comfort_penalty = np.sum(x_domain * aggregated_mf) / sum_mf
        return float(np.round(comfort_penalty, 4))


if __name__ == "__main__":
    evaluator = FuzzyComfortEvaluator()

    # Test Scenarios
    scenarios = [
        {"name": "Ideal (No Temp Error, No Delay, Cheap)", "t": 0.0, "d": 0.0, "p": 0.10},
        {"name": "Peak Price Tradeoff (2°C Temp Error, Peak Price)", "t": 2.0, "d": 1.0, "p": 0.38},
        {"name": "Bad Delay (4h Delay during Cheap Energy)", "t": 0.5, "d": 4.0, "p": 0.10},
        {"name": "High Discomfort (3.5°C Temp Error)", "t": 3.5, "d": 0.5, "p": 0.18},
    ]

    print("--- Fuzzy Comfort Evaluator Test Results ---")
    for s in scenarios:
        penalty = evaluator.evaluate_comfort_penalty(s["t"], s["d"], s["p"])
        print(f"Scenario: {s['name']}")
        print(f"  -> Comfort Penalty Score: {penalty:.4f}\n")