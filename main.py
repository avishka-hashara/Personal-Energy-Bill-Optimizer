import utils.transport
import utils.session
import json
import os
import sys
import numpy as np
import torch

from data.generate_dataset import generate_synthetic_day_data, generate_tou_tariff
from models.forecasting import create_lag_features, train_forecasting_model
from optimizer.ga_scheduler import GeneticApplianceScheduler


def forecast_next_24h_baseline(model, scaler_X, scaler_y, recent_df):
    """Uses the trained NN model to forecast the next 24-hour baseline load."""
    featured_df = create_lag_features(recent_df)
    feature_cols = [
        "hour",
        "day_of_week",
        "temperature_c",
        "occupancy",
        "lag_1h",
        "lag_24h",
        "temp_diff_1h",
    ]

    # Predict for the last 24 available hours as tomorrow's horizon simulation
    last_24h_features = featured_df[feature_cols].iloc[-24:].values
    last_24h_scaled = scaler_X.transform(last_24h_features)

    model.eval()
    with torch.no_grad():
        preds_scaled = model(
            torch.tensor(last_24h_scaled, dtype=torch.float32)
        ).numpy()

    preds = scaler_y.inverse_transform(preds_scaled).flatten()
    return np.clip(preds, 0.2, 5.0)


def generate_smart_home_dispatch(
    best_chrom, appliances, unopt_cost, opt_cost
):
    """Formats optimized schedule into a JSON dispatch payload for Home Assistant/MQTT."""
    dispatch = {
        "status": "SUCCESS",
        "summary": {
            "unoptimized_cost_usd": round(unopt_cost, 2),
            "optimized_cost_usd": round(opt_cost, 2),
            "savings_usd": round(unopt_cost - opt_cost, 2),
            "savings_percent": round(
                ((unopt_cost - opt_cost) / unopt_cost) * 100, 1
            ),
        },
        "hvac_control": {
            "mode": "PEAK_SAVINGS",
            "peak_setback_celsius": round(best_chrom[-1], 2),
            "effective_hours": "16:00 - 21:00",
        },
        "appliance_schedules": [],
    }

    for idx, app in enumerate(appliances):
        start_h = int(best_chrom[idx])
        end_h = (start_h + app.duration) % 24
        dispatch["appliance_schedules"].append(
            {
                "appliance_name": app.name,
                "power_rating_kw": app.power_kw,
                "scheduled_start_time": f"{start_h:02d}:00",
                "scheduled_end_time": f"{end_h:02d}:00",
                "smart_plug_command": "TURN_ON",
            }
        )

    return dispatch


def run_pipeline():
    print("==========================================================")
    print("      SMART HOME PERSONAL ENERGY BILL OPTIMIZER           ")
    print("==========================================================")

    # Step 1: Historical Data & Training
    print("\n[1/4] Ingesting historical usage data & training Neural Net Forecaster...")
    historical_df = generate_synthetic_day_data(days=90)
    model, scaler_X, scaler_y, _ = train_forecasting_model(historical_df)

    # Step 2: 24-Hour Forecast
    print("\n[2/4] Generating 24-hour load predictions for tomorrow...")
    predicted_baseline = forecast_next_24h_baseline(
        model, scaler_X, scaler_y, historical_df
    )
    tou_tariff = generate_tou_tariff()

    # Step 3: Optimization
    print("\n[3/4] Running GA Scheduler with Fuzzy Comfort Constraints...")
    scheduler = GeneticApplianceScheduler(predicted_baseline, tou_tariff)
    best_chrom, opt_cost, opt_penalty = scheduler.optimize()

    # Compute unoptimized reference cost
    habit_chrom = [8, 19, 6, 18, 0.0]
    _, unopt_cost, _ = scheduler.evaluate_chromosome(habit_chrom)

    # Step 4: Dispatch Payload Generation
    print("\n[4/4] Generating Smart Home Automation Payload...")
    dispatch_payload = generate_smart_home_dispatch(
        best_chrom, scheduler.appliances, unopt_cost, opt_cost
    )

    print("\n------------------- OPTIMIZATION SUMMARY -------------------")
    print(f"Unoptimized Estimated Bill: ${unopt_cost:.2f}")
    print(f"Optimized Estimated Bill:   ${opt_cost:.2f}")
    print(
        f"Total Projected Savings:    ${dispatch_payload['summary']['savings_usd']:.2f} ({dispatch_payload['summary']['savings_percent']}%)"
    )

    print("\n---------------- SMART HOME DISPATCH PAYLOAD ---------------")
    print(json.dumps(dispatch_payload, indent=2))


if __name__ == "__main__":
    run_pipeline()