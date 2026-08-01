import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.generate_dataset import generate_synthetic_day_data, generate_tou_tariff
from models.forecasting import create_lag_features, train_forecasting_model
from optimizer.ga_scheduler import GeneticApplianceScheduler

# Page Config
st.set_page_config(
    page_title="Personal Energy Bill Optimizer",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Personal Smart Home Energy Bill Optimizer")
st.markdown(
    "Predicts hourly energy demand, evaluates fuzzy comfort trade-offs, and optimizes appliance schedules around TOU electricity tariffs."
)

# Sidebar Configuration
st.sidebar.header("⚙️ Optimization Parameters")
pop_size = st.sidebar.slider("GA Population Size", 40, 150, 80)
generations = st.sidebar.slider("GA Generations", 30, 150, 80)
alpha_weight = st.sidebar.slider("Comfort Penalty Weight (USD Equiv.)", 0.5, 10.0, 2.5)

run_button = st.sidebar.button("🚀 Run Optimizer Simulation")


@st.cache_data
def load_and_train():
    df = generate_synthetic_day_data(days=60)
    model, scaler_X, scaler_y, feature_cols = train_forecasting_model(df)
    return df, model, scaler_X, scaler_y, feature_cols


df_hist, model, scaler_X, scaler_y, feature_cols = load_and_train()


if run_button or "best_chrom" not in st.session_state:
    # 24-hour simulation inputs
    tou_rates = generate_tou_tariff()
    baseline_load = np.array(
        [
            0.6, 0.5, 0.5, 0.4, 0.4, 0.5, 0.8, 1.1,
            1.2, 0.9, 0.8, 0.7, 0.7, 0.8, 0.9, 1.1,
            1.5, 1.8, 2.0, 1.9, 1.6, 1.2, 0.9, 0.7,
        ]
    )

    scheduler = GeneticApplianceScheduler(
        baseline_load, tou_rates, pop_size=pop_size, generations=generations
    )

    # Habitual Schedule (Unoptimized)
    habit_chrom = [8, 19, 6, 18, 0.0]
    _, unopt_cost, unopt_penalty = scheduler.evaluate_chromosome(habit_chrom)

    # Run GA Optimization
    best_chrom, opt_cost, opt_penalty = scheduler.optimize()

    st.session_state.best_chrom = best_chrom
    st.session_state.unopt_cost = unopt_cost
    st.session_state.opt_cost = opt_cost
    st.session_state.unopt_penalty = unopt_penalty
    st.session_state.opt_penalty = opt_penalty
    st.session_state.scheduler = scheduler
    st.session_state.tou_rates = tou_rates
    st.session_state.baseline_load = baseline_load

# Retrieve session state
best_chrom = st.session_state.best_chrom
unopt_cost = st.session_state.unopt_cost
opt_cost = st.session_state.opt_cost
scheduler = st.session_state.scheduler
tou_rates = st.session_state.tou_rates
baseline_load = st.session_state.baseline_load

# --- Metrics Dashboard ---
col1, col2, col3, col4 = st.columns(4)
savings_usd = unopt_cost - opt_cost
savings_pct = (savings_usd / unopt_cost) * 100

col1.metric("Unoptimized Daily Cost", f"${unopt_cost:.2f}")
col2.metric("Optimized Daily Cost", f"${opt_cost:.2f}", delta=f"-${savings_usd:.2f}")
col3.metric("Cost Savings (%)", f"{savings_pct:.1f}%")
col4.metric("HVAC Peak Setback", f"{best_chrom[-1]:.2f} °C")

st.markdown("---")

# --- Charts Section ---
tab1, tab2, tab3 = st.tabs(["📈 Load Curve & Pricing", "📅 Appliance Timeline", "🤖 JSON Dispatch Payload"])

with tab1:
    st.subheader("Hourly Energy Load Profile vs. TOU Tariff")

    opt_load = scheduler._build_load_curve(best_chrom)
    hours = np.arange(24)

    fig, ax1 = plt.subplots(figsize=(10, 4))

    color = "tab:blue"
    ax1.set_xlabel("Hour of Day (00:00 - 23:00)")
    ax1.set_ylabel("Electricity Load (kW)", color=color)
    ax1.plot(hours, baseline_load, label="Baseline Load (kW)", color="gray", linestyle="--", linewidth=1.5)
    ax1.plot(hours, opt_load, label="Optimized Load (kW)", color=color, linewidth=2.5)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # Overlay TOU Rate
    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("TOU Rate ($/kWh)", color=color)
    ax2.step(hours, tou_rates, where="post", color=color, linestyle="-.", linewidth=2, label="TOU Tariff ($)")
    ax2.tick_params(axis="y", labelcolor=color)

    fig.tight_layout()
    st.pyplot(fig)

with tab2:
    st.subheader("Appliance Shift Comparison")

    schedule_data = []
    habit_chrom = [8, 19, 6, 18, 0.0]

    for idx, app in enumerate(scheduler.appliances):
        schedule_data.append({
            "Appliance": app.name,
            "Habitual Start": f"{habit_chrom[idx]:02d}:00",
            "Optimized Start": f"{int(best_chrom[idx]):02d}:00",
            "Duration (h)": app.duration,
            "Power (kW)": app.power_kw
        })

    st.table(pd.DataFrame(schedule_data))

with tab3:
    st.subheader("Smart Home Automation Dispatch (JSON)")
    
    dispatch_payload = {
        "status": "SUCCESS",
        "summary": {
            "unoptimized_cost_usd": round(unopt_cost, 2),
            "optimized_cost_usd": round(opt_cost, 2),
            "savings_usd": round(savings_usd, 2),
            "savings_percent": round(savings_pct, 1),
        },
        "hvac_control": {
            "mode": "PEAK_SAVINGS",
            "peak_setback_celsius": round(best_chrom[-1], 2),
            "effective_hours": "16:00 - 21:00",
        },
        "appliance_schedules": [
            {
                "appliance_name": app.name,
                "power_rating_kw": app.power_kw,
                "scheduled_start_time": f"{int(best_chrom[idx]):02d}:00",
                "smart_plug_command": "TURN_ON",
            }
            for idx, app in enumerate(scheduler.appliances)
        ],
    }

    st.json(dispatch_payload)