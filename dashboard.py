import json
import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.generate_dataset import generate_synthetic_day_data, generate_tou_tariff
from models.forecasting import train_forecasting_model
from optimizer.ga_scheduler import GeneticApplianceScheduler, Appliance

# Page Setup
st.set_page_config(
    page_title="Personal Energy Bill Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Glassmorphism & Premium Dark Theme look)
st.markdown(
    """
    <style>
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .metric-label {
        font-size: 1rem;
        color: #94a3b8;
        font-weight: 500;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 4px;
    }
    .metric-delta.positive {
        color: #10b981;
        font-weight: 600;
    }
    .metric-delta.negative {
        color: #ef4444;
        font-weight: 600;
    }
    .metric-delta.neutral {
        color: #3b82f6;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚡ Smart Home Personal Energy Bill Optimizer")
st.markdown(
    "<p style='color: #94a3b8; font-size: 1.1rem;'>Interactive analytics dashboard visualizing hourly load forecasting, fuzzy comfort constraints, and genetic load-shifting optimization.</p>",
    unsafe_allow_html=True
)

# Sidebar Parameters
st.sidebar.header("⚙️ Simulation Controls")

with st.sidebar.expander("🧬 Genetic Algorithm Params", expanded=True):
    pop_size = st.slider("Population Size", 40, 200, 80)
    generations = st.slider("Generations", 30, 200, 80)
    alpha_weight = st.slider("Comfort Penalty Weight ($)", 0.5, 10.0, 2.5)

run_sim = st.sidebar.button("🚀 Re-Run Optimization", use_container_width=True, type="primary")

@st.cache_data
def load_and_forecast():
    df = generate_synthetic_day_data(days=60)
    model, scaler_X, scaler_y, feature_cols = train_forecasting_model(df)
    return df

df_hist = load_and_forecast()

# Simulation Execution
if run_sim or "best_chrom" not in st.session_state:
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
    
    # We can override appliances here if we want, but for now we use default.
    habit_chrom = [8, 19, 6, 18, 0.0]
    _, unopt_cost, unopt_penalty = scheduler.evaluate_chromosome(habit_chrom)
    best_chrom, opt_cost, opt_penalty = scheduler.optimize()

    st.session_state.best_chrom = best_chrom
    st.session_state.unopt_cost = unopt_cost
    st.session_state.opt_cost = opt_cost
    st.session_state.scheduler = scheduler
    st.session_state.tou_rates = tou_rates
    st.session_state.baseline_load = baseline_load

# Retrieve State
best_chrom = st.session_state.best_chrom
unopt_cost = st.session_state.unopt_cost
opt_cost = st.session_state.opt_cost
scheduler = st.session_state.scheduler
tou_rates = st.session_state.tou_rates
baseline_load = st.session_state.baseline_load
opt_load = scheduler._build_load_curve(best_chrom)

# --- Top KPIs ---
savings_usd = unopt_cost - opt_cost
savings_pct = (savings_usd / unopt_cost) * 100

st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

# Custom Metric Cards HTML
def metric_card(title, value, delta=None, delta_type="neutral"):
    delta_html = f"<div class='metric-delta {delta_type}'>{delta}</div>" if delta else ""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{title}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """

col1.markdown(metric_card("Habitual Daily Cost", f"${unopt_cost:.2f}"), unsafe_allow_html=True)
col2.markdown(metric_card("Optimized Daily Cost", f"${opt_cost:.2f}", f"-${savings_usd:.2f} savings", "positive"), unsafe_allow_html=True)
col3.markdown(metric_card("Projected Savings", f"{savings_pct:.1f}%", "Impact", "neutral"), unsafe_allow_html=True)
col4.markdown(metric_card("Peak HVAC Setback", f"{best_chrom[-1]:.2f} °C", "Eco Mode", "positive"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- Interactive Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📈 Interactive Load Curve",
        "📅 Appliance Timeline (Gantt)",
        "💰 Cumulative Cost Growth",
        "🤖 Smart Home Dispatch JSON",
    ]
)

# Tab 1: Interactive Plotly Dual-Axis Load Profile with TOU Zones
with tab1:
    st.subheader("Hourly Household Demand Profile & Tariff Zones")

    hours = list(range(24))
    time_labels = [f"{h:02d}:00" for h in hours]

    fig = go.Figure()

    # TOU Zone Background Rectangles
    fig.add_vrect(x0=0, x1=6, fillcolor="#10b981", opacity=0.1, layer="below", line_width=0, annotation_text="Off-Peak")
    fig.add_vrect(x0=6, x1=16, fillcolor="#f59e0b", opacity=0.1, layer="below", line_width=0, annotation_text="Mid-Peak")
    fig.add_vrect(x0=16, x1=21, fillcolor="#ef4444", opacity=0.15, layer="below", line_width=0, annotation_text="Peak Price")
    fig.add_vrect(x0=21, x1=23, fillcolor="#f59e0b", opacity=0.1, layer="below", line_width=0)

    # Traces
    fig.add_trace(go.Scatter(
        x=time_labels, y=baseline_load, name="Baseline Demand (kW)",
        mode="lines+markers", line=dict(color="#64748b", width=2, dash="dash"),
        hovertemplate="Hour: %{x}<br>Baseline Load: %{y:.2f} kW<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=time_labels, y=opt_load, name="Optimized Load (kW)",
        mode="lines+markers", line=dict(color="#3b82f6", width=4),
        hovertemplate="Hour: %{x}<br>Optimized Load: %{y:.2f} kW<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=time_labels, y=tou_rates, name="TOU Tariff ($/kWh)",
        mode="lines", line=dict(color="#f43f5e", width=2, shape="hv"),
        yaxis="y2", hovertemplate="Hour: %{x}<br>Tariff Rate: $%{y:.2f}/kWh<extra></extra>",
    ))

    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Hour of Day", showgrid=False),
        yaxis=dict(title="Electricity Consumption (kW)", side="left", showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
        yaxis2=dict(title="Electricity Rate ($/kWh)", side="right", overlaying="y", showgrid=False),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)", font=dict(color="white")),
        hovermode="x unified",
        margin=dict(l=40, r=40, t=30, b=40),
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)

# Tab 2: Interactive Appliance Schedule Timeline (Gantt Chart)
with tab2:
    st.subheader("Appliance Schedule Shift (Habitual vs. Optimized)")

    habit_chrom = [8, 19, 6, 18, 0.0]
    gantt_data = []

    for idx, app in enumerate(scheduler.appliances):
        # Habitual Start
        h_start = habit_chrom[idx]
        h_end = (h_start + app.duration) % 24
        gantt_data.append(dict(
            Appliance=app.name, Type="Habitual (Unoptimized)",
            Start=f"2026-01-01 {h_start:02d}:00:00",
            Finish=f"2026-01-01 {h_end if h_end > h_start else 23:02d}:59:00",
            Start_Hour=f"{h_start:02d}:00", Power=f"{app.power_kw} kW",
        ))
        # Optimized Start
        o_start = int(best_chrom[idx])
        o_end = (o_start + app.duration) % 24
        gantt_data.append(dict(
            Appliance=app.name, Type="Optimized (GA)",
            Start=f"2026-01-01 {o_start:02d}:00:00",
            Finish=f"2026-01-01 {o_end if o_end > o_start else 23:02d}:59:00",
            Start_Hour=f"{o_start:02d}:00", Power=f"{app.power_kw} kW",
        ))

    df_gantt = pd.DataFrame(gantt_data)
    fig_gantt = px.timeline(
        df_gantt, x_start="Start", x_end="Finish", y="Appliance", color="Type",
        hover_data=["Start_Hour", "Power"],
        color_discrete_map={"Habitual (Unoptimized)": "#f59e0b", "Optimized (GA)": "#10b981"},
    )
    fig_gantt.update_yaxes(autorange="reversed")
    fig_gantt.update_layout(
        template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        height=400, margin=dict(l=40, r=40, t=40, b=40)
    )
    st.plotly_chart(fig_gantt, use_container_width=True)

# Tab 3: Cumulative Cost Growth Comparison
with tab3:
    st.subheader("24-Hour Cumulative Electricity Spending ($)")
    habit_load = scheduler._build_load_curve(habit_chrom)
    cum_unopt = np.cumsum(habit_load * tou_rates)
    cum_opt = np.cumsum(opt_load * tou_rates)

    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(
        x=time_labels, y=cum_unopt, name="Habitual Cumulative Cost ($)",
        mode="lines+markers", line=dict(color="#f59e0b", width=3, dash="dash"),
        hovertemplate="Hour: %{x}<br>Habitual Cost: $%{y:.2f}<extra></extra>",
    ))
    fig_cum.add_trace(go.Scatter(
        x=time_labels, y=cum_opt, name="Optimized Cumulative Cost ($)",
        mode="lines+markers", line=dict(color="#10b981", width=4),
        hovertemplate="Hour: %{x}<br>Optimized Cost: $%{y:.2f}<extra></extra>",
    ))

    fig_cum.update_layout(
        template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Hour of Day", showgrid=False),
        yaxis=dict(title="Cumulative Cost ($ USD)", showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
        hovermode="x unified", height=450,
    )
    st.plotly_chart(fig_cum, use_container_width=True)

# Tab 4: Live Dispatch Payload JSON
with tab4:
    st.subheader("Smart Home IoT Integration Dispatch")
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