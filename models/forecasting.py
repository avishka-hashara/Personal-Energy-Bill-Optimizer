import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)


# --- PyTorch MLP Architecture ---
class LoadForecasterNN(nn.Module):

    def __init__(self, input_dim):
        super(LoadForecasterNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x)


# --- Data Pipeline & Feature Engineering ---
def create_lag_features(df):
    """Creates temporal lag features for forecasting."""
    data = df.copy()
    data["lag_1h"] = data["baseline_load_kw"].shift(1)
    data["lag_24h"] = data["baseline_load_kw"].shift(24)
    data["temp_diff_1h"] = data["temperature_c"].diff(1)

    # Drop NaNs created by shifts
    data = data.dropna().reset_index(drop=True)
    return data


def train_forecasting_model(df):
    """Trains the Neural Network on historical data."""
    df_featured = create_lag_features(df)

    feature_cols = [
        "hour",
        "day_of_week",
        "temperature_c",
        "occupancy",
        "lag_1h",
        "lag_24h",
        "temp_diff_1h",
    ]
    X = df_featured[feature_cols].values
    y = df_featured["baseline_load_kw"].values.reshape(-1, 1)

    # Train/Test Split (80/20 Chronological)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Standardization
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    # Convert to Tensors
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)

    # Model, Optimizer, Loss
    model = LoadForecasterNN(input_dim=len(feature_cols))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training Loop
    epochs = 150
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        preds_scaled = model(X_test_t).numpy()

    preds = scaler_y.inverse_transform(preds_scaled)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    print(f"--- Forecasting Model Performance ---")
    print(f"MAE:  {mae:.4f} kW")
    print(f"RMSE: {rmse:.4f} kW")

    return model, scaler_X, scaler_y, feature_cols


if __name__ == "__main__":
    # Import dataset generator from Phase 1
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.generate_dataset import generate_synthetic_day_data

    # Generate 90 days of synthetic data
    df = generate_synthetic_day_data(days=90)

    # Train model
    model, scaler_X, scaler_y, feature_cols = train_forecasting_model(df)