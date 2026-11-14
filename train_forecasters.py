from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data.generate_data import DEFAULT_OUTPUT

ROOT = Path(__file__).resolve().parent.parent
SAVE_DIR = ROOT / "models" / "saved_models"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

SEQUENCE_LENGTH = 24


class ForecastLSTM(nn.Module):
    def __init__(self, input_size: int = 5, hidden_size: int = 32, output_size: int = 24):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class ForecastGRU(nn.Module):
    def __init__(self, input_size: int = 5, hidden_size: int = 32, output_size: int = 24):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    return df


def create_sequence_data(df: pd.DataFrame, target_col: str, seq_len: int = SEQUENCE_LENGTH):
    feature_cols = [target_col, "hour_sin", "hour_cos", "day_sin", "day_cos"]
    values = df[feature_cols].to_numpy(dtype=np.float32)
    X, y = [], []
    max_start = len(values) - 2 * seq_len + 1

    for idx in range(max_start):
        history = values[idx : idx + seq_len]
        target = values[idx + seq_len : idx + 2 * seq_len, 0]
        X.append(history)
        y.append(target)

    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float32)
    return X_arr, y_arr


def train_model(model, X_train, y_train, epochs: int = 12, batch_size: int = 32, lr: float = 1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)

        avg_loss = running_loss / len(loader.dataset)
        if (epoch + 1) % 4 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.6f}")

    return model


def evaluate_model(model, X_test, y_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    with torch.no_grad():
        x_test = torch.tensor(X_test).to(device)
        y_pred = model(x_test).cpu().numpy()

    y_true = y_test
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {"rmse": float(rmse), "mae": float(mae), "predictions": y_pred}


def load_training_data():
    df = pd.read_csv(DEFAULT_OUTPUT)
    df = add_time_features(df)
    return df


def train_target_model(target_col: str, model_type: str = "lstm"):
    df = load_training_data()
    X, y = create_sequence_data(df, target_col=target_col)
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    if model_type == "gru":
        model = ForecastGRU(input_size=5, hidden_size=32, output_size=SEQUENCE_LENGTH)
    else:
        model = ForecastLSTM(input_size=5, hidden_size=32, output_size=SEQUENCE_LENGTH)

    model = train_model(model, X_train, y_train, epochs=12, batch_size=32, lr=1e-3)
    metrics = evaluate_model(model, X_test, y_test)

    save_path = SAVE_DIR / f"{target_col.replace('_kw', '')}_{model_type}.pth"
    torch.save(model.state_dict(), save_path)

    return {"model": model, "metrics": metrics, "path": str(save_path)}


def main():
    results = {}
    for target_col, model_type in [("solar_kw", "lstm"), ("wind_kw", "lstm"), ("demand_kw", "gru")]:
        result = train_target_model(target_col=target_col, model_type=model_type)
        results[target_col] = {
            "model_type": model_type,
            "rmse": result["metrics"]["rmse"],
            "mae": result["metrics"]["mae"],
            "path": result["path"],
        }

    summary_path = SAVE_DIR / "forecast_metrics.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
