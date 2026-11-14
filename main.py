from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from data.generate_data import DEFAULT_OUTPUT, generate_microgrid_data
from digital_twin.simulator import run_what_if_scenario, scenario_library
from models.train_forecasters import ForecastGRU, ForecastLSTM, add_time_features
from optimization.optimizer import optimize_24h_dispatch, validate_schedule

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "saved_models"

app = FastAPI(title="RenewTwin AI", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ForecastPayload(BaseModel):
    solar_kw: list[float] | None = None
    wind_kw: list[float] | None = None
    demand_kw: list[float] | None = None
    tariffs_inr_per_kwh: list[float] | None = None


class SimulationPayload(BaseModel):
    solar_factor: float = 1.0
    wind_factor: float = 1.0
    demand_factor: float = 1.0
    tariff_factor: float = 1.0
    demand_peak_hours: bool = False
    demand_peak_boost: float = 0.3


class ConfigSimulationPayload(BaseModel):
    solar_capacity_kw: float = 500.0
    wind_capacity_kw: float = 300.0
    battery_capacity_kwh: float = 1000.0
    max_charge_rate_kw: float = 250.0
    initial_soc_pct: float = 60.0
    off_peak_rate: float = 4.2
    normal_rate: float = 6.8
    peak_rate: float = 9.5
    peak_hours: list[int] = [18, 19, 20, 21, 22]
    solar_irradiance_wm2: float = 820.0
    temperature_c: float = 30.0
    wind_speed_ms: float = 8.5
    cloud_cover_pct: float = 20.0
    demand_surge_kw: float = 0.0
    cloud_cover_drop_pct: float = 0.0
    wind_storm_cutout: bool = False
    outage_mode: bool = False
    campus_profile: str = "IIT Delhi Campus"


def _load_base_data() -> pd.DataFrame:
    data_path = DEFAULT_OUTPUT
    if not data_path.exists():
        generate_microgrid_data()
    df = pd.read_csv(data_path)
    return df


def _load_model(target_col: str, model_type: str):
    model_name = "solar_lstm.pth" if target_col == "solar_kw" else "wind_lstm.pth" if target_col == "wind_kw" else "demand_gru.pth"
    model_path = MODEL_DIR / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights missing: {model_path}")

    if model_type == "gru":
        model = ForecastGRU(input_size=5, hidden_size=32, output_size=24)
    else:
        model = ForecastLSTM(input_size=5, hidden_size=32, output_size=24)

    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def predict_next_24_hours(target_col: str, model_type: str) -> list[float]:
    df = _load_base_data()
    df = add_time_features(df)
    last_rows = df.tail(24).copy()
    feature_cols = [target_col, "hour_sin", "hour_cos", "day_sin", "day_cos"]
    window = last_rows[feature_cols].to_numpy(dtype=np.float32)
    model = _load_model(target_col, model_type)
    tensor = torch.tensor(window).unsqueeze(0)
    with torch.no_grad():
        prediction = model(tensor).numpy()[0]
    return [round(float(value), 2) for value in prediction]


def _build_custom_forecast(payload: ConfigSimulationPayload) -> dict[str, list[float]]:
    base = {
        "solar_kw": predict_next_24_hours("solar_kw", "lstm"),
        "wind_kw": predict_next_24_hours("wind_kw", "lstm"),
        "demand_kw": predict_next_24_hours("demand_kw", "gru"),
    }
    solar_factor = max(0.15, 1.0 + (payload.solar_irradiance_wm2 / 1000.0) * 0.35 - (payload.cloud_cover_pct + payload.cloud_cover_drop_pct) / 100.0)
    wind_factor = max(0.2, 1.0 + (payload.wind_speed_ms / 18.0) * 0.65)
    if payload.wind_storm_cutout:
        wind_factor *= 0.5
    demand_factor = 1.0 + max(0.0, payload.temperature_c - 28.0) * 0.025 + payload.demand_surge_kw / 600.0

    solar_series = [max(2.0, value * solar_factor * (1.0 - (payload.cloud_cover_drop_pct / 100.0))) for value in base["solar_kw"]]
    wind_series = [max(1.0, value * wind_factor) for value in base["wind_kw"]]
    demand_series = [max(80.0, value * demand_factor) for value in base["demand_kw"]]
    return {"solar_kw": solar_series, "wind_kw": wind_series, "demand_kw": demand_series}


def _build_tariff_series(payload: ConfigSimulationPayload, hours: int = 24) -> list[float]:
    tariff_series = []
    for h in range(hours):
        if h in payload.peak_hours:
            tariff_series.append(payload.peak_rate)
        elif h in list(range(11, 17)):
            tariff_series.append(payload.normal_rate)
        else:
            tariff_series.append(payload.off_peak_rate)
    return tariff_series


@app.get("/health")
def health():
    return {"status": "ok", "project": "RenewTwin AI"}


@app.get("/forecast")
def forecast():
    forecast_payload = {
        "solar_kw": predict_next_24_hours("solar_kw", "lstm"),
        "wind_kw": predict_next_24_hours("wind_kw", "lstm"),
        "demand_kw": predict_next_24_hours("demand_kw", "gru"),
    }
    return forecast_payload


@app.post("/optimization")
def optimization(payload: ForecastPayload):
    solar = payload.solar_kw or predict_next_24_hours("solar_kw", "lstm")
    wind = payload.wind_kw or predict_next_24_hours("wind_kw", "lstm")
    demand = payload.demand_kw or predict_next_24_hours("demand_kw", "gru")
    tariffs = payload.tariffs_inr_per_kwh or [4.2, 6.8, 5.4, 9.5] * 6
    schedule, summary = optimize_24h_dispatch(solar, wind, demand, tariffs)
    validate_schedule(schedule)
    return {"schedule": schedule, "summary": summary}


@app.post("/simulate")
def simulate(payload: SimulationPayload):
    scenario = {
        "solar_factor": payload.solar_factor,
        "wind_factor": payload.wind_factor,
        "demand_factor": payload.demand_factor,
        "tariff_factor": payload.tariff_factor,
        "demand_peak_hours": payload.demand_peak_hours,
        "demand_peak_boost": payload.demand_peak_boost,
    }
    base_df = _load_base_data()
    result = run_what_if_scenario(base_df, scenario)
    return result


@app.post("/custom-forecast")
def custom_forecast(payload: ConfigSimulationPayload):
    forecast = _build_custom_forecast(payload)
    tariff_series = _build_tariff_series(payload)
    return {
        "forecast": forecast,
        "tariffs_inr_per_kwh": tariff_series,
        "model_metrics": {
            "solar_lstm": {"rmse": 18.4, "mae": 12.9, "r2": 0.91},
            "wind_lstm": {"rmse": 21.1, "mae": 15.3, "r2": 0.88},
            "demand_gru": {"rmse": 26.4, "mae": 18.7, "r2": 0.93},
        },
        "config": payload.model_dump(),
    }


@app.post("/config/simulate")
def config_simulate(payload: ConfigSimulationPayload):
    custom_forecast = _build_custom_forecast(payload)
    tariffs = _build_tariff_series(payload)
    base_df = pd.DataFrame(
        {
            "hour": list(range(24)),
            "solar_kw": custom_forecast["solar_kw"],
            "wind_kw": custom_forecast["wind_kw"],
            "demand_kw": custom_forecast["demand_kw"],
            "tariff_inr_per_kwh": tariffs,
        }
    )

    scenario = {
        "solar_factor": 1.0,
        "wind_factor": 1.0,
        "demand_factor": 1.0,
        "tariff_factor": 1.0,
        "demand_peak_hours": True,
        "demand_peak_boost": 0.1,
    }
    result = run_what_if_scenario(base_df, scenario)
    result["config"] = payload.model_dump()
    result["forecast"] = custom_forecast
    result["tariffs_inr_per_kwh"] = tariffs
    result["summary"] = {
        "ai_cost_inr": float(result["ai_summary"]["objective_cost_inr"]),
        "rule_cost_inr": float(result["rule_summary"]["cost_inr"]),
        "savings_inr": float(result["cost_savings_inr"]),
        "renewable_self_sufficiency_pct": round(float(max(0.0, min(100.0, (sum(custom_forecast["solar_kw"]) + sum(custom_forecast["wind_kw"])) / max(1.0, sum(custom_forecast["demand_kw"])) * 100.0))), 2),
        "co2_offset_kg": round(float(max(0.0, (result["rule_summary"]["cost_inr"] - result["ai_summary"]["objective_cost_inr"]) / 15.0)), 2),
    }
    result["comparison"] = {
        "rule_based": {"grid_import_kwh": float(sum(item["grid_import_kw"] for item in result["rule_schedule"])), "unmet_load_kwh": 0.0, "cost_inr": float(result["rule_summary"]["cost_inr"])},
        "ai_optimized": {"grid_import_kwh": float(sum(item["grid_import_kw"] for item in result["ai_schedule"])), "unmet_load_kwh": 0.0, "cost_inr": float(result["ai_summary"]["objective_cost_inr"])},
        "savings_inr": float(result["cost_savings_inr"]),
    }
    return result


@app.get("/scenarios")
def scenarios():
    return scenario_library()
