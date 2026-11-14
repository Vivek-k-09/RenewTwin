from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.main import app
from data.generate_data import generate_microgrid_data
from digital_twin.simulator import run_what_if_scenario
from optimization.optimizer import optimize_24h_dispatch, validate_schedule


def test_generate_data_creates_csv():
    df = generate_microgrid_data(days=7, output_path="data/generated/test_microgrid.csv")
    assert {"timestamp", "solar_kw", "wind_kw", "demand_kw"}.issubset(df.columns)
    assert len(df) == 7 * 24


def test_optimization_respects_constraints():
    solar = [80, 150, 220, 260, 240, 180, 100] * 3
    wind = [50, 70, 85, 90, 100, 120, 110] * 3
    demand = [300, 310, 330, 360, 420, 470, 440] * 3
    tariffs = [4.2, 5.4, 7.1, 7.1, 8.5, 9.5, 9.0] * 3
    schedule, summary = optimize_24h_dispatch(solar, wind, demand, tariffs)
    validation = validate_schedule(schedule)
    assert validation["valid"] is True
    assert summary["objective_cost_inr"] >= 0


def test_scenario_simulation_runs():
    df = pd.read_csv("data/generated/test_microgrid.csv")
    result = run_what_if_scenario(df, {"solar_factor": 0.7, "wind_factor": 1.2, "demand_factor": 1.1, "tariff_factor": 1.2})
    assert "cost_savings_inr" in result
    assert result["ai_summary"]["objective_cost_inr"] >= 0


def test_config_simulation_api_returns_summary():
    client = TestClient(app)
    payload = {
        "solar_capacity_kw": 500.0,
        "wind_capacity_kw": 300.0,
        "battery_capacity_kwh": 1000.0,
        "max_charge_rate_kw": 250.0,
        "initial_soc_pct": 60.0,
        "off_peak_rate": 4.2,
        "normal_rate": 6.8,
        "peak_rate": 9.5,
        "peak_hours": [18, 19, 20, 21, 22],
        "solar_irradiance_wm2": 820.0,
        "temperature_c": 31.0,
        "wind_speed_ms": 8.5,
        "cloud_cover_pct": 20.0,
        "demand_surge_kw": 0.0,
        "cloud_cover_drop_pct": 0.0,
        "wind_storm_cutout": False,
        "outage_mode": False,
        "campus_profile": "IIT Delhi Campus",
    }
    response = client.post("/config/simulate", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "summary" in body
    assert "cost_savings_inr" in body
    assert "ai_schedule" in body
