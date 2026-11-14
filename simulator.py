from __future__ import annotations

import pandas as pd

from optimization.optimizer import optimize_24h_dispatch, rule_based_dispatch, validate_schedule


def run_what_if_scenario(base_forecast: pd.DataFrame, scenario: dict):
    """Apply scenario multipliers to the forecast and compare AI vs rule-based dispatch."""
    df = base_forecast.copy()
    df["solar_kw"] = df["solar_kw"] * scenario.get("solar_factor", 1.0)
    df["wind_kw"] = df["wind_kw"] * scenario.get("wind_factor", 1.0)
    df["demand_kw"] = df["demand_kw"] * scenario.get("demand_factor", 1.0)

    if scenario.get("demand_peak_hours", False):
        peak_mask = (df["hour"] >= 17) & (df["hour"] <= 22)
        df.loc[peak_mask, "demand_kw"] = df.loc[peak_mask, "demand_kw"] * (1 + scenario.get("demand_peak_boost", 0.3))

    tariff_factor = scenario.get("tariff_factor", 1.0)
    df["tariff_inr_per_kwh"] = df["tariff_inr_per_kwh"] * tariff_factor

    solar = df["solar_kw"].tolist()
    wind = df["wind_kw"].tolist()
    demand = df["demand_kw"].tolist()
    tariffs = df["tariff_inr_per_kwh"].tolist()

    ai_schedule, ai_summary = optimize_24h_dispatch(solar, wind, demand, tariffs)
    validate_schedule(ai_schedule)
    rule_schedule, rule_summary = rule_based_dispatch(solar, wind, demand, tariffs)

    cost_savings = rule_summary["cost_inr"] - ai_summary["objective_cost_inr"]
    result = {
        "scenario": scenario,
        "ai_summary": ai_summary,
        "rule_summary": rule_summary,
        "cost_savings_inr": float(cost_savings),
        "ai_schedule": ai_schedule,
        "rule_schedule": rule_schedule,
        "forecast": df.to_dict(orient="records"),
    }
    return result


def scenario_library():
    return {
        "normal": {"solar_factor": 1.0, "wind_factor": 1.0, "demand_factor": 1.0, "tariff_factor": 1.0},
        "cloudy_day": {"solar_factor": 0.4, "wind_factor": 1.0, "demand_factor": 1.0, "tariff_factor": 1.0},
        "high_wind": {"solar_factor": 1.0, "wind_factor": 1.4, "demand_factor": 1.0, "tariff_factor": 1.0},
        "exam_peak": {"solar_factor": 1.0, "wind_factor": 1.0, "demand_factor": 1.3, "demand_peak_hours": True, "demand_peak_boost": 0.3, "tariff_factor": 1.0},
        "tariff_shift": {"solar_factor": 1.0, "wind_factor": 1.0, "demand_factor": 1.0, "tariff_factor": 1.3},
    }
