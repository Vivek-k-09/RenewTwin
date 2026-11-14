from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import pulp

BATTERY_CAPACITY_KWH = 1000.0
BATTERY_MIN_SOC = 0.2 * BATTERY_CAPACITY_KWH
BATTERY_MAX_SOC = 0.9 * BATTERY_CAPACITY_KWH
MAX_POWER_KW = 250.0
ROUND_TRIP_EFFICIENCY = 0.95


def _build_forecast_frame(solar_kw, wind_kw, demand_kw, tariffs):
    if not (len(solar_kw) == len(wind_kw) == len(demand_kw) == len(tariffs)):
        raise ValueError("Solar, wind, demand, and tariff arrays must have equal length.")
    return pd.DataFrame(
        {
            "hour": np.arange(len(solar_kw)),
            "solar_kw": np.asarray(solar_kw, dtype=float),
            "wind_kw": np.asarray(wind_kw, dtype=float),
            "demand_kw": np.asarray(demand_kw, dtype=float),
            "tariff_inr_per_kwh": np.asarray(tariffs, dtype=float),
        }
    )


def optimize_24h_dispatch(solar_kw, wind_kw, demand_kw, tariffs, initial_soc_kwh: float = 600.0):
    """Minimize grid cost over a 24-hour horizon while satisfying the battery constraints."""
    forecast = _build_forecast_frame(solar_kw, wind_kw, demand_kw, tariffs)
    horizon = len(forecast)

    if horizon <= 0:
        raise ValueError("Forecast horizon must contain at least one interval.")

    prob = pulp.LpProblem("campus_microgrid_dispatch", pulp.LpMinimize)
    charge = [pulp.LpVariable(f"charge_{t}", lowBound=0, upBound=MAX_POWER_KW, cat="Continuous") for t in range(horizon)]
    discharge = [pulp.LpVariable(f"discharge_{t}", lowBound=0, upBound=MAX_POWER_KW, cat="Continuous") for t in range(horizon)]
    grid_import = [pulp.LpVariable(f"grid_{t}", lowBound=0, cat="Continuous") for t in range(horizon)]
    curtailment = [pulp.LpVariable(f"curtail_{t}", lowBound=0, cat="Continuous") for t in range(horizon)]
    soc = [pulp.LpVariable(f"soc_{t}", lowBound=BATTERY_MIN_SOC, upBound=BATTERY_MAX_SOC, cat="Continuous") for t in range(horizon + 1)]
    charging_mode = [pulp.LpVariable(f"mode_{t}", lowBound=0, upBound=1, cat="Binary") for t in range(horizon)]

    prob += pulp.lpSum(grid_import[t] * forecast.loc[t, "tariff_inr_per_kwh"] for t in range(horizon))

    for t in range(horizon):
        renewable = forecast.loc[t, "solar_kw"] + forecast.loc[t, "wind_kw"]
        demand = forecast.loc[t, "demand_kw"]
        prob += renewable + grid_import[t] + discharge[t] == demand + charge[t] + curtailment[t]
        prob += soc[t + 1] == soc[t] + charge[t] * ROUND_TRIP_EFFICIENCY - discharge[t]
        prob += charge[t] <= MAX_POWER_KW * charging_mode[t]
        prob += discharge[t] <= MAX_POWER_KW * (1 - charging_mode[t])
        prob += charge[t] <= MAX_POWER_KW
        prob += discharge[t] <= MAX_POWER_KW

    prob += soc[0] == initial_soc_kwh
    solver = pulp.PULP_CBC_CMD(msg=False)
    status = prob.solve(solver)

    if pulp.LpStatus[status] != "Optimal":
        fallback_schedule, fallback_summary = rule_based_dispatch(solar_kw, wind_kw, demand_kw, tariffs, initial_soc_kwh)
        fallback_summary["objective_cost_inr"] = fallback_summary["cost_inr"]
        validation = validate_schedule(fallback_schedule)
        if not validation["valid"]:
            raise RuntimeError(f"Optimization failed with status: {pulp.LpStatus[status]} and fallback dispatch is invalid")
        return fallback_schedule, fallback_summary

    schedule = []
    for t in range(horizon):
        renewable = float(forecast.loc[t, "solar_kw"]) + float(forecast.loc[t, "wind_kw"])
        demand = float(forecast.loc[t, "demand_kw"])
        schedule.append(
            {
                "hour": int(t),
                "solar_kw": float(forecast.loc[t, "solar_kw"]),
                "wind_kw": float(forecast.loc[t, "wind_kw"]),
                "demand_kw": float(demand),
                "renewable_kw": float(renewable),
                "grid_import_kw": float(grid_import[t].value()),
                "curtailment_kw": float(curtailment[t].value()),
                "battery_charge_kw": float(charge[t].value()),
                "battery_discharge_kw": float(discharge[t].value()),
                "soc_kwh": float(soc[t + 1].value()),
                "tariff_inr_per_kwh": float(forecast.loc[t, "tariff_inr_per_kwh"]),
                "cost_inr": float(grid_import[t].value() * forecast.loc[t, "tariff_inr_per_kwh"]),
                "power_balance_error": float(renewable + grid_import[t].value() + discharge[t].value() - demand - charge[t].value() - curtailment[t].value()),
            }
        )

    total_cost = float(sum(item["cost_inr"] for item in schedule))
    demand_array = np.asarray(demand_kw, dtype=float)
    solar_array = np.asarray(solar_kw, dtype=float)
    wind_array = np.asarray(wind_kw, dtype=float)
    summary = {
        "objective_cost_inr": total_cost,
        "initial_soc_kwh": float(initial_soc_kwh),
        "final_soc_kwh": schedule[-1]["soc_kwh"],
        "total_demand_kwh": float(demand_array.sum()),
        "total_renewable_kwh": float(solar_array.sum() + wind_array.sum()),
    }
    return schedule, summary


def validate_schedule(schedule, initial_soc_kwh: float = 600.0):
    soc = initial_soc_kwh
    for idx, item in enumerate(schedule):
        charge = item["battery_charge_kw"]
        discharge = item["battery_discharge_kw"]
        renewable = item["renewable_kw"]
        demand = item["demand_kw"]
        grid = item["grid_import_kw"]

        curtailment = item.get("curtailment_kw", 0.0)
        if abs((renewable + grid + discharge) - (demand + charge + curtailment)) > 1e-4:
            raise ValueError(f"Power balance violation at hour {idx}: {renewable + grid + discharge} != {demand + charge + curtailment}")

        soc = soc + charge * ROUND_TRIP_EFFICIENCY - discharge
        if soc < BATTERY_MIN_SOC - 1e-4 or soc > BATTERY_MAX_SOC + 1e-4:
            raise ValueError(f"SoC outside bounds at hour {idx}: {soc} kWh")
        if charge + discharge > MAX_POWER_KW + 1e-4:
            raise ValueError(f"Charge/discharge rate violated at hour {idx}")

    return {"valid": True, "final_soc_kwh": float(soc), "initial_soc_kwh": float(initial_soc_kwh)}


def rule_based_dispatch(solar_kw, wind_kw, demand_kw, tariffs, initial_soc_kwh: float = 600.0):
    solar = np.asarray(solar_kw, dtype=float)
    wind = np.asarray(wind_kw, dtype=float)
    demand = np.asarray(demand_kw, dtype=float)
    tariff = np.asarray(tariffs, dtype=float)

    soc = initial_soc_kwh
    schedule = []
    total_cost = 0.0
    for t in range(len(demand)):
        renewable = solar[t] + wind[t]
        charge = 0.0
        discharge = 0.0
        curtailment = 0.0
        grid = 0.0

        if renewable > demand[t]:
            surplus = renewable - demand[t]
            charge = min(MAX_POWER_KW, surplus, BATTERY_MAX_SOC - soc)
            soc = soc + charge * ROUND_TRIP_EFFICIENCY
            curtailment = max(0.0, surplus - charge)
        elif demand[t] > renewable:
            deficit = demand[t] - renewable
            discharge = min(MAX_POWER_KW, deficit, soc - BATTERY_MIN_SOC)
            soc = soc - discharge
            if deficit > discharge:
                grid = deficit - discharge

        if soc > BATTERY_MAX_SOC:
            soc = BATTERY_MAX_SOC
        if soc < BATTERY_MIN_SOC:
            soc = BATTERY_MIN_SOC

        grid = max(0.0, demand[t] + charge + curtailment - renewable - discharge)
        total_cost += grid * tariff[t]

        schedule.append(
            {
                "hour": t,
                "solar_kw": float(solar[t]),
                "wind_kw": float(wind[t]),
                "demand_kw": float(demand[t]),
                "renewable_kw": float(renewable),
                "grid_import_kw": float(grid),
                "curtailment_kw": float(curtailment),
                "battery_charge_kw": float(charge),
                "battery_discharge_kw": float(discharge),
                "soc_kwh": float(soc),
                "tariff_inr_per_kwh": float(tariff[t]),
                "cost_inr": float(grid * tariff[t]),
            }
        )

    return schedule, {"cost_inr": total_cost, "final_soc_kwh": float(soc), "objective_cost_inr": total_cost}
