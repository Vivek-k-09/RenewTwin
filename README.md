# RenewTwin AI

A campus microgrid energy management and digital twin platform built to forecast renewable generation and optimize battery/grid dispatch.

## Overview
Universities rely heavily on grid power during evening peak hours even when they have rooftop solar or campus wind turbines. RenewTwin AI models a campus microgrid (Solar PV, Wind Turbines, Battery Storage, and Utility Grid) to predict next-day generation and schedule battery charging/discharging to cut electricity costs.

## What It Does
- **24-Hour Forecasting:** Uses LSTM models on hourly weather and generation data to estimate solar and wind output.
- **Smart Battery Dispatch:** Schedules when to store excess renewable energy and when to discharge during expensive Time-of-Use (ToU) peak tariff windows.
- **Interactive Digital Twin:** Lets users adjust campus hardware sizes (Solar kW, Wind kW, Battery kWh), test weather disruptions (cloudy days, load spikes, grid outages), and compare AI-optimized costs against a standard baseline.

## Tech Stack
- **Backend:** Python, FastAPI, PyTorch, PuLP (Optimization), Pandas
- **Frontend:** React (Vite), Tailwind CSS, Recharts, Lucide Icons

## Project Structure
```text
RenewTwin/
├── backend/         # FastAPI server and simulation endpoints
├── data/            # Hourly solar, wind, and campus load datasets
├── digital_twin/    # Scenario and microgrid simulation logic
├── frontend/        # React multi-page dashboard
├── models/          # Forecasting models and training scripts
└── optimization/    # Battery and grid dispatch optimizer
