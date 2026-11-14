# RenewTwin AI

RenewTwin AI is a campus microgrid digital twin and AI optimization platform for a university energy system with solar PV, wind generation, a battery energy storage system, campus demand, and utility grid imports.

## Implementation plan

1. Build realistic hourly microgrid data generation for a full year and save it to the data layer.
2. Train lightweight forecasting models for solar generation, wind generation, and campus demand using PyTorch.
3. Implement a 24-hour optimization engine that minimizes utility cost while honoring battery and grid constraints.
4. Add a digital twin simulation layer for scenario analysis and rule-based vs AI dispatch comparison.
5. Expose the logic through a FastAPI backend and a Streamlit dashboard for interactive energy analysis.
6. Run the verification scripts to confirm the models train, the optimizer satisfies constraints, and the app launches locally.

## Repository structure

- `data/` — data generation and time series loading
- `models/` — PyTorch forecasting models and saved weights
- `optimization/` — dispatch optimization engine
- `digital_twin/` — scenario simulation and baseline comparison
- `backend/` — FastAPI API server
- `frontend/` — Streamlit dashboard
- `tests/` — validation scripts and smoke tests

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python data/generate_data.py
python models/train_forecasters.py
python -m uvicorn backend.main:app --reload
streamlit run frontend/app.py
```
