# Quick Start Guide

## Prerequisites

```bash
python --version
pip install -r requirements-simulator.txt
```

## Steps

### 1) Generate reservoir model
```bash
python src/reservoir_model.py
```

### 2) Convert data for IMPES
```bash
python convert_phase1_data.py
```

### 3) Run 12‑month simulation with monthly realloc
```bash
cd simulator
python IMPES_phase1.py --tfinal 365 --dt 1 --realloc_days 30
```

Outputs are written to `results/impes_sim/` (NPZ + plots).

## Optional: Monthly schedules
- Place CSVs (12 rows, one per month) under `data/impes_input/`:
  - `schedule_producer_bhp.csv` or `schedule_producer_choke.csv`
  - `schedule_injector_rates.csv`

## Batch scenarios
```bash
python utils/scenario_runner.py --scenario scenarios/ --tfinal 365 --dt 1 --realloc_days 30
```
KPIs saved under `results/scenarios/` with a `summary.csv`.

Last Updated: 2025-10-30
