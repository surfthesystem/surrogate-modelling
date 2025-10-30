# Reservoir Simulation + Surrogate Modeling

This repo runs a 2‑phase IMPES reservoir simulator and prepares data to train a spatiotemporal GNN surrogate. It now supports monthly control updates (every 30 days) for producer BHP/choke and injector rates with field caps.

## What’s included
- Reservoir model generator: `src/reservoir_model.py` + `config.yaml`
- Data conversion for simulator: `convert_phase1_data.py` → `data/impes_input/`
- IMPES simulator with monthly reallocation, per‑well BHP/choke and injector caps: `simulator/`
- Scenario runner to execute monthly schedules and collect KPIs: `utils/scenario_runner.py`

## Quick start
```bash
python src/reservoir_model.py
python convert_phase1_data.py

cd simulator
python IMPES_phase1.py --tfinal 365 --dt 1 --realloc_days 30
```

## Controls
- Producers: `config.yaml → wells.producers`
  - control: "bhp" (per‑well BHP) or "choke" (0–1 scaling of BHP term)
  - defaults: `bhp_default`, `choke_default`
  - optional schedules (12×N_prod):
    - `data/impes_input/schedule_producer_bhp.csv`
    - `data/impes_input/schedule_producer_choke.csv`
- Injectors: `config.yaml → wells.injectors`
  - per‑well `rate_max`, field `total_injection_min/max`, `distribution: proportional`
  - optional schedule (12×N_inj): `data/impes_input/schedule_injector_rates.csv`

## Scenario runner
Run multiple 12‑month scenarios and collect KPIs to `results/scenarios/summary.csv`:
```bash
python utils/scenario_runner.py --scenario scenarios/ --tfinal 365 --dt 1 --realloc_days 30
```
Scenario JSON fields (broadcast or full matrices): `producer_bhp`, `producer_choke`, `injector_rates`.

## Notes
- Initialize water saturation slightly above residual: handled in `input_file_phase1.py`.
- Results and plots are written to `results/impes_sim/` for each run.

Last Updated: 2025-10-30
