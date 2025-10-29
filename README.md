# GNN-LSTM Reservoir Surrogate Model

A physics-based reservoir simulation project for training Graph Neural Network (GNN) and Long Short-Term Memory (LSTM) surrogate models for fast reservoir forecasting.

## Project Overview

This project generates high-fidelity reservoir simulation data using a validated IMPES (Implicit Pressure, Explicit Saturation) multiphase flow simulator. The data will be used to train machine learning surrogate models that can predict reservoir behavior orders of magnitude faster than traditional simulators.

### Project Status
- Phase 1 Complete: Heterogeneous reservoir model generation
- IMPES Integration Complete: Validated multiphase simulator working
- Phase 2 In Progress: Training data generation (200-300 scenarios)
- Phase 3 Planned: GNN-LSTM surrogate model training

## Quick Start

### Prerequisites
```bash
pip install numpy scipy matplotlib pandas pyyaml
```

### Phase 1: Generate Reservoir Model
```bash
python src/reservoir_model.py
```

### Convert Data for IMPES
```bash
python convert_phase1_data.py
```

### Run Simulation
```bash
cd "Reservoir-Simulator/proj2/Problem 2"
python IMPES_phase1.py
```

## CRITICAL: Sw Initialization Bug Fix

**DO NOT** initialize water saturation exactly at residual (Sw = Swr). This causes mathematical singularity:

```python
# WRONG - Causes NaN!
IC.Sw = petro.Swr * np.ones((N,1))  # Sw = 0.20 exactly

# CORRECT - Avoids singularity
IC.Sw = (petro.Swr + 0.01) * np.ones((N,1))  # Sw = 0.21
```

**Why?** When Sw = Swr, capillary pressure calculation becomes:
- S = (Sw - Swr) / (1 - Swr - Sor) = 0.0
- Pc = Pe * S^(-1/λ) = infinity
- This propagates NaN throughout simulation

See CRITICAL_FIX.md for detailed analysis.

## Reservoir Model Specs

- Grid: 100x100 cells, 5000m x 5000m domain
- Permeability: 10-500 mD, spatially correlated
- Porosity: 0.15-0.16, correlated with permeability
- Wells: 58 total (40 producers, 18 injectors)
- Fluids: Oil-water two-phase flow
- Simulator: Validated IMPES (tested vs CMG)

## Results (10-day test)

- Runtime: 4.3 minutes (10,000 cells)
- Pressure: 4500 to 4389 psi decline
- Numerically stable: No NaN values
- Physics correct: Realistic pressure evolution

Last Updated: 2025-10-28
