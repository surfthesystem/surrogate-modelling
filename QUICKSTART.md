# Quick Start Guide

## Prerequisites

```bash
# Ensure Python 3.14+ is installed
python --version

# Install required packages
pip install numpy scipy matplotlib pandas pyyaml
```

## Running a Simulation (3 Steps)

### Step 1: Generate Reservoir Model (Phase 1)

```bash
python src/reservoir_model.py
```

**What it does:**
- Creates 100x100 heterogeneous permeability field (10-500 mD)
- Generates correlated porosity field
- Places 58 wells (40 producers, 18 injectors)
- Saves to `data/` folder

**Runtime**: ~1 minute  
**Outputs**: 
- `data/permeability_field.npy`
- `data/porosity_field.npy` 
- `data/well_locations.csv`
- `results/reservoir_visualization.png`

### Step 2: Convert Data Format

```bash
python convert_phase1_data.py
```

**What it does:**
- Converts NumPy arrays to tab-separated text
- Converts units: meters → feet
- Selects subset of wells for simulation (10 producers, 5 injectors)
- Creates IMPES-compatible input files

**Runtime**: <10 seconds  
**Outputs**: `data/impes_input/*.txt` and `*.csv`

### Step 3: Run IMPES Simulation

```bash
cd "Reservoir-Simulator/proj2/Problem 2"
python IMPES_phase1.py
```

**What it does:**
- Runs validated multiphase flow simulator
- Solves pressure (implicit) and saturation (explicit) at each timestep
- Generates visualization plots

**Runtime**: ~4-5 minutes for 10 days  
**Outputs**:
- `results/impes_sim/Phase1_n10000_t10_days.npz` (simulation data)
- `results/impes_sim/Phase1_pressure_evolution.png`
- `results/impes_sim/Phase1_saturation_evolution.png`

## Changing Simulation Parameters

### Run Longer Simulation

Edit `Reservoir-Simulator/proj2/Problem 2/input_file_phase1.py`:

```python
numerical.tfinal = 100  # Change from 10 to 100 days
```

### Adjust Well Rates

```python
# For injectors (line 168-169):
well.constraint = [
    ...
    [1000.0], [1000.0], ...  # Injector rates in STB/day
]
```

### Change Initial Pressure

```python
reservoir.Pref = 5000.0  # Change from 4500 psi
```

## Troubleshooting

### Error: "NaN values in simulation"
**Cause**: Initial Sw set exactly at Swr  
**Fix**: Ensure line 219 has: `IC.Sw = (petro.Swr + 0.01)*np.ones(...)`  
**See**: CRITICAL_FIX.md for details

### Error: "File not found"
**Cause**: Haven't run previous steps  
**Fix**: Run Step 1 (reservoir_model.py) → Step 2 (convert) → Step 3 (IMPES)

### Simulation runs but no production
**Cause**: 10 days too short for water breakthrough  
**Fix**: Increase `numerical.tfinal` to 50-200 days

## File Locations

- **Phase 1 Data**: `data/*.npy`, `data/*.csv`
- **IMPES Input**: `data/impes_input/*.txt`
- **Results**: `results/reservoir_visualization.png`, `results/impes_sim/*.png`
- **Simulation Data**: `results/impes_sim/*.npz` (load with `np.load()`)

## Next Steps

1. Validate 10-day test run works
2. Run longer simulation (50-200 days) to see water breakthrough
3. Vary reservoir properties / well patterns for training data
4. Generate 200-300 scenarios for Phase 2 (ML surrogate training)

## Getting Help

- **README.md**: Full project documentation
- **CRITICAL_FIX.md**: Detailed bug fix explanation
- **archive/ARCHIVE_README.md**: Why certain files were removed

Last Updated: 2025-10-28
