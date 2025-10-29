# Project Structure

## Root Directory
```
surrogate modelling/
├── README.md                      # Main project documentation
├── QUICKSTART.md                  # Quick start guide (3 steps)
├── CRITICAL_FIX.md                # Sw singularity bug documentation
├── PROJECT_STRUCTURE.md           # This file
├── config.yaml                    # Project configuration
├── requirements.txt               # Python dependencies
│
├── src/                          # Phase 1: Reservoir generation
│   ├── reservoir_model.py        # Generate permeability, porosity, wells
│   └── __init__.py
│
├── convert_phase1_data.py        # Convert Phase 1 → IMPES format
│
├── Reservoir-Simulator/          # Validated IMPES simulator
│   └── proj2/Problem 2/
│       ├── input_file_phase1.py  # Simulation configuration
│       ├── IMPES_phase1.py       # Main simulation script
│       ├── myarrays.py           # Matrix assembly
│       ├── updatewells.py        # Well management
│       ├── cap_press.py          # Capillary pressure
│       ├── rel_perm.py           # Relative permeability
│       ├── fluid_properties.py   # Fluid PVT
│       └── [other modules]       # Supporting functions
│
├── data/                         # Generated data
│   ├── permeability_field.npy    # 100x100 permeability (mD)
│   ├── porosity_field.npy        # 100x100 porosity
│   ├── well_locations.csv        # Well positions (meters)
│   ├── reservoir_config.json     # Metadata
│   └── impes_input/              # IMPES-formatted inputs
│       ├── permeability.txt      # Tab-separated
│       ├── porosity.txt
│       ├── depth.txt
│       ├── well_locations_ft.csv # Wells in feet
│       ├── selected_wells.csv    # Subset for simulation
│       └── well_configuration.csv
│
├── results/                      # Simulation outputs
│   ├── reservoir_visualization.png
│   ├── permeability_histogram.png
│   └── impes_sim/
│       ├── Phase1_n10000_t10_days.npz
│       ├── Phase1_pressure_evolution.png
│       └── Phase1_saturation_evolution.png
│
├── utils/                        # Diagnostic tools
│   ├── diagnose_wells.py         # Check well positions
│   └── diagnose_permeability.py  # Validate permeability field
│
└── archive/                      # Obsolete code (kept for reference)
    ├── ARCHIVE_README.md         # Why files are archived
    ├── test_simulation.py        # Old single-phase test
    ├── multiphase_simulation.py  # Failed custom IMPES
    └── IMPES_phase1_debug.py     # Debugging tool (successful)
```

## File Categories

### Essential Files (Active)
- `src/reservoir_model.py` - Phase 1 reservoir generation ✅
- `convert_phase1_data.py` - Data format conversion ✅
- `Reservoir-Simulator/proj2/Problem 2/IMPES_phase1.py` - Simulation ✅
- `Reservoir-Simulator/proj2/Problem 2/input_file_phase1.py` - Configuration ✅

### Documentation (Read First)
- `README.md` - Full project overview
- `QUICKSTART.md` - 3-step usage guide
- `CRITICAL_FIX.md` - Critical Sw singularity bug fix
- `PROJECT_STRUCTURE.md` - This file

### Utilities (Optional)
- `utils/diagnose_wells.py` - Debugging tool for well placement
- `utils/diagnose_permeability.py` - Validation for permeability field

### Archive (Reference Only)
- `archive/test_simulation.py` - Superseded by IMPES
- `archive/multiphase_simulation.py` - Failed implementation (mass balance issues)
- `archive/IMPES_phase1_debug.py` - Used to find Sw bug (no longer needed)

## Data Flow

```
Step 1: reservoir_model.py
    ↓
    data/*.npy, data/*.csv
    ↓
Step 2: convert_phase1_data.py
    ↓
    data/impes_input/*.txt
    ↓
Step 3: IMPES_phase1.py
    ↓
    results/impes_sim/*.npz, *.png
```

## Key Discoveries

### Critical Bug Fix
- **Location**: `input_file_phase1.py` line 219
- **Issue**: Sw = Swr causes capillary pressure singularity (NaN)
- **Fix**: Sw = Swr + 0.01 (0.21 instead of 0.20)
- **Documentation**: See `CRITICAL_FIX.md`

### Validated Components
- IMPES simulator tested against CMG commercial software ✅
- Mass balance validated (unlike custom implementation) ✅
- Numerical stability confirmed (no NaN after fix) ✅

## File Sizes (Approximate)

- `permeability_field.npy`: 78 KB (100x100 floats)
- `Phase1_n10000_t10_days.npz`: ~8 MB (10,000 cells × 11 timesteps)
- `reservoir_visualization.png`: ~200 KB
- Total data folder: ~10 MB for one simulation

## Maintenance

### When to Clean Up
- After generating 200-300 scenarios, archive old test runs
- Keep final training dataset, remove intermediate tests
- Maintain archive/ for reference

### What NOT to Delete
- `src/reservoir_model.py` - Core Phase 1 code
- `convert_phase1_data.py` - Required for all simulations
- `CRITICAL_FIX.md` - Critical bug documentation
- Validated IMPES files in `Reservoir-Simulator/`

Last Updated: 2025-10-28
