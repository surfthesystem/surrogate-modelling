# IMPES Two-Phase Reservoir Simulator

This folder contains the IMPES (Implicit Pressure, Explicit Saturation) two-phase oil-water reservoir simulator.

## Main Files

### Entry Points
- **IMPES_phase1.py** - Main simulation script for Phase 1 integration (YOUR CUSTOM CODE)
- **input_file_phase1.py** - Configuration file for Phase 1 simulations (YOUR CUSTOM CODE)
- **IMPES.py** - Core IMPES solver implementation
- **input_file_2D.py** - Original template configuration file

### Core Modules
- **myarrays.py** - Matrix assembly for pressure and saturation equations
- **updatewells.py** - Well source/sink term management
- **prodindex.py** - Peaceman well model productivity index calculations
- **Thalf.py** - Transmissibility calculations with upwinding

### Physical Properties
- **fluid_properties.py** - PVT properties (Bo, Bw, viscosities, compressibilities)
- **cap_press.py** - Capillary pressure with hysteresis (Corey-Brooks model)
- **rel_perm.py** - Relative permeability curves (Corey power law)
- **petrophysics.py** - Rock property definitions

### Utilities
- **postprocess.py** - Post-processing and data extraction
- **init_plot.py** - Initialization plotting utilities
- **petroplots.py** - Petrophysical property plotting
- **mobilityfun.py** - Phase mobility calculations
- **spdiaginv.py** - Sparse diagonal matrix inversion
- **assignment1_reservior_init.py** - Reservoir initialization utilities

## Usage

Run the simulator with Phase 1 generated data:
```bash
python IMPES_phase1.py
```

The simulator will:
1. Read reservoir properties from `../data/impes_input/`
2. Solve pressure and saturation equations using IMPES
3. Output results to `../results/impes_sim/`

## Simulator Features

### Physical Models
- Two-phase oil-water flow
- Capillary pressure with hysteresis
- Gravity segregation
- Black oil PVT model (undersaturated oil)
- Anisotropic heterogeneous permeability

### Well Models
- Peaceman well model for vertical and horizontal wells
- Rate-constrained wells
- BHP-constrained wells
- Phase splitting based on fractional flow

### Numerical Method
- IMPES (Implicit Pressure, Explicit Saturation)
- Potential-based upwinding
- Harmonic averaging for transmissibility
- Fixed time stepping

## Attribution

Based on reservoir simulator code by Mohammad Afzal Shadab (Problem_14new_FullTwoPhaseComplex).
Modified and integrated for surrogate modeling project.

See `../archive/Reservoir-Simulator/readme.md` for original author information.

## Related Documentation

- [Project README](../README.md) - Main project overview
- [QUICKSTART](../QUICKSTART.md) - Getting started guide
- [CRITICAL_FIX](../CRITICAL_FIX.md) - Known issues and fixes
- [Archive README](../archive/ARCHIVE_README.md) - Reference materials

---

Last Updated: 2025-10-29
