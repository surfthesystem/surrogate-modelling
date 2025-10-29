# Archive Folder

This folder contains reference materials and old experimental results that are not part of the active codebase but may be useful for historical reference.

## Contents

### class_examples/
Contains all 17 class problem examples from the original Reservoir-Simulator repository by Mohammad Shadab. These are educational examples demonstrating various reservoir simulation concepts:

- Problem_0 through Problem_14: Progressive examples covering 1D/2D flow, multiphase flow, gravity, wells, capillary pressure
- Additional petrophysics examples

**Note:** The active simulator code in `simulator/` is based on Problem_14new_FullTwoPhaseComplex but has been customized for this project.

### old_results/
Contains results from earlier experimental simulations that were superseded by the validated IMPES implementation:

- `test_simulation/`: Early single-phase test results
- `multiphase_sim/`: Custom IMPES implementation that was replaced by the validated version

## Attribution

The original Reservoir-Simulator code was developed by Mohammad Afzal Shadab.
See `Reservoir-Simulator/readme.md` for full attribution and licensing information.

## Active Code Location

The current production simulator is located in:
- `simulator/` - IMPES-based two-phase reservoir simulator
- `src/` - Phase 1 reservoir model generation

Validated results are in:
- `results/impes_sim/` - Current simulation outputs

---

Last Updated: 2025-10-29
