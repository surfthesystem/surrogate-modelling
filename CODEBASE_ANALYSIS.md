# Comprehensive Codebase Analysis: Surrogate Modelling Project

**Date**: 2025-10-30
**Location**: `/mnt/disks/mydata/surrogate-modelling-1`
**Repository**: Git repo (main branch)

---

## Executive Summary

This is a **Reservoir Simulation + GNN-LSTM Surrogate Modeling** project with the following state:

- **Code Quality**: Good organization with clear separation of concerns
- **Project Maturity**: Early-to-mid stage; simulator fully functional, ML training pipeline incomplete
- **Disk Usage**: 11.1 GB total (7.2 GB ml_venv, 3.9 GB pip_cache - can be removed)
- **Dead Code**: ~1800 lines (test files and obsolete studies)
- **Actionable Issues**: Organization inconsistencies, missing ML training scripts, venv clutter

---

## PART 1: CURRENT PROJECT STRUCTURE

### Overview
```
Root (11.1 GB)
├── Simulator Layer (IMPES Phase 1)
├── Data Generation Layer (LHS Design of Experiments)
├── ML Infrastructure Layer (GNN-LSTM model, incomplete training)
├── Utilities & Scripts
└── Virtual Environments (11.1 GB of 11.1 GB total!)
```

### Directory Breakdown

#### 1. Core Simulator (`/simulator/`)
**Status**: Production-ready
**Purpose**: 2-phase IMPES reservoir simulator for waterflooding

**Essential Files** (230 KB):
- `IMPES_phase1.py` (500 lines) - Main Phase 1 simulator with monthly reallocation
- `IMPES.py` (210 lines) - Legacy 2D simulator (superseded by IMPES_phase1)
- `input_file_phase1.py` (372 lines) - Current input configuration
- `input_file_phase1_TEMPLATE.py` (273 lines) - Template for code generation
- `input_file_2D.py` (191 lines) - Legacy 2D input (not used)
- Support modules (myarrays, updatewells, rel_perm, etc.)

**Organization Issue**: Two versions of IMPES (IMPES.py and IMPES_phase1.py) coexist
- IMPES.py is legacy (2D, no monthly reallocation)
- IMPES_phase1.py is current (Phase 1, monthly control)
- input_file_2D.py only works with IMPES.py (unused)

#### 2. Data & Configuration (`/data/`)
**Status**: Active
**Size**: 484 KB

**Essential Files**:
- `config.yaml` - Main project config (well definitions, control parameters)
- `reservoir_config.json` - Reservoir model parameters
- `well_locations.csv` - Current well locations (15 wells)
- `/impes_input/` - Pre-processed data for IMPES
  - `depth.txt`, `porosity.txt`, `permeability.txt` (100x100 grid)
  - `schedule_*.csv` - Monthly control schedules
  - `well_configuration.csv` - Well setup

**Data Files**:
- `porosity_field.npy` / `permeability_field.npy` - Generated fields

#### 3. Simulator Data Generation (`/optimal_injection_study/`)
**Status**: Experimental/Test code
**Purpose**: LHS-based design of experiments for injection scenarios
**Size**: 104 KB

**Code Hierarchy**:
```
run_lhs_optimization.py (516 lines)
├── Main LHS orchestration
├── Generates input files on-the-fly
└── Runs simulations in parallel

Test/Diagnostic Files (do not use):
├── quick_test.py (322 lines) - Single 10-day test
├── test_generate_input.py (187 lines) - Input file generation test
├── test_single_run.py (183 lines) - Single simulation test
├── test_import.py (29 lines) - Simple import test
├── diagnose_error.py (59 lines) - Error diagnosis script
└── test_input_gen/ - Generated test inputs (old runs)
```

**Issue**: This directory is titled "optimal_injection_study" but contains generic LHS infrastructure.
Should probably be renamed to `lhs_runner/` or `batch_simulator_lhs/`.

#### 4. Utilities (`/utils/`)
**Status**: Active, well-organized
**Size**: 64 KB

**Key Tools**:
- `scenario_runner.py` (234 lines) - Execute 12-month scenarios from JSON definitions
- `batch_simulator.py` (323 lines) - Batch run coordinator with parallel execution
- `generate_visualizations.py` (411 lines) - Post-processing and plotting
- `doe_sampler.py` (149 lines) - Design of Experiments sampling
- `diagnose_wells.py` (92 lines) - Well positioning diagnostic
- `diagnose_permeability.py` (82 lines) - Permeability field diagnostic

**Organization**: Clean, focused utilities. No issues.

#### 5. ML Pipeline (`/ml/`)
**Status**: Implemented but incomplete
**Size**: 228 KB
**Missing**: Training & evaluation scripts

**Structure**:
```
ml/
├── data/ (4/4 modules complete)
│   ├── graph_builder.py (349 lines) ✓ Voronoi + k-NN graphs
│   ├── preprocessing.py (402 lines) ✓ 10-D edge features
│   ├── normalizers.py (293 lines) ✓ Feature scaling
│   └── dataset.py (370 lines) ✓ PyTorch Dataset
├── models/ (4/4 modules complete)
│   ├── gnn.py (364 lines) ✓ Enhanced GNN
│   ├── lstm.py (339 lines) ✓ Temporal LSTM variants
│   ├── surrogate.py (302 lines) ✓ Full model
│   └── losses.py (339 lines) ✓ Custom loss functions
├── training/
│   ├── config.yaml ✓ Hyperparameters
│   ├── trainer.py ✗ MISSING
│   └── evaluator.py ✗ MISSING
├── utils/ (1/2 modules complete)
│   ├── helpers.py (307 lines) ✓
│   └── visualization.py ✗ MISSING
├── scripts/ (0/3 complete)
│   ├── train.py ✗ MISSING
│   ├── preprocess_all.py ✗ MISSING
│   └── evaluate.py ✗ MISSING
└── notebooks/ (EMPTY - 0/4 notebooks)
    ├── 01_data_exploration.ipynb ✗
    ├── 02_edge_features.ipynb ✗
    ├── 03_model_training.ipynb ✗
    └── 04_results_analysis.ipynb ✗
```

**Status**: Data loading + model architecture complete; training pipeline incomplete.

#### 6. Source Code (`/src/`)
**Status**: Active
**Size**: 28 KB

**Files**:
- `reservoir_model.py` (594 lines) - Generates synthetic reservoir with permeability/porosity fields
- `__init__.py` - Package initialization

#### 7. Scenarios & Results
**Size**: 1.1 GB (includes 100 training datasets)

`/scenarios/`:
- 100 JSON files (doe_0000.json - doe_0099.json) - LHS sampled control parameters
- Each: ~1.8 KB (producer BHP, injector rates, simulation time)

`/results/`:
- `impes_sim/` - 100 simulation outputs (NPZ + plots)
- `training_data/` - Preprocessed training data for ML (~102 runs)
- `ml_experiments/` - Empty (ready for ML training outputs)
- Visualization PNGs (reservoir, permeability histogram)

---

## PART 2: DEAD/UNUSED CODE (Candidates for Deletion)

### A. Legacy Simulator Files (Should Delete)

| File | Lines | Status | Reason |
|------|-------|--------|--------|
| `/simulator/IMPES.py` | 210 | **SUPERSEDED** | Replaced by IMPES_phase1.py (no monthly reallocation) |
| `/simulator/input_file_2D.py` | 191 | **OBSOLETE** | For legacy IMPES.py (2D, no longer used) |
| `/simulator/input_file_phase1.py` | 372 | **CONFIG FILE** | Actually used; generated by `convert_phase1_data.py` |

**Action**: Delete IMPES.py and input_file_2D.py

---

### B. Test/Diagnostic Files in `optimal_injection_study/` (Can Delete)

These are temporary test scripts from development:

| File | Lines | Purpose | Status | Recommendation |
|------|-------|---------|--------|-----------------|
| `quick_test.py` | 322 | Test 1 sim in 10 days | Development | **DELETE** - superseded by utils/scenario_runner.py |
| `test_generate_input.py` | 187 | Test input file generation | Development | **DELETE** - diagnostic only |
| `test_single_run.py` | 183 | Test 1 simulation | Development | **DELETE** - diagnostic only |
| `test_import.py` | 29 | Test import functionality | Minimal | **DELETE** - no value |
| `diagnose_error.py` | 59 | Debug import errors | Development | **DELETE** - fixed in recent commits |
| `plot_results.py` | 571 | Plot LHS results | Analysis | **KEEP** - may be useful for result analysis |
| `run_lhs_optimization.py` | 516 | Main LHS orchestration | Active | **KEEP** - main executable |
| `/test_input_gen/` | Old | Generated test inputs | Stale | **DELETE** - previous run output |

**Total to Delete**: ~1200 lines + old test directory

---

### C. Log Files (Temporary Output)

| File | Size | Status |
|------|------|--------|
| `batch_full.log` | 32 KB | Execution log from LHS run |
| `batch_10scenarios.log` | 3.7 KB | Partial run log |
| `generate_viz.log` | 4.8 KB | Visualization generation log |
| `monitor_output.log` | 3.7 KB | Monitor script output |
| `test_run.log` | 208 B | Minimal test output |
| `parallel_joblog.tsv` | 4 KB | GNU parallel job log |

**Action**: All logs can be deleted (regenerated on each run)

---

### D. Virtual Environments (Performance Bloat!)

| Directory | Size | Status | Recommendation |
|-----------|------|--------|-----------------|
| `.venv/` | ~1 GB | Simulator dependencies | **KEEP** - needed for simulator |
| `ml_venv/` | 7.2 GB | ML dependencies (PyTorch, GNN libs) | **KEEP but REBUILD** - outdated, should be in .venv |
| `pip_cache/` | 3.9 GB | PIP cache | **DELETE** - can be regenerated |

**Action**: Delete `pip_cache/`, optionally merge `ml_venv` into `.venv`

---

### E. Documentation (Superseded)

The following files show git status as **Deleted** (D), meaning they're in git history but removed:

```
D CLEANUP_REPORT.md                          (superseded - context lost)
D CRITICAL_FIX.md                            (obsolete - issue fixed)
D GITHUB_PUSH_INSTRUCTIONS.md                (one-time use)
D GNN_LSTM_Reservoir_Surrogate_Specification.md (superseded by ml/README.md)
D GOOGLE_CLOUD_DEPLOYMENT.md                 (not active)
D INSTALL_PYTHON.md                          (superseded by requirements files)
D PHASE1_SETUP.md                            (superseded by QUICKSTART.md)
D PROJECT_STRUCTURE.md                       (incomplete)
D archive/                                   (entire directory ~4 GB)
```

These have already been removed from git (git status shows "D"). Archive can stay if needed for historical reference.

---

## PART 3: CODE ORGANIZATION ISSUES

### Issue 1: Duplicate Simulator Implementations
**Severity**: Medium
**Files Affected**: 
- `simulator/IMPES.py` vs `simulator/IMPES_phase1.py`
- `simulator/input_file_2D.py` vs `simulator/input_file_phase1_TEMPLATE.py`

**Problem**: 
- Two incompatible IMPES versions coexist
- Legacy code creates confusion about which to use
- IMPES.py has no monthly reallocation (outdated for current use)

**Fix**: 
- Delete IMPES.py and input_file_2D.py
- Keep IMPES_phase1.py as the canonical simulator
- Update any references

---

### Issue 2: Misnamed Directory
**Severity**: Low
**Path**: `/optimal_injection_study/`

**Problem**:
- Name suggests optimization-specific code
- Actually contains generic LHS infrastructure
- Could apply to any design of experiments

**Fix**:
```bash
mv optimal_injection_study/ lhs_runner/
# Or: batch_simulator_lhs/
```

Update references in:
- README.md
- run_lhs_study.py (if any)
- Documentation

---

### Issue 3: Empty ML Pipeline Directories
**Severity**: Low
**Paths**: 
- `/ml/scripts/` - empty (should contain train.py, evaluate.py, preprocess_all.py)
- `/ml/notebooks/` - empty (should contain 4 example notebooks)

**Problem**:
- Structure suggests completed ML pipeline
- Actual training code missing
- Notebooks promised but not provided

**Status**: Documented in ml/IMPLEMENTATION_STATUS.md and ml/NEXT_STEPS.md

**Fix**: Either:
1. Complete the training scripts (3-4 hours work)
2. Remove the directory structure and move to `ml_deprecated/` if not used

---

### Issue 4: Multiple Venv Directories
**Severity**: High (disk usage)
**Paths**: `.venv/` (1 GB) + `ml_venv/` (7.2 GB)

**Problem**:
- Two separate virtual environments
- Duplicated dependencies
- PIP cache adds 3.9 GB
- Total: 11.1 GB (nearly 100% of repo size!)

**Recommendation**:
1. Use single `.venv/` with both simulator + ML dependencies
2. Delete `ml_venv/` and `pip_cache/` (freeing ~11 GB)
3. Requirements already split: `requirements-simulator.txt` and `requirements-ml.txt`

**Fix**:
```bash
pip install -r requirements-simulator.txt -r requirements-ml.txt
# Clean up
rm -rf ml_venv/ pip_cache/
```

---

### Issue 5: Input File Management
**Severity**: Medium
**Files**:
- `/simulator/input_file_phase1_TEMPLATE.py` (template for code generation)
- `/simulator/input_file_phase1.py` (generated/actual configuration)

**Problem**:
- TEMPLATE and actual file coexist and may diverge
- Generated files typically should not be in version control
- Code generation pattern is fragile

**Current Status**: This is working but could be cleaner

**Recommendation**: 
- Consider .gitignore for generated input files
- Or add documentation explaining the relationship

---

### Issue 6: Test Files Cluttering optimal_injection_study/
**Severity**: Low
**Files**:
```
optimal_injection_study/
├── quick_test.py              (test)
├── test_generate_input.py     (test)
├── test_single_run.py         (test)
├── test_import.py             (test)
├── diagnose_error.py          (diagnostic)
├── test_input_gen/            (old test outputs)
├── run_lhs_optimization.py    (actual code)
└── plot_results.py            (utility)
```

**Problem**: 5-6 test files for a single main script creates clutter

**Recommendation**:
- Create `tests/` subdirectory
- Move test files there
- Keep only run_lhs_optimization.py and plot_results.py at root

---

## PART 4: RECOMMENDATIONS BY PRIORITY

### PRIORITY 1: Quick Wins (Delete These)

1. **Delete Simulator Obsolescence** (5 min)
   - Remove: `/simulator/IMPES.py`
   - Remove: `/simulator/input_file_2D.py`
   - Update git tracking

2. **Delete Virtual Environment Cache** (5 min, saves ~11 GB)
   - Remove: `ml_venv/`
   - Remove: `pip_cache/`
   - Merge into single `.venv`

3. **Delete Test/Log Files** (5 min)
   - Remove: All `.log` files
   - Remove: `/optimal_injection_study/test_input_gen/`
   - Remove: Test files from optimal_injection_study/ (or move to tests/ dir)

**Total Impact**: Free ~11.2 GB, remove ~1200 lines of dead code

---

### PRIORITY 2: Organization Improvements (30 min)

1. **Rename Directory**
   ```bash
   mv optimal_injection_study/ batch_simulator_lhs/
   ```

2. **Create Tests Subdirectory**
   ```bash
   mkdir optimal_injection_study/tests/
   mv quick_test.py test_generate_input.py test_single_run.py test_import.py diagnose_error.py tests/
   ```

3. **Update Documentation**
   - Update README.md with correct directory names
   - Add section on test files location

---

### PRIORITY 3: Missing ML Training Scripts (3-4 hours)

Complete the ML pipeline by adding:
1. `ml/scripts/train.py` (~200 lines) - Main training entry point
2. `ml/training/trainer.py` (~300 lines) - Training loop
3. `ml/scripts/preprocess_all.py` (~150 lines) - Preprocessing orchestration
4. `ml/utils/visualization.py` (~200 lines) - Result plotting

Current status is documented in `ml/IMPLEMENTATION_STATUS.md` and `ml/NEXT_STEPS.md`

---

## PART 5: SUMMARY TABLE

### Files/Dirs to DELETE
| Item | Type | Size | Reason |
|------|------|------|--------|
| simulator/IMPES.py | File | 7 KB | Superseded by IMPES_phase1.py |
| simulator/input_file_2D.py | File | 6 KB | Legacy (only for IMPES.py) |
| ml_venv/ | Directory | 7.2 GB | Duplicate venv |
| pip_cache/ | Directory | 3.9 GB | Can be regenerated |
| *.log | Files | 48 KB | Temporary logs |
| optimal_injection_study/test_*.py | Files | 399 lines | Development/test only |
| optimal_injection_study/diagnose_error.py | File | 59 lines | Diagnostic only |
| optimal_injection_study/test_input_gen/ | Directory | Old | Stale test outputs |

**Total**: ~11.2 GB freed, ~1200 lines of dead code removed

---

### Files/Dirs to KEEP (Essential)
| Item | Type | Purpose |
|------|------|---------|
| simulator/IMPES_phase1.py | File | Main simulator (active) |
| simulator/input_file_phase1_TEMPLATE.py | File | Code generation template |
| simulator/input_file_phase1.py | File | Generated config (in use) |
| optimal_injection_study/run_lhs_optimization.py | File | LHS orchestration (active) |
| optimal_injection_study/plot_results.py | File | Result analysis utility |
| .venv/ | Directory | Simulator dependencies |
| ml/ | Directory | Model infrastructure (8/11 complete) |
| utils/ | Directory | Well-organized utilities |
| data/ | Directory | Configuration and inputs |
| src/reservoir_model.py | File | Data generation |

---

### Files/Dirs to IMPROVE
| Item | Type | Issues |
|------|------|--------|
| ml/scripts/ | Directory | Empty (missing train.py, evaluate.py, preprocess_all.py) |
| ml/notebooks/ | Directory | Empty (should have 4 examples) |
| ml/training/trainer.py | File | Missing (3-4 hours work) |
| optional_injection_study/ | Directory | Name is misleading (should be batch_simulator_lhs/) |

---

## PART 6: GIT STATUS INTERPRETATION

Current git status shows 270+ deleted documentation files. These are intentional cleanups already done.

**Already Deleted** (not in working tree):
- Old docs (CLEANUP_REPORT.md, CRITICAL_FIX.md, etc.)
- Archive directory with ~200 old class examples
- Ancient cover photos and examples

**Recently Modified** (watch for):
- `.claude/settings.local.json` - Claude Code settings (local only)
- `.gitignore` - Updated to exclude venv
- `QUICKSTART.md` and `README.md` - Updated docs
- `config.yaml` - Updated configuration
- `data/` files - Modified by data generation

---

## FINAL RECOMMENDATIONS

### Short Term (Do Now)
1. Delete 5 obsolete files (11.2 GB saved)
2. Move test files to subdirectory (cleaner structure)
3. Rebuild single .venv with all requirements

### Medium Term (This Week)
1. Rename `optimal_injection_study/` → `batch_simulator_lhs/`
2. Add missing ML training scripts (3-4 hours)
3. Create example notebooks for ML pipeline

### Long Term (Next Phase)
1. Complete ML training and evaluation
2. Document the full pipeline end-to-end
3. Consider publishing to academic venues (paper already in repo!)

---

## File Checklist for Cleanup

**EXECUTE THESE COMMANDS:**
```bash
# 1. Remove obsolete simulator files
rm simulator/IMPES.py
rm simulator/input_file_2D.py

# 2. Remove venv clutter
rm -rf ml_venv pip_cache

# 3. Remove logs and temp files  
rm *.log
rm parallel_joblog.tsv

# 4. Remove test outputs
rm -rf optimal_injection_study/test_input_gen

# 5. Reorganize test files (optional)
mkdir -p optimal_injection_study/tests
mv optimal_injection_study/test_*.py optimal_injection_study/tests/ 2>/dev/null
mv optimal_injection_study/diagnose_error.py optional_injection_study/tests/ 2>/dev/null

# 6. Update git
git add -A
git status  # Verify changes
git commit -m "Clean up obsolete code and venv clutter"
```

---

