# Quick Cleanup Checklist

## Summary
- **Disk Space to Free**: 11.2 GB
- **Dead Code to Remove**: 1,200+ lines
- **Time Required**: 15 minutes
- **Complexity**: Low

---

## IMMEDIATE ACTIONS (Do First)

### Action 1: Delete Obsolete Simulator Files
```bash
rm /mnt/disks/mydata/surrogate-modelling-1/simulator/IMPES.py
rm /mnt/disks/mydata/surrogate-modelling-1/simulator/input_file_2D.py
```

**Impact**: Remove 401 lines of superseded code
- IMPES.py is replaced by IMPES_phase1.py
- input_file_2D.py only works with legacy IMPES.py

**Verify After**:
```bash
ls simulator/*.py  # Should NOT list IMPES.py or input_file_2D.py
```

---

### Action 2: Delete Virtual Environment Bloat (BIGGEST IMPACT)
```bash
rm -rf /mnt/disks/mydata/surrogate-modelling-1/ml_venv
rm -rf /mnt/disks/mydata/surrogate-modelling-1/pip_cache
```

**Impact**: Free 11.1 GB (!!)
- ml_venv: 7.2 GB (duplicate Python environment)
- pip_cache: 3.9 GB (can be regenerated)
- These are not needed; use single .venv instead

**After Cleanup, Rebuild Once**:
```bash
cd /mnt/disks/mydata/surrogate-modelling-1
pip install -r requirements-simulator.txt -r requirements-ml.txt
```

**Verify After**:
```bash
du -sh ml_venv pip_cache 2>&1  # Should show "No such file"
du -sh .venv  # Should show ~1-2 GB
```

---

### Action 3: Delete Temporary Log Files
```bash
rm /mnt/disks/mydata/surrogate-modelling-1/*.log
rm /mnt/disks/mydata/surrogate-modelling-1/parallel_joblog.tsv
```

**Files Deleted**:
- batch_full.log (32 KB)
- batch_10scenarios.log (3.7 KB)
- generate_viz.log (4.8 KB)
- monitor_output.log (3.7 KB)
- test_run.log (208 B)
- parallel_joblog.tsv (4 KB)

**Impact**: Free 49 KB, cleaner root directory

**Verify After**:
```bash
ls -la *.log 2>&1  # Should show "No such file"
```

---

### Action 4: Delete Old Test Artifacts
```bash
rm -rf /mnt/disks/mydata/surrogate-modelling-1/optimal_injection_study/test_input_gen
```

**Impact**: Remove stale generated test files from previous runs

**Verify After**:
```bash
ls optimal_injection_study/  # Should NOT show test_input_gen/
```

---

## RECOMMENDED ACTIONS (Better Organization)

### Action 5: Delete Test Scripts (Optional - More Aggressive)
```bash
# Only do this if you're confident in the main run_lhs_optimization.py
rm /mnt/disks/mydata/surrogate-modelling-1/optimal_injection_study/quick_test.py
rm /mnt/disks/mydata/surrogate-modelling-1/optimal_injection_study/test_generate_input.py
rm /mnt/disks/mydata/surrogate-modelling-1/optimal_injection_study/test_single_run.py
rm /mnt/disks/mydata/surrogate-modelling-1/optimal_injection_study/test_import.py
rm /mnt/disks/mydata/surrogate-modelling-1/optimal_injection_study/diagnose_error.py
```

**Impact**: Remove 780 lines of development/diagnostic code

**What to Keep**:
- `run_lhs_optimization.py` - Main LHS orchestration (KEEP)
- `plot_results.py` - Result analysis utility (KEEP)

**Verify After**:
```bash
ls optimal_injection_study/*.py  # Should only show: run_lhs_optimization.py, plot_results.py
```

---

### Action 6: Reorganize Tests (Optional - Best Practice)
```bash
# Create tests subdirectory
mkdir -p /mnt/disks/mydata/surrogate-modelling-1/optimal_injection_study/tests

# Move test files there (only if Action 5 not taken)
mv optimal_injection_study/quick_test.py tests/ 2>/dev/null
mv optimal_injection_study/test_*.py tests/ 2>/dev/null
mv optimal_injection_study/diagnose_error.py tests/ 2>/dev/null
```

**Result**: Cleaner structure:
```
optimal_injection_study/
├── run_lhs_optimization.py (main code)
├── plot_results.py (utility)
└── tests/ (all test files)
    ├── quick_test.py
    ├── test_generate_input.py
    ├── test_single_run.py
    ├── test_import.py
    └── diagnose_error.py
```

---

## GIT COMMIT (Final Step)

After all cleanup actions:

```bash
cd /mnt/disks/mydata/surrogate-modelling-1

# Check what will be deleted
git status

# Stage changes
git add -A

# Verify staging
git status  # Should show deletions ready to commit

# Commit
git commit -m "Clean up obsolete code and venv bloat

- Remove legacy IMPES.py (superseded by IMPES_phase1.py)
- Remove input_file_2D.py (only for legacy simulator)
- Delete ml_venv/ and pip_cache/ (11.1 GB freed)
- Remove temporary log files
- Delete stale test artifacts from test_input_gen/

Impact:
- Saves 11.2 GB disk space
- Removes ~1,200 lines of dead code
- Cleaner repository structure
- Single .venv for all dependencies"

# Verify commit
git log -1 --stat
```

---

## Summary Table

### Deletions to Perform

| File/Dir | Size | Reason | Priority |
|----------|------|--------|----------|
| simulator/IMPES.py | 7 KB | Legacy, superseded | HIGH |
| simulator/input_file_2D.py | 6 KB | Legacy, unused | HIGH |
| ml_venv/ | 7.2 GB | Duplicate venv | CRITICAL |
| pip_cache/ | 3.9 GB | Regenerable cache | CRITICAL |
| *.log (6 files) | 49 KB | Temporary output | HIGH |
| optimal_injection_study/test_input_gen/ | ? | Stale test data | HIGH |
| optimal_injection_study/test_*.py (4 files) | 779 lines | Test code | MEDIUM |
| optimal_injection_study/diagnose_error.py | 59 lines | Diagnostic | MEDIUM |

**Total Impact**: 11.2 GB freed, 1,200+ lines of dead code removed

---

## Verification Checklist

After cleanup, verify:

- [ ] `simulator/IMPES.py` deleted
- [ ] `simulator/input_file_2D.py` deleted
- [ ] `ml_venv/` directory gone
- [ ] `pip_cache/` directory gone
- [ ] All `.log` files deleted
- [ ] `optimal_injection_study/test_input_gen/` deleted
- [ ] `optimal_injection_study/` still contains:
  - [ ] `run_lhs_optimization.py`
  - [ ] `plot_results.py`
  - [ ] `requirements.txt`
  - [ ] (Optional) `/tests/` subdirectory with test files
- [ ] Git commit created successfully
- [ ] Repository structure clean and organized

---

## Before and After

### Before Cleanup
```
11.1 GB total
├── 7.2 GB ml_venv/          <- DELETE
├── 3.9 GB pip_cache/        <- DELETE
├── 1 GB .venv/              <- KEEP
├── 110 MB ml/
├── 100+ MB results/
├── 48 KB *.log              <- DELETE
├── 104 KB optimal_injection_study/
│   ├── test_input_gen/      <- DELETE
│   ├── test_*.py (4 files)  <- DELETE
│   ├── diagnose_error.py    <- DELETE
│   ├── run_lhs_optimization.py  <- KEEP
│   └── plot_results.py      <- KEEP
└── (other files)
```

### After Cleanup
```
~50 MB total                    (11.1 GB -> ~50 MB!)
├── 1 GB .venv/              <- Single unified venv
├── 110 MB ml/
├── 100+ MB results/
├── 232 KB optimal_injection_study/
│   ├── run_lhs_optimization.py
│   ├── plot_results.py
│   ├── requirements.txt
│   └── tests/               <- Organized
│       ├── quick_test.py
│       ├── test_generate_input.py
│       ├── test_single_run.py
│       ├── test_import.py
│       └── diagnose_error.py
└── (other files)
```

---

## Troubleshooting

### Q: What if I accidentally delete something important?
A: Git has it! Recover with:
```bash
git checkout HEAD~1 -- <filename>
```

### Q: Will utils/diagnose_wells.py and diagnose_permeability.py be deleted?
A: No! Those are in `/utils/` and are useful. Only delete the one in `optimal_injection_study/diagnose_error.py`

### Q: Do I need to recreate .venv?
A: Not immediately. The .venv you have works. But after deleting ml_venv, if you want to be sure everything is in one place:
```bash
pip install -r requirements-simulator.txt -r requirements-ml.txt --upgrade
```

### Q: Can I delete the scenarios/ directory?
A: Not recommended - it contains the DOE sample definitions (100 scenarios). Keep it.

### Q: Is the PDF safe to keep?
A: Yes. "3. GNN-LSTM_Surrogate_Reservoir.pdf" (4.9 MB) is the research paper. Keep it.

---

## Expected Result

After cleanup:
1. Repository size: **11.1 GB → ~50 MB** (plus .venv)
2. Code clarity: **Better** (legacy code removed)
3. Maintenance: **Easier** (single venv, organized structure)
4. Build time: **Faster** (less clutter to navigate)

---
