# How to Clean Up and Restart the LHS Study

## Problem
The simulations are failing because they're using **old input files** that were generated BEFORE we fixed the indentation bug. The new code fixes are in place, but the old buggy input files are still in `lhs_results/`.

## Solution: Delete Old Results and Start Fresh

### On Your Google Cloud VM

```bash
cd ~/surrogate-modelling/optimal_injection_study

# Make sure you have the latest code
git pull origin main

# Delete the old results folder (contains buggy input files)
rm -rf lhs_results

# Reactivate virtual environment
source venv/bin/activate

# Run the study fresh
python run_lhs_optimization.py
```

## What This Does

1. **`rm -rf lhs_results`** - Deletes all the old input files that have the indentation bug
2. **`python run_lhs_optimization.py`** - Generates NEW input files with the fix and runs all 30 simulations

## Why This is Necessary

The `run_lhs_optimization.py` script generates input files once at the beginning. When simulations failed earlier, those buggy input files stayed in the folders. Even though we fixed the code and pushed to GitHub, the OLD generated files are still there.

By deleting `lhs_results/` and running again, it will:
- Generate 30 NEW input files (with proper indentation)
- Run all 30 simulations from scratch
- Complete successfully!

##  Alternative: Keep Completed Simulations (if any succeeded)

If some simulations DID complete successfully and you want to keep them:

```bash
# Check which simulations succeeded
ls lhs_results/

# Delete only the failed ones (example: run_000 and run_001)
rm -rf lhs_results/run_000
rm -rf lhs_results/run_001
# ... delete other failed ones

# Then run again - it will skip successful ones
python run_lhs_optimization.py
```

But honestly, it's **easier and cleaner to just delete everything** and start fresh:
```bash
rm -rf lhs_results
python run_lhs_optimization.py
```

## Monitoring Progress

Once running, you can monitor with:
```bash
tail -f nohup.out  # if running in background
```

Or just watch the output for:
```
Running Simulation 001/030
✓ Simulation completed in XX minutes
```

---

**Bottom line**: Delete `lhs_results/` folder and run the script again!
