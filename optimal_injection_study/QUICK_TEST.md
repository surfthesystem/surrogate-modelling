# Quick Test Before Running Full LHS Study

## Bug Fixed
- **Scipy import error** in `IMPES_phase1.py` has been fixed
- Changed from deprecated `scipy.sparse.construct.eye` to `scipy.sparse.eye`
- This fix has been committed and pushed to GitHub

## On Google Cloud VM

### 1. Update the repository
```bash
cd ~/surrogate-modelling
git pull origin main
```

### 2. Set up Python environment
```bash
cd optimal_injection_study

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install numpy scipy matplotlib pandas pyyaml plotly scikit-learn
```

### 3. Run the full LHS study
```bash
# Run 30 simulations (will take ~7-10 hours)
python run_lhs_optimization.py
```

### 4. Monitor progress
The script will print progress updates like:
```
======================================================================
Running Simulation 001/030
======================================================================
```

### 5. After completion, generate plots
```bash
python plot_results.py
```

### 6. Download results to your local machine
Open a terminal on your **local machine** and run:
```bash
gcloud compute scp --recurse vanlaar_ca@instance-20251029-113858:~/surrogate-modelling/optimal_injection_study/lhs_results ./lhs_results --zone=us-central1-c
```

## Expected Results

After completion, you'll have:
- `lhs_results/` folder with 30 simulation runs
- Evolution plots showing pressure & saturation at 10 timesteps
- Interactive HTML plots for cumulative production per well
- Comparison plot ranking all 30 scenarios by oil recovery
- `summary_results.json` with key metrics

## Troubleshooting

If simulations still fail:
1. Check the error in `lhs_results/run_XXX/` folders
2. Look at the generated `input_sample_XXX.py` files
3. Verify that `../data/impes_input/` files exist

## Cost Optimization

To save money, use a preemptible VM and set auto-shutdown:
```bash
# Install auto-shutdown script
sudo apt-get install at
echo "sudo poweroff" | at now + 12 hours
```

This will automatically shut down the VM after 12 hours (should be enough for 30 simulations).
