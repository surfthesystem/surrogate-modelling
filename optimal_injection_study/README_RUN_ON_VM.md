# Running LHS Optimal Injection Study on Google Cloud VM

Complete guide for deploying and running the LHS study on Google Cloud Platform.

---

## Quick Start (TL;DR)

```bash
# On Google Cloud VM
cd ~/surrogate-modelling/optimal_injection_study
python run_lhs_optimization.py   # Run LHS study (~4-6 hours)
python plot_results.py           # Generate plots
```

---

## Step-by-Step Instructions

### 1. Upload Code to GitHub

On your local machine:

```bash
cd "c:\Users\H199031\OneDrive - Halliburton\Documents\0. Landmark\10.Github Rep\surrogate modelling"

# Add new files
git add optimal_injection_study/
git add docs/OPTIMAL_INJECTION_STUDY_DESIGN.md
git commit -m "Add optimal injection LHS study with all 58 wells"
git push origin main
```

### 2. Set Up Google Cloud VM

Follow the instructions in [GOOGLE_CLOUD_DEPLOYMENT.md](../GOOGLE_CLOUD_DEPLOYMENT.md) to create your VM.

**Recommended VM configuration**:
```
Machine type: e2-standard-8 (8 vCPU, 32 GB RAM)  # Recommended for parallel runs
# Or: e2-standard-4 (4 vCPU, 16 GB RAM)  # Budget option
Boot disk: 50 GB
```

### 3. Connect to VM and Clone Repository

```bash
# Connect to VM
gcloud compute ssh reservoir-simulator-vm --zone=us-central1-a

# Clone repository
cd ~
git clone https://github.com/YOUR_USERNAME/surrogate-modelling.git
cd surrogate-modelling
```

### 4. Install Dependencies

```bash
# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install Python 3.11
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev

# Install build tools
sudo apt-get install -y build-essential gfortran libopenblas-dev liblapack-dev

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install Python packages
pip install --upgrade pip
pip install numpy scipy pandas matplotlib seaborn plotly kaleido
```

### 5. Run the LHS Study

```bash
cd ~/surrogate-modelling/optimal_injection_study

# Activate virtual environment
source ../venv/bin/activate

# Run the study (this will take 4-6 hours)
nohup python run_lhs_optimization.py > lhs_study.log 2>&1 &

# Monitor progress
tail -f lhs_study.log

# Check if still running
ps aux | grep python
```

**Expected runtime**:
- 30 samples × 15-20 minutes per sample = **7.5-10 hours** (serial)
- With multiprocessing (optional): ~2-3 hours

### 6. Monitor Progress

While the study is running:

```bash
# Watch log file
tail -f lhs_study.log

# Check completed simulations
ls lhs_results/run_*/Phase1_*.npz | wc -l

# Check directory size
du -sh lhs_results/
```

### 7. Generate Plots

After simulations complete:

```bash
# Run post-processing
python plot_results.py

# Check generated plots
ls lhs_results/plots/
```

### 8. Download Results

From your local machine:

```bash
# Download all results
gcloud compute scp --recurse \
    reservoir-simulator-vm:~/surrogate-modelling/optimal_injection_study/lhs_results \
    ./local_results \
    --zone=us-central1-a

# Or download just plots
gcloud compute scp --recurse \
    reservoir-simulator-vm:~/surrogate-modelling/optimal_injection_study/lhs_results/plots \
    ./local_plots \
    --zone=us-central1-a

# Or download summary CSV
gcloud compute scp \
    reservoir-simulator-vm:~/surrogate-modelling/optimal_injection_study/lhs_results/results_summary.csv \
    ./results_summary.csv \
    --zone=us-central1-a
```

---

## Configuration Options

### Adjust Number of Samples

Edit `run_lhs_optimization.py`:

```python
N_SAMPLES = 50  # Change from 30 to 50 for more thorough study
```

### Adjust Injection Constraint

Edit `run_lhs_optimization.py`:

```python
MAX_TOTAL_INJECTION = 36000  # Increase from 27000 STB/day
```

### Run Fewer Wells for Testing

To test with fewer wells (faster), edit the script to use only the 15 selected wells instead of all 58.

---

## Running in Parallel (Advanced)

To speed up execution using multiprocessing:

### Option 1: GNU Parallel (Recommended)

```bash
# Install GNU Parallel
sudo apt-get install -y parallel

# Create a list of simulation IDs
seq 0 29 > sim_list.txt

# Run in parallel (4 simultaneous jobs)
cat sim_list.txt | parallel -j 4 'python run_single_simulation.py {}'
```

### Option 2: Python Multiprocessing

Modify `run_lhs_optimization.py` to use multiprocessing:

```python
from multiprocessing import Pool

# In main(), replace the loop with:
with Pool(processes=4) as pool:
    pool.starmap(run_simulation, [(i, input_files[i], OUTPUT_DIR)
                                   for i in range(N_SAMPLES)])
```

---

## Troubleshooting

### Issue: Out of Memory

**Solution**: Reduce grid resolution or use larger VM

```bash
# Stop VM
gcloud compute instances stop reservoir-simulator-vm --zone=us-central1-a

# Resize to larger machine
gcloud compute instances set-machine-type reservoir-simulator-vm \
    --machine-type=e2-standard-8 \
    --zone=us-central1-a

# Restart
gcloud compute instances start reservoir-simulator-vm --zone=us-central1-a
```

### Issue: Simulation Fails

Check the log file in the specific run directory:

```bash
cat lhs_results/run_000/IMPES_phase1.log
```

### Issue: Slow Progress

**Tips**:
- Reduce N_SAMPLES
- Reduce SIMULATION_DAYS
- Use coarser time steps
- Run in parallel

### Issue: Disk Full

```bash
# Check disk usage
df -h

# Clean up old results
rm -rf lhs_results/run_*/Phase1_*.npz

# Resize disk if needed
gcloud compute disks resize reservoir-simulator-vm \
    --size=100GB \
    --zone=us-central1-a
```

---

## Cost Optimization

### Use Preemptible VM (Save 70%)

```bash
# Create preemptible VM
gcloud compute instances create reservoir-sim-preempt \
    --preemptible \
    --machine-type=e2-standard-4 \
    --zone=us-central1-a \
    # ... other options
```

**Note**: VM may be terminated after 24 hours. Save checkpoints!

### Auto-Shutdown After Completion

Add to end of `run_lhs_optimization.py`:

```python
# At end of main()
import subprocess
subprocess.run(['sudo', 'shutdown', '-h', 'now'])
```

### Schedule VM Start/Stop

```bash
# Create schedule
gcloud compute resource-policies create instance-schedule my-schedule \
    --vm-start-schedule='0 9 * * *' \
    --vm-stop-schedule='0 18 * * *' \
    --timezone='America/Chicago'

# Attach to VM
gcloud compute instances add-resource-policies reservoir-simulator-vm \
    --resource-policies=my-schedule \
    --zone=us-central1-a
```

---

## Expected Results

After completion, you'll have:

### Files Generated

```
lhs_results/
├── lhs_design.csv                    # LHS parameter samples
├── all_parameters.csv                # All injection strategies
├── results_summary.csv               # Key metrics from all runs
├── params_sample_000.json            # Parameters for each sample
├── run_000/                          # Results for each simulation
│   ├── Phase1_n10000_t90_days.npz   # Main results file
│   └── input_file_phase1.py         # Input file used
├── plots/                            # All visualizations
│   ├── evolution_sample_000.png     # Pressure/saturation evolution
│   ├── well_production_sample_000.html  # Interactive well plots
│   ├── overall_comparison.png       # Comparison of all runs
│   └── sensitivity_analysis.png     # Tornado chart
```

### Key Metrics

From `results_summary.csv`:
- Cumulative oil production (STB)
- Recovery factor (%)
- Water cut (%)
- Final average pressure (psi)
- Individual well performance

### Plots

1. **Evolution plots** (one per simulation):
   - Top row: Pressure evolution (10 snapshots)
   - Bottom row: Saturation evolution (10 snapshots)

2. **Interactive well plots** (one per simulation):
   - Left panel: 40 producer cumulative curves
   - Right panel: 18 injector cumulative curves
   - Hoverable for exact values

3. **Overall comparison**:
   - Bar chart: All simulations ranked by recovery
   - Scatter plots: Recovery vs. parameters
   - Table: Top 5 strategies

4. **Sensitivity analysis**:
   - Tornado chart: Impact of each parameter
   - Spearman correlation coefficients

---

## Batch Submission (For Many Runs)

If running 100+ samples, create a batch script:

```bash
#!/bin/bash
# run_batch.sh

source ~/surrogate-modelling/venv/bin/activate
cd ~/surrogate-modelling/optimal_injection_study

START_ID=$1
END_ID=$2

for i in $(seq $START_ID $END_ID); do
    echo "Running sample $i"
    python run_single_simulation.py $i
done

echo "Batch $START_ID-$END_ID complete"
```

Run in parallel:

```bash
chmod +x run_batch.sh

# Split into 4 batches
nohup ./run_batch.sh 0 24 > batch1.log 2>&1 &
nohup ./run_batch.sh 25 49 > batch2.log 2>&1 &
nohup ./run_batch.sh 50 74 > batch3.log 2>&1 &
nohup ./run_batch.sh 75 99 > batch4.log 2>&1 &
```

---

## Next Steps After Results

1. **Analyze optimal strategy**: Identify best injection allocation
2. **Validate**: Run longer simulation with optimal parameters
3. **Build surrogate**: Use results to train ML model
4. **Optimize**: Use surrogate for real-time optimization

---

## Support

If you encounter issues:

1. Check log files: `lhs_results/*/IMPES_phase1.log`
2. Verify data files: `ls ../data/impes_input/`
3. Test single run: Manually run one simulation
4. Check VM resources: `htop`, `df -h`

---

## Summary

```bash
# Complete workflow
git push                                    # Upload code
gcloud compute ssh reservoir-simulator-vm   # Connect to VM
git clone YOUR_REPO                        # Clone repo
python3.11 -m venv venv                    # Setup environment
pip install requirements                   # Install packages
nohup python run_lhs_optimization.py &     # Run study
python plot_results.py                     # Generate plots
gcloud compute scp --recurse results ./    # Download results
```

**Total time**: ~5-10 hours for 30 samples on e2-standard-4

**Total cost**: ~$5-10 (depending on VM size and runtime)

**Output**: Complete optimal injection strategy analysis with visualizations

---

**You're ready to find the optimal water injection strategy!** 🚀
