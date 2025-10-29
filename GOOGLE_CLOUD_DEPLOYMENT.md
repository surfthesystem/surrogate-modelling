# Google Cloud VM Deployment Guide

This guide explains how to deploy and run the reservoir simulator on a Google Cloud Platform (GCP) VM instance.

---

## Prerequisites

- Google Cloud Platform account
- GCP project with billing enabled
- Basic knowledge of SSH and Linux commands

---

## Step 1: Create a GCP VM Instance

### Recommended VM Configuration:

```bash
# VM Specifications
Machine type: e2-standard-4 (4 vCPU, 16 GB RAM)
# Or for larger simulations: n2-standard-8 (8 vCPU, 32 GB RAM)

OS: Ubuntu 22.04 LTS
Boot disk: 50 GB Standard persistent disk
Region: us-central1 (or your preferred region)
```

### Create VM via gcloud CLI:

```bash
gcloud compute instances create reservoir-simulator-vm \
    --project=YOUR_PROJECT_ID \
    --zone=us-central1-a \
    --machine-type=e2-standard-4 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=50GB \
    --boot-disk-type=pd-standard \
    --tags=reservoir-sim
```

### Or via GCP Console:

1. Go to **Compute Engine** > **VM instances**
2. Click **Create Instance**
3. Configure:
   - **Name**: reservoir-simulator-vm
   - **Region**: us-central1 (or preferred)
   - **Machine type**: e2-standard-4
   - **Boot disk**: Ubuntu 22.04 LTS, 50 GB
4. Click **Create**

---

## Step 2: Connect to Your VM

### Option A: Via gcloud CLI

```bash
gcloud compute ssh reservoir-simulator-vm --zone=us-central1-a
```

### Option B: Via GCP Console

1. Go to **Compute Engine** > **VM instances**
2. Click **SSH** button next to your instance

---

## Step 3: Install Python and Dependencies

Once connected to your VM:

### Update system packages:

```bash
sudo apt-get update
sudo apt-get upgrade -y
```

### Install Python 3.11 and dependencies:

```bash
# Install Python 3.11
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev

# Install pip
curl -sS https://bootstrap.pypa.io/get-pip.py | sudo python3.11

# Install git
sudo apt-get install -y git

# Install system dependencies for numpy/scipy
sudo apt-get install -y build-essential gfortran libopenblas-dev liblapack-dev
```

---

## Step 4: Clone Your Repository

```bash
# Navigate to home directory
cd ~

# Clone your repository
git clone https://github.com/YOUR_USERNAME/surrogate-modelling.git

# Navigate to project directory
cd surrogate-modelling
```

---

## Step 5: Set Up Python Virtual Environment

```bash
# Create virtual environment
python3.11 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt
```

---

## Step 6: Run the Simulator

### Phase 1: Generate Reservoir Model

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Generate reservoir data
python src/reservoir_model.py

# Convert data to IMPES format
python convert_phase1_data.py

# Verify data was created
ls -lh data/impes_input/
```

### Phase 2: Run IMPES Simulation

```bash
# Navigate to simulator directory
cd simulator

# Run simulation
python IMPES_phase1.py

# Check results
ls -lh ../results/impes_sim/
```

---

## Step 7: Transfer Results Back to Your Local Machine

### Option A: Via gcloud CLI (from your local machine)

```bash
# Copy entire results folder
gcloud compute scp --recurse \
    reservoir-simulator-vm:~/surrogate-modelling/results \
    ./local_results \
    --zone=us-central1-a

# Or copy specific files
gcloud compute scp \
    reservoir-simulator-vm:~/surrogate-modelling/results/impes_sim/Phase1_n10000_t10_days.npz \
    ./local_results/ \
    --zone=us-central1-a
```

### Option B: Via Cloud Storage

On the VM:
```bash
# Install gsutil (usually pre-installed)
sudo apt-get install -y google-cloud-sdk

# Copy results to Cloud Storage bucket
gsutil -m cp -r results/ gs://YOUR_BUCKET_NAME/reservoir-sim-results/

# Or compress first to save bandwidth
tar -czf results.tar.gz results/
gsutil cp results.tar.gz gs://YOUR_BUCKET_NAME/
```

On your local machine:
```bash
# Download from Cloud Storage
gsutil -m cp -r gs://YOUR_BUCKET_NAME/reservoir-sim-results/ ./local_results/
```

---

## Step 8: Running Large Batch Simulations

For running multiple simulations (e.g., different well configurations):

### Create a batch script:

```bash
# Create run_batch.sh
cat > run_batch.sh << 'EOF'
#!/bin/bash
source ~/surrogate-modelling/venv/bin/activate
cd ~/surrogate-modelling

echo "Starting batch simulations at $(date)"

for i in {1..10}; do
    echo "Running simulation $i of 10"
    python simulator/IMPES_phase1.py
    mv results/impes_sim/Phase1_n10000_t10_days.npz \
       results/impes_sim/Phase1_sim_${i}.npz
done

echo "Batch complete at $(date)"
EOF

chmod +x run_batch.sh
```

### Run in background with nohup:

```bash
nohup ./run_batch.sh > batch_output.log 2>&1 &

# Monitor progress
tail -f batch_output.log

# Check if still running
ps aux | grep python
```

---

## Step 9: Automating with Startup Scripts

To run simulations automatically when VM starts:

### Create startup script:

```bash
cat > startup-script.sh << 'EOF'
#!/bin/bash
cd /home/YOUR_USERNAME/surrogate-modelling
source venv/bin/activate
python simulator/IMPES_phase1.py
# Upload results to Cloud Storage
gsutil cp results/impes_sim/Phase1_n10000_t10_days.npz \
    gs://YOUR_BUCKET_NAME/results/run_$(date +%Y%m%d_%H%M%S).npz
# Optionally shutdown VM after completion
# sudo shutdown -h now
EOF
```

### Add to VM metadata:

```bash
gcloud compute instances add-metadata reservoir-simulator-vm \
    --metadata-from-file startup-script=startup-script.sh \
    --zone=us-central1-a
```

---

## Cost Optimization Tips

### 1. Use Preemptible VMs (60-91% cheaper)

```bash
gcloud compute instances create reservoir-simulator-vm \
    --preemptible \
    --machine-type=e2-standard-4 \
    # ... other options
```

**Note**: Preemptible VMs can be terminated after 24 hours. Good for fault-tolerant workloads.

### 2. Stop VM When Not in Use

```bash
# Stop VM (preserves disk, no compute charges)
gcloud compute instances stop reservoir-simulator-vm --zone=us-central1-a

# Start when needed
gcloud compute instances start reservoir-simulator-vm --zone=us-central1-a
```

### 3. Auto-shutdown After Job Completion

Add to your simulation script:
```bash
python simulator/IMPES_phase1.py
gsutil cp results/* gs://YOUR_BUCKET/
sudo shutdown -h now  # Automatically stop VM
```

### 4. Use Scheduled Instances

For regular simulation runs:
```bash
# Create instance schedule
gcloud compute resource-policies create instance-schedule sim-schedule \
    --vm-start-schedule='0 9 * * MON-FRI' \
    --vm-stop-schedule='0 18 * * MON-FRI' \
    --timezone='America/Chicago'

# Attach to instance
gcloud compute instances add-resource-policies reservoir-simulator-vm \
    --resource-policies=sim-schedule \
    --zone=us-central1-a
```

---

## Monitoring and Debugging

### Check resource usage:

```bash
# CPU and memory
htop

# Disk usage
df -h

# Monitor Python process
watch -n 1 "ps aux | grep python"
```

### View simulation logs:

```bash
# Real-time log monitoring
tail -f simulator/simulation.log

# Or redirect output when running
python simulator/IMPES_phase1.py 2>&1 | tee simulation.log
```

### Check GCP logging:

```bash
# View serial console output
gcloud compute instances get-serial-port-output reservoir-simulator-vm \
    --zone=us-central1-a
```

---

## Troubleshooting

### Issue: Out of memory

**Solution**: Upgrade to larger machine type or optimize code
```bash
# Stop VM
gcloud compute instances stop reservoir-simulator-vm --zone=us-central1-a

# Change machine type
gcloud compute instances set-machine-type reservoir-simulator-vm \
    --machine-type=e2-standard-8 \
    --zone=us-central1-a

# Start VM
gcloud compute instances start reservoir-simulator-vm --zone=us-central1-a
```

### Issue: Disk full

**Solution**: Increase disk size or clean up old results
```bash
# Check disk usage
du -sh *

# Clean up old results
rm -rf results/old_*

# Or resize disk
gcloud compute disks resize reservoir-simulator-vm \
    --size=100GB \
    --zone=us-central1-a
```

### Issue: Package installation fails

**Solution**: Install build dependencies
```bash
sudo apt-get install -y python3.11-dev build-essential gfortran
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

## Security Best Practices

### 1. Use Service Accounts

```bash
# Create service account
gcloud iam service-accounts create reservoir-sim-sa \
    --display-name="Reservoir Simulator Service Account"

# Grant necessary permissions
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:reservoir-sim-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.objectAdmin"

# Attach to VM
gcloud compute instances set-service-account reservoir-simulator-vm \
    --service-account=reservoir-sim-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com \
    --zone=us-central1-a
```

### 2. Restrict SSH Access

```bash
# Only allow SSH from your IP
gcloud compute firewall-rules create allow-ssh-from-my-ip \
    --allow=tcp:22 \
    --source-ranges=YOUR_IP_ADDRESS/32 \
    --target-tags=reservoir-sim
```

### 3. Enable OS Patching

```bash
# Automatic security updates
sudo apt-get install -y unattended-upgrades
sudo dpkg-reconfigure -plow unattended-upgrades
```

---

## Estimated Costs (us-central1)

| Resource | Configuration | Cost (USD/month) |
|----------|--------------|------------------|
| e2-standard-4 (730 hrs) | 4 vCPU, 16 GB | ~$122 |
| e2-standard-4 (Preemptible) | 4 vCPU, 16 GB | ~$36 |
| Storage (50 GB) | Standard persistent disk | ~$2 |
| Network Egress | <1 GB (typical) | ~$0.12/GB |

**Total**: ~$124-38/month depending on preemptible usage

**Cost savings**:
- Stop VM when idle: Pay only for storage (~$2/month)
- Use preemptible: Save 70%
- Commit to 1-year: Save 37%

---

## Quick Reference Commands

```bash
# SSH to VM
gcloud compute ssh reservoir-simulator-vm --zone=us-central1-a

# Copy file TO vm
gcloud compute scp local_file.txt reservoir-simulator-vm:~/ --zone=us-central1-a

# Copy file FROM vm
gcloud compute scp reservoir-simulator-vm:~/results.npz ./ --zone=us-central1-a

# Stop VM
gcloud compute instances stop reservoir-simulator-vm --zone=us-central1-a

# Start VM
gcloud compute instances start reservoir-simulator-vm --zone=us-central1-a

# Delete VM (CAUTION!)
gcloud compute instances delete reservoir-simulator-vm --zone=us-central1-a
```

---

## Next Steps

After successful deployment:

1. **Validate Results**: Compare VM results with local results
2. **Optimize Performance**: Profile code and adjust VM resources
3. **Automate Workflows**: Set up scheduled simulations
4. **Scale Up**: Run parameter sweeps across multiple VMs
5. **ML Training**: Use VM for training surrogate models (Phase 3)

---

## Support

- **GCP Documentation**: https://cloud.google.com/compute/docs
- **Project Issues**: https://github.com/YOUR_USERNAME/surrogate-modelling/issues
- **GCP Pricing Calculator**: https://cloud.google.com/products/calculator

---

**Last Updated**: 2025-10-29
