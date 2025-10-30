# Distributed Training Guide - GNN-LSTM Surrogate Model

This guide explains how to train the GNN-LSTM surrogate model across multiple nodes/machines for faster training.

## Overview

- **Single Node Training**: ~1.3 it/s on CPU (~27 seconds per epoch for 70 scenarios)
- **20-Node Distributed**: ~20× speedup (theoretical), ~1.4 seconds per epoch
- **Effective Batch Size**: `batch_size_per_node × num_nodes`

---

## Prerequisites

### 1. Network Setup

All nodes must be able to communicate with each other. Check connectivity:

```bash
# From each worker node, ping the master node
ping <master-node-ip>
```

### 2. Software Setup on ALL Nodes

Each node needs the same environment. Run on **every node**:

```bash
# Clone repository (if not already done)
cd /mnt/disks/mydata
git clone <your-repo-url> surrogate-modelling-1
cd surrogate-modelling-1

# Create virtual environment
python3 -m venv ml_venv

# Install dependencies
ml_venv/bin/pip install --upgrade pip setuptools wheel
ml_venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
ml_venv/bin/pip install numpy scipy matplotlib pandas pyyaml scikit-learn \
    tqdm shapely networkx torch-geometric

# Verify installation
ml_venv/bin/python -c "import torch; import torch_geometric; print('✓ Ready')"
```

### 3. Data Setup on ALL Nodes

**Option A: Shared Filesystem** (Recommended if available)
- Mount shared storage (NFS, GCS Fuse, etc.) to same path on all nodes
- Data only needs to be on shared storage once

**Option B: Copy Data to Each Node**
```bash
# On master node, tar the data
cd /mnt/disks/mydata/surrogate-modelling-1
tar -czf data.tar.gz ml/data/preprocessed results/training_data

# Copy to each worker node
for i in {1..19}; do
    scp data.tar.gz node$i:/mnt/disks/mydata/surrogate-modelling-1/
done

# On each worker, extract
tar -xzf data.tar.gz
```

---

## Quick Start - Running Distributed Training

### Step 1: Configure the Launch Script

Edit `launch_distributed.sh`:

```bash
# Find master node's internal IP
hostname -I | awk '{print $1}'
# Example output: 10.128.0.2

# Edit launch_distributed.sh
nano launch_distributed.sh

# Set these variables:
MASTER_ADDR="10.128.0.2"  # Your master node IP
NUM_NODES=20               # Your total number of nodes
BATCH_SIZE=4               # Batch size per node
NUM_EPOCHS=50              # Total epochs
```

### Step 2: Copy Launch Script to All Nodes

```bash
# From master node
for i in {1..19}; do
    scp launch_distributed.sh node$i:/mnt/disks/mydata/surrogate-modelling-1/
done
```

### Step 3: Launch Training

**On master node (node 0):**
```bash
cd /mnt/disks/mydata/surrogate-modelling-1
bash launch_distributed.sh 0
```

**On each worker node (nodes 1-19):**
```bash
# On node 1
cd /mnt/disks/mydata/surrogate-modelling-1
bash launch_distributed.sh 1

# On node 2
bash launch_distributed.sh 2

# ... and so on for nodes 3-19
```

### Step 4: Monitor Training

Training progress is printed on each node. The master node (rank 0) will also save checkpoints.

```bash
# On master node, watch the output
tail -f nohup.out

# Or check saved models
ls -lh results/ml_experiments/gnn_lstm_distributed_20nodes/
```

---

## Alternative: Using tmux/screen for Background Execution

To run training in background on all nodes:

### On Master Node:
```bash
# Start tmux session
tmux new -s training

# Run training
bash launch_distributed.sh 0

# Detach: Press Ctrl+B, then D
# Reattach later: tmux attach -t training
```

### On Each Worker Node:
```bash
# SSH to worker
ssh node1

# Start tmux and run
tmux new -s training
bash launch_distributed.sh 1
# Detach: Ctrl+B, D
```

---

## Alternative: Parallel Launch Script

Create `launch_all_nodes.sh` to launch on all nodes from master:

```bash
#!/bin/bash
# launch_all_nodes.sh

MASTER_NODE="master-hostname"
WORKER_NODES="node1 node2 node3 node4 ..."  # List all worker nodes

# Launch on master (background)
bash launch_distributed.sh 0 > /tmp/train_rank0.log 2>&1 &

# Launch on workers via SSH (background)
i=1
for node in $WORKER_NODES; do
    ssh $node "cd /mnt/disks/mydata/surrogate-modelling-1 && \
               bash launch_distributed.sh $i > /tmp/train_rank$i.log 2>&1" &
    i=$((i + 1))
done

echo "Distributed training launched on all nodes"
echo "Monitor logs at /tmp/train_rank*.log on each node"
```

---

## Troubleshooting

### Issue: "Connection refused" or "Timeout"

**Cause**: Nodes can't communicate

**Fix**:
1. Check firewall rules: `sudo ufw status`
2. Allow port 29500: `sudo ufw allow 29500/tcp`
3. Verify connectivity: `telnet <master-ip> 29500`

### Issue: "Address already in use"

**Cause**: Previous training session didn't clean up

**Fix**:
```bash
# Kill any existing training processes
pkill -f train_distributed.py

# Change port in launch_distributed.sh
MASTER_PORT=29501  # Use different port
```

### Issue: Different data on different nodes

**Cause**: Data not synced across nodes

**Fix**:
```bash
# Verify data checksums match
md5sum ml/data/preprocessed/graph_data.npz
md5sum results/training_data/doe_0000/doe_0000.npz

# Re-copy data from master to workers if mismatched
```

### Issue: "ModuleNotFoundError"

**Cause**: Virtual environment not set up on all nodes

**Fix**: Re-run installation on affected node (see Prerequisites #2)

### Issue: Training hangs at initialization

**Cause**: Not all nodes started, or one node failed

**Fix**:
1. Ensure all nodes running: `ps aux | grep train_distributed`
2. Check logs on each node
3. Restart failed nodes

---

## Performance Tuning

### 1. Batch Size

- Start with small batch size per node (2-4)
- Increase if nodes have more RAM available
- Total batch size = `batch_size × num_nodes`

### 2. Number of Workers

```bash
# In launch_distributed.sh, can increase dataloader workers
# Edit train_distributed.py, line with num_workers=0:
num_workers=2  # Use 2-4 workers per node
```

### 3. Network Optimization

For Google Cloud / AWS:
- Use nodes in same zone/region
- Use high-bandwidth instance types
- Enable jumbo frames if supported

---

## Expected Performance

### Single Node (Baseline)
- 70 scenarios, batch_size=2
- ~1.27 it/s
- ~27 seconds per epoch
- **50 epochs**: ~23 minutes

### 20 Nodes (Distributed)
- 70 scenarios, batch_size=4 per node
- Effective batch_size=80
- ~20× speedup (theoretical)
- ~1.4 seconds per epoch
- **50 epochs**: ~1-2 minutes

### 100 Scenarios (Full Dataset)
- Single node: ~50 epochs in ~33 minutes
- 20 nodes: ~50 epochs in ~2 minutes

---

## Saving and Loading Models

### Checkpoints

Saved only on master node (rank 0):
```
results/ml_experiments/gnn_lstm_distributed_20nodes/
├── best_model.pth
├── checkpoint_epoch_10.pth
├── checkpoint_epoch_20.pth
└── ...
```

### Loading for Inference

```python
import torch
from ml.models.surrogate import GNN_LSTM_Surrogate

# Load checkpoint
checkpoint = torch.load('results/ml_experiments/.../best_model.pth')

# Initialize model
model = GNN_LSTM_Surrogate(...)

# Handle DDP wrapper (model was wrapped with DistributedDataParallel)
if 'module.' in list(checkpoint['model_state_dict'].keys())[0]:
    # Remove 'module.' prefix from state dict keys
    state_dict = {k.replace('module.', ''): v
                  for k, v in checkpoint['model_state_dict'].items()}
else:
    state_dict = checkpoint['model_state_dict']

model.load_state_dict(state_dict)
model.eval()
```

---

## Next Steps

After distributed training completes:

1. **Evaluate Model**: Run `ml/scripts/evaluate.py` (to be implemented)
2. **Hyperparameter Tuning**: Run multiple experiments in parallel across nodes
3. **Increase Dataset**: Scale to 400-500 scenarios like the paper

---

## References

- [PyTorch Distributed Tutorial](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [GNN-LSTM Surrogate Paper](docs/3.%20GNN-LSTM_Surrogate_Reservoir.pdf)
- [Project Overview](PROJECT_OVERVIEW.md)

---

**Questions?** Check logs on each node or open an issue.
