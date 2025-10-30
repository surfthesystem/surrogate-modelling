#!/bin/bash
#
# Launch script for distributed training across multiple nodes.
#
# Usage:
#   1. Edit this script to set MASTER_ADDR and NUM_NODES
#   2. Run on master node (node 0):
#      bash launch_distributed.sh 0
#   3. Run on each worker node (nodes 1, 2, 3, ...):
#      bash launch_distributed.sh 1
#      bash launch_distributed.sh 2
#      etc.

set -e

# ============================================================================
# CONFIGURATION - EDIT THESE VARIABLES
# ============================================================================

# Master node IP address (the first node that coordinates training)
MASTER_ADDR="10.128.0.2"  # This node's IP (confirmed)

# Master port for communication
MASTER_PORT=29500

# Total number of nodes participating in training
NUM_NODES=20  # Adjust this to your actual number of nodes

# Number of processes per node (usually 1 for CPU training)
NPROC_PER_NODE=1

# Batch size per node
BATCH_SIZE=4

# Number of epochs
NUM_EPOCHS=50

# Experiment name
EXP_NAME="gnn_lstm_distributed_20nodes"

# ============================================================================
# SCRIPT - DO NOT EDIT BELOW THIS LINE
# ============================================================================

# Get node rank from command line argument
if [ $# -eq 0 ]; then
    echo "Error: Please provide node rank as argument"
    echo "Usage: bash launch_distributed.sh <node_rank>"
    echo "  node_rank: 0 for master, 1, 2, 3, ... for workers"
    exit 1
fi

NODE_RANK=$1

# Validate node rank
if [ $NODE_RANK -ge $NUM_NODES ]; then
    echo "Error: Node rank $NODE_RANK exceeds total nodes $NUM_NODES"
    exit 1
fi

# Set environment variables for PyTorch distributed
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export WORLD_SIZE=$NUM_NODES
export RANK=$NODE_RANK
export LOCAL_RANK=0  # For CPU, always 0

# Print configuration
echo "======================================================================"
echo "Distributed Training Launch Configuration"
echo "======================================================================"
echo "Node rank:           $NODE_RANK / $NUM_NODES"
echo "Master address:      $MASTER_ADDR:$MASTER_PORT"
echo "Processes per node:  $NPROC_PER_NODE"
echo "Batch size per node: $BATCH_SIZE"
echo "Total batch size:    $((BATCH_SIZE * NUM_NODES))"
echo "Epochs:              $NUM_EPOCHS"
echo "Experiment name:     $EXP_NAME"
echo "======================================================================"
echo

# Navigate to project directory
cd /mnt/disks/mydata/surrogate-modelling-1

# Activate virtual environment
if [ ! -d "ml_venv" ]; then
    echo "Error: Virtual environment 'ml_venv' not found"
    echo "Please run setup first on all nodes"
    exit 1
fi

# Launch distributed training
echo "Starting training on node $NODE_RANK..."
echo

ml_venv/bin/python -m torch.distributed.launch \
    --nproc_per_node=$NPROC_PER_NODE \
    --nnodes=$NUM_NODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    ml/scripts/train_distributed.py \
    --exp_name $EXP_NAME \
    --batch_size $BATCH_SIZE \
    --epochs $NUM_EPOCHS

echo
echo "======================================================================"
echo "Training finished on node $NODE_RANK"
echo "======================================================================"
