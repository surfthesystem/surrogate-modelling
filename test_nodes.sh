#!/bin/bash
#
# Test connectivity to worker nodes and prepare for distributed training
#

echo "======================================================================"
echo "Testing Node Connectivity"
echo "======================================================================"
echo

# List your worker node hostnames or IPs here
# Example: WORKER_NODES="node1 node2 node3"
# Or: WORKER_NODES="10.128.0.3 10.128.0.4 10.128.0.5"
WORKER_NODES=""

if [ -z "$WORKER_NODES" ]; then
    echo "Please edit this script and set WORKER_NODES variable"
    echo "Example: WORKER_NODES=\"node1 node2 node3 node4\""
    echo "Or with IPs: WORKER_NODES=\"10.128.0.3 10.128.0.4 10.128.0.5\""
    exit 1
fi

echo "Master Node: $(hostname -I | awk '{print $1}')"
echo "Worker Nodes: $WORKER_NODES"
echo

# Test SSH connectivity
echo "1. Testing SSH connectivity..."
for node in $WORKER_NODES; do
    if ssh -o ConnectTimeout=5 -o BatchMode=yes $node "echo 'OK'" &>/dev/null; then
        echo "  ✓ $node - SSH working"
    else
        echo "  ✗ $node - SSH failed (setup passwordless SSH)"
    fi
done
echo

# Test if project directory exists
echo "2. Checking if project exists on worker nodes..."
for node in $WORKER_NODES; do
    if ssh $node "test -d /mnt/disks/mydata/surrogate-modelling-1"; then
        echo "  ✓ $node - Project directory exists"
    else
        echo "  ✗ $node - Project directory missing"
    fi
done
echo

# Test if virtual environment exists
echo "3. Checking if ml_venv exists on worker nodes..."
for node in $WORKER_NODES; do
    if ssh $node "test -d /mnt/disks/mydata/surrogate-modelling-1/ml_venv"; then
        echo "  ✓ $node - ml_venv exists"
    else
        echo "  ✗ $node - ml_venv missing (run setup)"
    fi
done
echo

# Test if data exists
echo "4. Checking if data exists on worker nodes..."
for node in $WORKER_NODES; do
    if ssh $node "test -f /mnt/disks/mydata/surrogate-modelling-1/ml/data/preprocessed/graph_data.npz"; then
        echo "  ✓ $node - Preprocessed data exists"
    else
        echo "  ✗ $node - Data missing (copy from master)"
    fi
done
echo

echo "======================================================================"
echo "Setup Commands"
echo "======================================================================"
echo
echo "To copy environment to all nodes:"
echo "  for node in $WORKER_NODES; do"
echo "    rsync -avz --exclude='*.pyc' --exclude='__pycache__' \\"
echo "      /mnt/disks/mydata/surrogate-modelling-1/ \\"
echo "      \$node:/mnt/disks/mydata/surrogate-modelling-1/"
echo "  done"
echo
echo "To launch training on all nodes:"
echo "  bash launch_distributed.sh 0  # Run on THIS master node"
echo "  # Then SSH to each worker and run:"
echo "  # bash launch_distributed.sh 1  # on node 1"
echo "  # bash launch_distributed.sh 2  # on node 2"
echo "  # etc."
