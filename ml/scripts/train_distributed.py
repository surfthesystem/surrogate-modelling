#!/usr/bin/env python
"""
Distributed training script for GNN-LSTM surrogate model using PyTorch DDP.

This script enables training across multiple nodes/machines using PyTorch's
DistributedDataParallel (DDP) for data parallelism.

Usage:
    # Single node, multiple processes (if you had GPUs)
    python -m torch.distributed.launch --nproc_per_node=2 ml/scripts/train_distributed.py

    # Multi-node setup (run on each node)
    # On master node (node_rank=0):
    python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --nnodes=20 \
        --node_rank=0 \
        --master_addr="<master-ip>" \
        --master_port=29500 \
        ml/scripts/train_distributed.py --config ml/training/config.yaml

    # On worker nodes (node_rank=1,2,3,...):
    python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --nnodes=20 \
        --node_rank=<node-rank> \
        --master_addr="<master-ip>" \
        --master_port=29500 \
        ml/scripts/train_distributed.py --config ml/training/config.yaml
"""

import argparse
import yaml
import torch
import torch.distributed as dist
import numpy as np
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.models.surrogate import GNN_LSTM_Surrogate
from ml.training.trainer import Trainer
from ml.data.simple_dataset import SimpleReservoirDataset, collate_batch
from torch.utils.data import DataLoader, DistributedSampler


def setup_distributed():
    """Initialize distributed training environment."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print("Not using distributed mode")
        return 0, 1, 0

    dist.init_process_group(backend='gloo')  # Use 'gloo' for CPU, 'nccl' for GPU

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = rank  # For CPU, local_rank = rank

    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def create_distributed_dataloaders(
    preprocessed_dir,
    batch_size=8,
    train_split=0.7,
    val_split=0.15,
    num_workers=0,
    rank=0,
    world_size=1,
):
    """
    Create dataloaders with DistributedSampler for multi-node training.

    Each node gets a subset of the data to process in parallel.
    """
    # Load preprocessed graph data
    graph_path = Path(preprocessed_dir) / 'graph_data.npz'
    graph_data = np.load(graph_path)

    # Load scenario list
    scenario_list_path = Path(preprocessed_dir) / 'scenario_list.txt'
    with open(scenario_list_path) as f:
        scenario_paths = [line.strip() for line in f.readlines()]

    if rank == 0:
        print(f"Found {len(scenario_paths)} scenarios")

    # Split into train/val/test
    num_scenarios = len(scenario_paths)
    num_train = int(num_scenarios * train_split)
    num_val = int(num_scenarios * val_split)

    train_scenarios = scenario_paths[:num_train]
    val_scenarios = scenario_paths[num_train:num_train+num_val]
    test_scenarios = scenario_paths[num_train+num_val:]

    if rank == 0:
        print(f"Split: {len(train_scenarios)} train, {len(val_scenarios)} val, {len(test_scenarios)} test")
        print(f"World size: {world_size} nodes")
        print(f"Effective batch size: {batch_size} per node × {world_size} nodes = {batch_size * world_size}")

    # Create datasets
    train_dataset = SimpleReservoirDataset(train_scenarios, graph_data)
    val_dataset = SimpleReservoirDataset(val_scenarios, graph_data)
    test_dataset = SimpleReservoirDataset(test_scenarios, graph_data)

    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )

    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_batch,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        collate_fn=collate_batch,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_batch,
        pin_memory=False,
    )

    return train_loader, val_loader, test_loader, graph_data, train_sampler


def main():
    parser = argparse.ArgumentParser(description='Distributed training for GNN-LSTM surrogate')
    parser.add_argument('--config', default='ml/training/config.yaml', help='Path to config file')
    parser.add_argument('--exp_name', default='gnn_lstm_distributed', help='Experiment name')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size per node (overrides config)')
    args = parser.parse_args()

    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    is_main_process = (rank == 0)

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        if is_main_process:
            print(f"Config file not found: {config_path}, using defaults")
        config = {
            'experiment': {'name': args.exp_name, 'seed': 42},
            'training': {
                'num_epochs': 50,
                'batch_size': 8,
                'learning_rate': 0.001,
                'weight_decay': 1e-5,
                'gradient_clip': 1.0,
                'save_every': 10,
                'early_stopping': {'patience': 10, 'min_delta': 1e-4},
                'lr_scheduler': {'type': 'StepLR', 'step_size': 20, 'gamma': 0.5},
            },
            'loss': {'beta': 80, 'alpha': 1},
            'model': {
                'gnn': {'hidden_dim': 64, 'num_layers': 3},
                'lstm': {'hidden_dim': 128, 'num_layers': 2},
            },
            'wells': {
                'num_producers': 10,
                'num_injectors': 5,
                'producer_node_dim': 10,
                'injector_node_dim': 8,
            },
            'edge_features': {'dimension': 10},
            'data': {'scenarios_dir': 'results/training_data'},
        }

    # Override with command line args
    if args.exp_name:
        config['experiment']['name'] = args.exp_name
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size

    # Set seed
    seed = config['experiment']['seed'] + rank  # Different seed per rank
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Device
    device = torch.device('cpu')
    if is_main_process:
        print(f"\nRank {rank}/{world_size}")
        print(f"Using device: {device}")
        print(f"Distributed training with {world_size} nodes\n")

    # Create dataloaders
    if is_main_process:
        print("Creating distributed dataloaders...")

    train_loader, val_loader, test_loader, graph_data, train_sampler = create_distributed_dataloaders(
        preprocessed_dir='ml/data/preprocessed',
        batch_size=config['training']['batch_size'],
        num_workers=0,
        rank=rank,
        world_size=world_size,
    )

    # Initialize model
    if is_main_process:
        print("Initializing model...")

    model = GNN_LSTM_Surrogate(
        num_producers=config['wells']['num_producers'],
        producer_node_dim=config['wells']['producer_node_dim'],
        injector_node_dim=config['wells']['injector_node_dim'],
        edge_dim=config['edge_features']['dimension'],
        gnn_hidden_dim=config['model']['gnn']['hidden_dim'],
        gnn_num_layers=config['model']['gnn']['num_layers'],
        lstm_hidden_dim=config['model']['lstm']['hidden_dim'],
        lstm_num_layers=config['model']['lstm']['num_layers'],
    ).to(device)

    # Wrap model with DDP
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=None,  # CPU training
    )

    if is_main_process:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {num_params:,}\n")

    # Initialize optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )

    # Train (only main process saves checkpoints)
    save_dir = Path('results/ml_experiments') / config['experiment']['name']
    if is_main_process:
        print(f"Saving results to: {save_dir}\n")

    trainer.fit(
        num_epochs=config['training']['num_epochs'],
        save_dir=save_dir if is_main_process else None
    )

    if is_main_process:
        print(f"\n✓ Distributed training complete!")
        print(f"  Best model saved to: {save_dir / 'best_model.pth'}")
        print(f"  Best validation loss: {trainer.best_val_loss:.4f}")

    # Cleanup
    cleanup_distributed()


if __name__ == '__main__':
    main()
