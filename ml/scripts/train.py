#!/usr/bin/env python
"""
Main training script for GNN-LSTM surrogate model.

Usage:
    python ml/scripts/train.py --config ml/training/config.yaml --exp_name my_experiment
"""

import argparse
import yaml
import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.models.surrogate import GNN_LSTM_Surrogate
from ml.training.trainer import Trainer
from ml.data.simple_dataset import SimpleReservoirDataset, collate_batch


def create_dataloaders(preprocessed_dir, data_dir, batch_size=8, train_split=0.7, val_split=0.15, num_workers=4):
    """
    Create dataloaders using the real SimpleReservoirDataset.

    Args:
        preprocessed_dir: Directory with graph_data.npz and scenario_list.txt
        data_dir: Directory with simulation results (not used, paths in scenario_list)
        batch_size: Batch size for training
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        num_workers: Number of dataloader workers

    Returns:
        train_loader, val_loader, test_loader, graph_data
    """
    from torch.utils.data import DataLoader

    # Load preprocessed graph data
    graph_path = Path(preprocessed_dir) / 'graph_data.npz'
    graph_data = np.load(graph_path)

    # Load scenario list
    scenario_list_path = Path(preprocessed_dir) / 'scenario_list.txt'
    with open(scenario_list_path) as f:
        scenario_paths = [line.strip() for line in f.readlines()]

    print(f"Found {len(scenario_paths)} scenarios")

    # Split into train/val/test
    num_scenarios = len(scenario_paths)
    num_train = int(num_scenarios * train_split)
    num_val = int(num_scenarios * val_split)

    train_scenarios = scenario_paths[:num_train]
    val_scenarios = scenario_paths[num_train:num_train+num_val]
    test_scenarios = scenario_paths[num_train+num_val:]

    print(f"Split: {len(train_scenarios)} train, {len(val_scenarios)} val, {len(test_scenarios)} test")

    # Create datasets
    train_dataset = SimpleReservoirDataset(train_scenarios, graph_data)
    val_dataset = SimpleReservoirDataset(val_scenarios, graph_data)
    test_dataset = SimpleReservoirDataset(test_scenarios, graph_data)

    # Create dataloaders with custom collate
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_batch,
        pin_memory=False,  # CPU training
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
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

    return train_loader, val_loader, test_loader, graph_data


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def print_model_summary(model):
    """Print model architecture and parameter count."""
    print("\n" + "=" * 80)
    print("Model Architecture:")
    print("=" * 80)
    print(model)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {num_params:,}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Train GNN-LSTM surrogate model')
    parser.add_argument('--config', default='ml/training/config.yaml', help='Path to config file')
    parser.add_argument('--exp_name', default='gnn_lstm_baseline', help='Experiment name')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (overrides config)')
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        print(f"Config file not found: {config_path}")
        print("Using default configuration")
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

    # Override config with command line args
    if args.exp_name:
        config['experiment']['name'] = args.exp_name
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size

    # Set random seed
    set_seed(config['experiment']['seed'])

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if device.type == 'cpu':
        print("NOTE: Training on CPU. This will be slower than GPU training.")
        print("      With 20+ nodes available, consider distributed training for speedup.\n")

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader, graph_data = create_dataloaders(
        preprocessed_dir='ml/data/preprocessed',
        data_dir=config['data']['scenarios_dir'],
        batch_size=config['training']['batch_size'],
        num_workers=0,  # Use 0 for single-node, increase for multi-node
    )

    # Initialize model
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

    # Print model summary
    print_model_summary(model)

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

    # Train
    save_dir = Path('results/ml_experiments') / config['experiment']['name']
    print(f"\nSaving results to: {save_dir}")

    trainer.fit(
        num_epochs=config['training']['num_epochs'],
        save_dir=save_dir
    )

    print(f"\n✓ Training complete!")
    print(f"  Best model saved to: {save_dir / 'best_model.pth'}")
    print(f"  Best validation loss: {trainer.best_val_loss:.4f}")


if __name__ == '__main__':
    main()
