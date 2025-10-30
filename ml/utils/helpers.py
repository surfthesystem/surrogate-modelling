"""
Utility functions for model training, checkpointing, and device management.
"""

import torch
import numpy as np
import random
import os
from pathlib import Path
from typing import Dict, Optional


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make CUDA operations deterministic (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed set to {seed}")


def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Get available device (GPU if available, else CPU).

    Args:
        prefer_gpu: Use GPU if available

    Returns:
        device: torch.device object
    """
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    return device


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """
    Count model parameters.

    Args:
        model: PyTorch model
        trainable_only: Count only trainable parameters

    Returns:
        num_params: Number of parameters
    """
    if trainable_only:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        num_params = sum(p.numel() for p in model.parameters())

    return num_params


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    save_path: str,
    **kwargs
):
    """
    Save model checkpoint.

    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch number
        loss: Current loss value
        save_path: Path to save checkpoint
        **kwargs: Additional items to save (e.g., scheduler, metrics)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }

    # Add any additional items
    checkpoint.update(kwargs)

    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = 'cpu',
) -> Dict:
    """
    Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: PyTorch model (to load state into)
        optimizer: Optional optimizer (to load state into)
        device: Device to load tensors to

    Returns:
        checkpoint: Dictionary with checkpoint data
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', float('inf'))

    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"  Epoch: {epoch}, Loss: {loss:.6f}")

    return checkpoint


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """
    Get current learning rate from optimizer.

    Args:
        optimizer: PyTorch optimizer

    Returns:
        lr: Current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def create_exp_dir(exp_name: str, base_dir: str = "results/ml_experiments") -> Path:
    """
    Create experiment directory with unique name.

    Args:
        exp_name: Experiment name
        base_dir: Base directory for experiments

    Returns:
        exp_dir: Path to experiment directory
    """
    exp_dir = Path(base_dir) / exp_name

    # Add counter if directory exists
    counter = 1
    original_exp_dir = exp_dir
    while exp_dir.exists():
        exp_dir = Path(str(original_exp_dir) + f"_{counter}")
        counter += 1

    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment directory created: {exp_dir}")

    return exp_dir


def print_model_summary(model: torch.nn.Module):
    """
    Print model architecture summary.

    Args:
        model: PyTorch model
    """
    print("\n" + "="*80)
    print("MODEL SUMMARY")
    print("="*80)

    print(f"\nModel: {model.__class__.__name__}")
    print(f"Total parameters: {count_parameters(model, trainable_only=False):,}")
    print(f"Trainable parameters: {count_parameters(model, trainable_only=True):,}")

    print("\nModel architecture:")
    print(model)

    print("="*80 + "\n")


class AverageMeter:
    """
    Computes and stores the average and current value.

    Useful for tracking metrics during training.
    """

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return f"{self.name}: {self.avg:.6f} (latest: {self.val:.6f})"


class EarlyStopping:
    """
    Early stopping handler to stop training when validation loss stops improving.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' (lower is better) or 'max' (higher is better)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current validation metric

        Returns:
            early_stop: True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"\nEarly stopping triggered after {self.counter} epochs without improvement")

        return self.early_stop


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")

    # Test set_seed
    set_seed(42)

    # Test get_device
    device = get_device()

    # Test AverageMeter
    meter = AverageMeter("Loss")
    meter.update(0.5)
    meter.update(0.3)
    meter.update(0.4)
    print(f"\nAverageMeter: {meter}")

    # Test EarlyStopping
    early_stop = EarlyStopping(patience=3)
    scores = [0.5, 0.4, 0.35, 0.34, 0.33, 0.33, 0.33, 0.33]
    for i, score in enumerate(scores):
        stop = early_stop(score)
        print(f"Epoch {i}: score={score}, stop={stop}")
        if stop:
            break

    print("\n✓ Utility tests passed!")
