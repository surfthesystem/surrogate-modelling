"""
Trainer class for GNN-LSTM surrogate model.

Handles training loop, validation, checkpointing, and early stopping.
"""

import torch
from pathlib import Path
from tqdm import tqdm


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, name):
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


class EarlyStopping:
    """Early stopping to stop training when validation loss plateaus."""

    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return False


def save_checkpoint(model, optimizer, epoch, val_loss, path, scheduler=None):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    torch.save(checkpoint, path)


def get_lr(optimizer):
    """Get current learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']


class Trainer:
    """Trainer for GNN-LSTM surrogate model."""

    def __init__(self, model, optimizer, train_loader, val_loader, config, device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Initialize tracking
        self.early_stopping = EarlyStopping(
            patience=config.get('training', {}).get('early_stopping', {}).get('patience', 10),
            min_delta=config.get('training', {}).get('early_stopping', {}).get('min_delta', 1e-4),
        )

        # Learning rate scheduler
        lr_scheduler_config = config.get('training', {}).get('lr_scheduler', {})
        if lr_scheduler_config.get('type') == 'StepLR':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=lr_scheduler_config.get('step_size', 30),
                gamma=lr_scheduler_config.get('gamma', 0.5),
            )
        elif lr_scheduler_config.get('type') == 'ReduceLROnPlateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=lr_scheduler_config.get('factor', 0.5),
                patience=lr_scheduler_config.get('patience', 5),
            )
        else:
            self.scheduler = None

        self.best_val_loss = float('inf')
        self.epoch = 0

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        loss_meter = AverageMeter('Loss')

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch}')
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            if isinstance(batch, dict):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

            # Forward pass
            self.optimizer.zero_grad()

            try:
                # Unpack batch and run model
                predictions = self.model(batch)

                # Compute loss (simple MSE for now)
                if isinstance(predictions, dict):
                    # If predictions is a dict with 'oil_rates' and 'water_rates'
                    target_oil = batch.get('target_oil_rate')
                    target_water = batch.get('target_water_rate')

                    # Model outputs (batch, T, num_prod), targets are (batch, T, num_prod)
                    loss_oil = torch.nn.functional.mse_loss(
                        predictions['oil_rates'], target_oil
                    )
                    loss_water = torch.nn.functional.mse_loss(
                        predictions['water_rates'], target_water
                    )

                    # Weighted loss (oil is more important)
                    beta = self.config.get('loss', {}).get('beta', 80)
                    alpha = self.config.get('loss', {}).get('alpha', 1)
                    loss = beta * loss_oil + alpha * loss_water
                else:
                    # Simple MSE loss
                    target = batch['target']
                    loss = torch.nn.functional.mse_loss(predictions, target)

                # Backward pass
                loss.backward()

                # Gradient clipping
                grad_clip = self.config.get('training', {}).get('gradient_clip', 1.0)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        grad_clip
                    )

                self.optimizer.step()
                loss_meter.update(loss.item())

                # Update progress bar
                pbar.set_postfix({'loss': f'{loss.item():.4f}',
                                 'avg_loss': f'{loss_meter.avg:.4f}'})

            except Exception as e:
                print(f"\nError in batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

        return loss_meter.avg

    def validate(self):
        """Validate the model."""
        self.model.eval()
        loss_meter = AverageMeter('Val Loss')

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Move batch to device
                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()}

                try:
                    predictions = self.model(batch)

                    # Compute loss
                    if isinstance(predictions, dict):
                        target_oil = batch.get('target_oil_rate')
                        target_water = batch.get('target_water_rate')

                        loss_oil = torch.nn.functional.mse_loss(
                            predictions['oil_rates'], target_oil
                        )
                        loss_water = torch.nn.functional.mse_loss(
                            predictions['water_rates'], target_water
                        )

                        beta = self.config.get('loss', {}).get('beta', 80)
                        alpha = self.config.get('loss', {}).get('alpha', 1)
                        loss = beta * loss_oil + alpha * loss_water
                    else:
                        target = batch['target']
                        loss = torch.nn.functional.mse_loss(predictions, target)

                    loss_meter.update(loss.item())

                except Exception as e:
                    print(f"Error in validation: {e}")
                    continue

        return loss_meter.avg

    def fit(self, num_epochs, save_dir):
        """Train the model for num_epochs."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nStarting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Training batches: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")
        print("=" * 80)

        for epoch in range(num_epochs):
            self.epoch = epoch

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_loss = self.validate()

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  LR: {get_lr(self.optimizer):.6f}")

            # Save checkpoint every N epochs
            save_every = self.config.get('training', {}).get('save_every', 10)
            if epoch % save_every == 0:
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_loss,
                    save_dir / f'checkpoint_epoch_{epoch}.pth',
                    scheduler=self.scheduler,
                )
                print(f"  ✓ Checkpoint saved")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_loss,
                    save_dir / 'best_model.pth',
                    scheduler=self.scheduler,
                )
                print(f"  ✓ Best model saved (val_loss: {val_loss:.4f})")

            # Early stopping
            if self.early_stopping(val_loss):
                print(f"\n✓ Early stopping at epoch {epoch}")
                print(f"   Best val loss: {self.best_val_loss:.4f}")
                break

        print(f"\n{'=' * 80}")
        print(f"✓ Training complete! Best val loss: {self.best_val_loss:.4f}")
        print(f"{'=' * 80}")
