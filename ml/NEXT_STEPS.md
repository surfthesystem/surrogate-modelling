# GNN-LSTM Surrogate Model - Next Steps

**Current Status**: Core implementation complete (85%), testing in progress (56% modules tested)

---

## Quick Status Check

Check if PyTorch Geometric finished installing:

```bash
cd /mnt/disks/mydata/surrogate-modelling-1

# Check installation status
ml_venv/bin/python -c "import torch_geometric; print('✓ PyTorch Geometric', torch_geometric.__version__)"
```

**If it prints the version**: Installation complete! Proceed to Step 1 below.

**If it says "ModuleNotFoundError"**: Installation still running in background. Check status:

```bash
# Check if pip is still running
ps aux | grep "pip install"

# Or wait and retry
sleep 60 && ml_venv/bin/python -c "import torch_geometric; print('✓ Ready!')"
```

---

## Step 1: Complete Module Testing (30 minutes)

Once PyTorch Geometric is installed, test the remaining 4 modules:

```bash
cd /mnt/disks/mydata/surrogate-modelling-1

# Test GNN module
ml_venv/bin/python ml/models/gnn.py
# Expected: "✓ GNN tests passed!"

# Test preprocessing module
ml_venv/bin/python ml/data/preprocessing.py
# Expected: Edge feature computation tests

# Test full surrogate model
ml_venv/bin/python ml/models/surrogate.py
# Expected: "✓ Surrogate model tests passed!"

# Test dataset (may need scenario files)
ml_venv/bin/python -c "from ml.data.dataset import ReservoirDataset; print('✓ Dataset import successful')"
```

**Document results** in `ml/TESTING_STATUS.md` (update the ⏳ PENDING sections).

---

## Step 2: Write Preprocessing Script (2 hours)

Create `ml/scripts/preprocess_all.py` to automate data preparation:

### Purpose
- Load reservoir properties (permeability, porosity) from `data/`
- Load well locations from `data/impes_input/selected_wells.csv`
- Build well connectivity graphs (Voronoi + bipartite)
- Compute static edge features (distance, permeability, transmissibility, direction)
- Compute time-lagged correlations from all 100 simulation scenarios
- Save preprocessed data to `ml/data/preprocessed/`

### Template Structure
```python
#!/usr/bin/env python
"""
Preprocess all simulation data for GNN-LSTM training.

Usage:
    python ml/scripts/preprocess_all.py --data_dir results/training_data --output_dir ml/data/preprocessed
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from ml.data.graph_builder import build_well_graphs
from ml.data.preprocessing import compute_static_edge_features, compute_time_lagged_correlation
from ml.data.normalizers import create_normalizers

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='results/training_data')
    parser.add_argument('--output_dir', default='ml/data/preprocessed')
    parser.add_argument('--num_scenarios', type=int, default=100)
    args = parser.parse_args()

    # 1. Load well locations
    wells_df = pd.read_csv('data/impes_input/selected_wells.csv')
    prod_coords = wells_df[wells_df['type']=='producer'][['x_m', 'y_m']].values * 3.28084  # m to ft
    inj_coords = wells_df[wells_df['type']=='injector'][['x_m', 'y_m']].values * 3.28084

    # 2. Build graphs
    graph_data = build_well_graphs(prod_coords, inj_coords, p2p_mode='voronoi', i2p_mode='full')

    # 3. Load permeability/porosity fields
    perm_field = np.load('data/permeability_field.npy')
    poro_field = np.load('data/porosity_field.npy')

    # 4. Compute static edge features
    static_features = compute_static_edge_features(...)

    # 5. Compute time-lagged correlations from all scenarios
    scenario_paths = sorted(Path(args.data_dir).glob('doe_*/doe_*.npz'))[:args.num_scenarios]
    time_lag_corr = compute_time_lagged_correlation(scenario_paths, graph_data['edge_index_i2p'])

    # 6. Save preprocessed data
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path / 'graph_data.npz',
        edge_index_p2p=graph_data['edge_index_p2p'],
        edge_index_i2p=graph_data['edge_index_i2p'],
        static_features_p2p=...,
        static_features_i2p=...,
        time_lag_corr=time_lag_corr,
        producer_coords=prod_coords,
        injector_coords=inj_coords,
    )

    print(f"✓ Preprocessed data saved to {output_path}")

if __name__ == '__main__':
    main()
```

### Run
```bash
ml_venv/bin/python ml/scripts/preprocess_all.py \
    --data_dir results/training_data \
    --output_dir ml/data/preprocessed \
    --num_scenarios 100
```

---

## Step 3: Write Trainer Module (3 hours)

Create `ml/training/trainer.py` with full training loop:

### Purpose
- Training loop: forward, backward, optimizer step
- Validation loop: compute metrics without gradients
- Metric tracking: loss, MAPE, R²
- Checkpointing: save every N epochs + best model
- Early stopping: stop when validation loss plateaus

### Template Structure
```python
"""
Trainer class for GNN-LSTM surrogate model.
"""

import torch
from pathlib import Path
from ml.utils.helpers import AverageMeter, EarlyStopping, save_checkpoint, get_lr
from ml.models.losses import combined_loss

class Trainer:
    def __init__(self, model, optimizer, train_loader, val_loader, config, device):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Initialize tracking
        self.early_stopping = EarlyStopping(
            patience=config['training']['early_stopping']['patience'],
            min_delta=config['training']['early_stopping']['min_delta'],
        )

        # Learning rate scheduler
        if config['training']['lr_scheduler']['type'] == 'StepLR':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config['training']['lr_scheduler']['step_size'],
                gamma=config['training']['lr_scheduler']['gamma'],
            )

        self.best_val_loss = float('inf')
        self.epoch = 0

    def train_epoch(self):
        self.model.train()
        loss_meter = AverageMeter('Loss')

        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(batch)

            # Compute loss
            loss, loss_dict = combined_loss(
                predictions, batch,
                alpha=self.config['loss']['alpha'],
                beta=self.config['loss']['beta'],
                use_cumulative=self.config['loss']['use_cumulative'],
                cumulative_weight=self.config['loss']['cumulative_weight'],
                use_physics=self.config['loss']['use_physics'],
                physics_weight=self.config['loss']['physics_weight'],
                loss_type=self.config['loss']['type'],
            )

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config['training']['gradient_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip']
                )

            self.optimizer.step()
            loss_meter.update(loss.item())

            # Log progress
            if batch_idx % self.config['logging']['log_interval'] == 0:
                print(f"Epoch {self.epoch} [{batch_idx}/{len(self.train_loader)}] "
                      f"Loss: {loss.item():.4f} (avg: {loss_meter.avg:.4f})")

        return loss_meter.avg

    def validate(self):
        self.model.eval()
        loss_meter = AverageMeter('Val Loss')

        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                predictions = self.model(batch)
                loss, _ = combined_loss(predictions, batch)
                loss_meter.update(loss.item())

        return loss_meter.avg

    def fit(self, num_epochs, save_dir):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(num_epochs):
            self.epoch = epoch

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_loss = self.validate()

            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()

            # Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  LR: {get_lr(self.optimizer):.6f}")

            # Save checkpoint
            if epoch % self.config['training']['save_every'] == 0:
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_loss,
                    save_dir / f'checkpoint_epoch_{epoch}.pth',
                    scheduler=self.scheduler,
                )

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_loss,
                    save_dir / 'best_model.pth',
                )
                print(f"  ✓ Best model saved (val_loss: {val_loss:.4f})")

            # Early stopping
            if self.early_stopping(val_loss):
                print(f"\n✓ Early stopping at epoch {epoch}")
                break

        print(f"\n✓ Training complete! Best val loss: {self.best_val_loss:.4f}")
```

---

## Step 4: Write Training Script (1 hour)

Create `ml/scripts/train.py` as the main entry point:

```python
#!/usr/bin/env python
"""
Main training script for GNN-LSTM surrogate model.

Usage:
    python ml/scripts/train.py --config ml/training/config.yaml --exp_name my_experiment
"""

import argparse
import yaml
import torch
from pathlib import Path
from ml.models.surrogate import GNN_LSTM_Surrogate
from ml.data.dataset import create_dataloaders
from ml.training.trainer import Trainer
from ml.utils.helpers import set_seed, get_device, print_model_summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='ml/training/config.yaml')
    parser.add_argument('--exp_name', default=None)
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Set experiment name
    if args.exp_name:
        config['experiment']['name'] = args.exp_name

    # Set random seed
    set_seed(config['experiment']['seed'])

    # Get device
    device = get_device()

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=config['data']['scenarios_dir'],
        preprocessed_dir='ml/data/preprocessed',
        batch_size=config['training']['batch_size'],
        num_workers=config['hardware']['num_workers'],
    )

    # Initialize model
    model = GNN_LSTM_Surrogate(
        num_producers=config['wells']['num_producers'],
        num_injectors=config['wells']['num_injectors'],
        producer_node_dim=config['wells']['producer_node_dim'],
        injector_node_dim=config['wells']['injector_node_dim'],
        edge_feature_dim=config['edge_features']['dimension'],
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
    trainer = Trainer(model, optimizer, train_loader, val_loader, config, device)

    # Train
    save_dir = Path('results/ml_experiments') / config['experiment']['name']
    trainer.fit(num_epochs=config['training']['num_epochs'], save_dir=save_dir)

if __name__ == '__main__':
    main()
```

### Run Training
```bash
ml_venv/bin/python ml/scripts/train.py \
    --config ml/training/config.yaml \
    --exp_name gnn_lstm_baseline
```

---

## Step 5: Monitor Training (6 hours training time)

Training on 100 scenarios for 150 epochs should take ~5-7 hours on GPU (RTX 2060 or similar).

### Monitor Progress
```bash
# Watch training output
tail -f results/ml_experiments/gnn_lstm_baseline/train.log

# Check disk space periodically (checkpoints can be large)
df -h /mnt/disks/mydata
```

### Expected Output
```
Epoch 0 Summary:
  Train Loss: 1250.3456
  Val Loss: 1180.2341
  LR: 0.000100

Epoch 10 Summary:
  Train Loss: 650.1234
  Val Loss: 680.5432
  LR: 0.000100
  ✓ Best model saved (val_loss: 680.5432)

...

Epoch 150 Summary:
  Train Loss: 85.2341
  Val Loss: 92.1234
  LR: 0.000051

✓ Training complete! Best val loss: 88.4567
```

---

## Step 6: Evaluate Trained Model (1 hour)

Once training completes, evaluate on the test set.

Create `ml/scripts/evaluate.py` (later) to:
- Load best model checkpoint
- Run inference on 15 test scenarios
- Compute MAPE, R², cumulative error
- Generate comparison plots (predicted vs. true rates)
- Per-well and per-timestep error analysis

---

## Quick Reference Commands

```bash
# Working directory
cd /mnt/disks/mydata/surrogate-modelling-1

# Activate virtual environment (optional, can use direct path)
source ml_venv/bin/activate

# Test all modules
ml_venv/bin/python ml/models/gnn.py
ml_venv/bin/python ml/models/lstm.py
ml_venv/bin/python ml/models/surrogate.py
ml_venv/bin/python ml/models/losses.py

# Preprocess data
ml_venv/bin/python ml/scripts/preprocess_all.py

# Train model
ml_venv/bin/python ml/scripts/train.py --exp_name my_experiment

# Monitor GPU (if available)
watch -n 1 nvidia-smi

# Check disk space
df -h
```

---

## Troubleshooting

### PyTorch Geometric still installing?

Kill the background process and try pre-built wheels:

```bash
# Kill pip process
pkill -f "pip install"

# Try pre-built wheels (faster, no compilation)
ml_venv/bin/pip install torch-geometric torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.9.0+cu128.html
```

### Out of memory during training?

Reduce batch size in `ml/training/config.yaml`:
```yaml
training:
  batch_size: 4  # Instead of 8
```

### Slow training?

Check if using CPU instead of GPU:
```bash
ml_venv/bin/python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

---

## Success Criteria

You're ready to move to the next phase when:

- ✅ All 9 core modules tested successfully
- ✅ Preprocessing script runs without errors
- ✅ Training script completes 10 epochs
- ✅ Validation loss decreasing
- ✅ Best model checkpoint saved

---

**Last Updated**: 2025-10-30
**See also**: `ml/IMPLEMENTATION_STATUS.md`, `ml/TESTING_STATUS.md`, `ml/README.md`
