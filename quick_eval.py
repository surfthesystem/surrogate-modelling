"""Quick evaluation script for trained model."""

import sys
sys.path.insert(0, '/mnt/disks/mydata/surrogate-modelling-1')

import torch
import numpy as np
from pathlib import Path
from ml.models.surrogate import GNN_LSTM_Surrogate
from ml.data.simple_dataset import SimpleReservoirDataset, collate_batch
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_absolute_percentage_error

# Load model
checkpoint_path = 'results/ml_experiments/normalized_run/best_model.pth'
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Initialize model with hardcoded params
model = GNN_LSTM_Surrogate(
    num_producers=10,
    producer_node_dim=10,
    injector_node_dim=8,
    edge_dim=10,
    gnn_hidden_dim=128,
    gnn_num_layers=3,
    lstm_hidden_dim=256,
    lstm_num_layers=2,
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✓ Model loaded from: {checkpoint_path}")
print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
print(f"  Val Loss: {checkpoint.get('val_loss', 'N/A'):.2f}")

# Load data
graph_data = np.load('ml/data/preprocessed/graph_data.npz')

# Load scenario list
with open('ml/data/preprocessed/scenario_list.txt') as f:
    scenario_paths = [line.strip() for line in f.readlines()]

num_scenarios = len(scenario_paths)
num_train = int(0.7 * num_scenarios)
num_val = int(0.15 * num_scenarios)
test_scenarios = scenario_paths[num_train+num_val:]

print(f"\nFound {num_scenarios} total scenarios")
print(f"Test set: {len(test_scenarios)} scenarios")

# Create test dataset WITH normalization (model was trained with normalized data)
# We need to use the same normalization stats as training
# For now, create a temporary training dataset to get stats
temp_train_scenarios = scenario_paths[:num_train]
temp_train_dataset = SimpleReservoirDataset(temp_train_scenarios, graph_data, normalize=True)
norm_stats = temp_train_dataset.normalization_stats

# Now create test dataset with same normalization
test_dataset = SimpleReservoirDataset(test_scenarios, graph_data, normalize=True, normalization_stats=norm_stats)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_batch)

print(f"\nUsing normalization stats from training set:")
print(f"  Oil rate: {norm_stats['oil_rate_mean']:.1f} ± {norm_stats['oil_rate_std']:.1f}")
print(f"  Water rate: {norm_stats['water_rate_mean']:.1f} ± {norm_stats['water_rate_std']:.1f}")

# Evaluate
all_pred_oil = []
all_pred_water = []
all_target_oil = []
all_target_water = []

print("\nRunning inference...")
with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        output = model(batch)

        pred_oil = output['oil_rates'].cpu().numpy()
        pred_water = output['water_rates'].cpu().numpy()
        target_oil = batch['target_oil_rate'].cpu().numpy()
        target_water = batch['target_water_rate'].cpu().numpy()

        # Denormalize predictions and targets back to original scale
        pred_oil = pred_oil * norm_stats['oil_rate_std'] + norm_stats['oil_rate_mean']
        pred_water = pred_water * norm_stats['water_rate_std'] + norm_stats['water_rate_mean']
        target_oil = target_oil * norm_stats['oil_rate_std'] + norm_stats['oil_rate_mean']
        target_water = target_water * norm_stats['water_rate_std'] + norm_stats['water_rate_mean']

        all_pred_oil.append(pred_oil)
        all_pred_water.append(pred_water)
        all_target_oil.append(target_oil)
        all_target_water.append(target_water)

# Concatenate
pred_oil = np.concatenate(all_pred_oil, axis=0)
pred_water = np.concatenate(all_pred_water, axis=0)
target_oil = np.concatenate(all_target_oil, axis=0)
target_water = np.concatenate(all_target_water, axis=0)

# Compute metrics
def compute_metrics(pred, target, name):
    pred_flat = pred.flatten()
    target_flat = target.flatten()

    # Remove very small values for MAPE
    nonzero_mask = target_flat > 1e-6
    if nonzero_mask.sum() > 0:
        mape = mean_absolute_percentage_error(target_flat[nonzero_mask], pred_flat[nonzero_mask]) * 100
    else:
        mape = 0.0

    r2 = r2_score(target_flat, pred_flat)
    mae = np.mean(np.abs(pred_flat - target_flat))
    rmse = np.sqrt(np.mean((pred_flat - target_flat) ** 2))

    print(f"\n{name}:")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R²: {r2:.4f}")
    print(f"  MAE: {mae:.2f} STB/day")
    print(f"  RMSE: {rmse:.2f} STB/day")

    return {'mape': mape, 'r2': r2, 'mae': mae, 'rmse': rmse}

print("\n" + "=" * 80)
print("EVALUATION RESULTS")
print("=" * 80)

oil_metrics = compute_metrics(pred_oil, target_oil, "Oil Rate Predictions")
water_metrics = compute_metrics(pred_water, target_water, "Water Rate Predictions")

print("\n" + "=" * 80)
print("✓ Evaluation complete!")
print("=" * 80)
