"""
Evaluation script for GNN-LSTM reservoir surrogate model.

Computes MAPE, R², and other metrics on test set.
Generates prediction vs. truth plots for visualization.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
from sklearn.metrics import r2_score, mean_absolute_percentage_error

from ml.models.surrogate import GNN_LSTM_Surrogate
from ml.data.simple_dataset import SimpleReservoirDataset, collate_batch
from torch.utils.data import DataLoader
import yaml


def load_config(config_path='config.yaml'):
    """Load configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_model(checkpoint_path, config, device):
    """Load trained model from checkpoint."""
    # Initialize model with hardcoded params (config structure varies)
    model = GNN_LSTM_Surrogate(
        num_producers=10,
        producer_node_dim=10,
        injector_node_dim=8,
        edge_dim=10,
        gnn_hidden_dim=128,
        gnn_num_layers=3,
        lstm_hidden_dim=256,
        lstm_num_layers=2,
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✓ Model loaded from: {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Val Loss: {checkpoint.get('val_loss', 'N/A'):.2e}")

    return model


def compute_metrics(predictions, targets):
    """
    Compute evaluation metrics.

    Args:
        predictions: (N, T, num_wells) array
        targets: (N, T, num_wells) array

    Returns:
        Dict of metrics
    """
    # Flatten for metric computation
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()

    # Remove zeros to avoid division issues in MAPE
    nonzero_mask = np.abs(target_flat) > 1e-6
    pred_nonzero = pred_flat[nonzero_mask]
    target_nonzero = target_flat[nonzero_mask]

    # Compute metrics
    if len(target_nonzero) > 0:
        mape = mean_absolute_percentage_error(target_nonzero, pred_nonzero) * 100
    else:
        mape = 0.0
    r2 = r2_score(target_flat, pred_flat)
    mae = np.mean(np.abs(pred_flat - target_flat))
    rmse = np.sqrt(np.mean((pred_flat - target_flat) ** 2))

    # Per-well metrics
    num_wells = predictions.shape[2]
    per_well_mape = []
    per_well_r2 = []

    for well_idx in range(num_wells):
        well_pred = predictions[:, :, well_idx].flatten()
        well_target = targets[:, :, well_idx].flatten()

        nonzero_mask = np.abs(well_target) > 1e-6
        if nonzero_mask.sum() > 0:
            well_mape = mean_absolute_percentage_error(
                well_target[nonzero_mask],
                well_pred[nonzero_mask]
            ) * 100
            well_r2 = r2_score(well_target, well_pred)
        else:
            well_mape = 0.0
            well_r2 = 1.0

        per_well_mape.append(well_mape)
        per_well_r2.append(well_r2)

    return {
        'mape': mape,
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'per_well_mape': per_well_mape,
        'per_well_r2': per_well_r2,
    }


def evaluate_model(model, test_loader, device, denorm_stats=None):
    """
    Evaluate model on test set.

    Args:
        model: Trained GNN-LSTM model
        test_loader: DataLoader for test set
        device: torch device
        denorm_stats: Normalization stats for denormalizing predictions

    Returns:
        predictions, targets, metrics
    """
    model.eval()

    all_pred_oil = []
    all_pred_water = []
    all_target_oil = []
    all_target_water = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Move to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            # Forward pass
            output = model(batch)

            # Extract predictions
            pred_oil = output['oil_rates'].cpu().numpy()  # (batch, T, num_producers)
            pred_water = output['water_rates'].cpu().numpy()
            target_oil = batch['target_oil_rate'].cpu().numpy()
            target_water = batch['target_water_rate'].cpu().numpy()

            # Denormalize if stats provided
            if denorm_stats is not None:
                pred_oil = pred_oil * denorm_stats['oil_rate_std'] + denorm_stats['oil_rate_mean']
                pred_water = pred_water * denorm_stats['water_rate_std'] + denorm_stats['water_rate_mean']
                target_oil = target_oil * denorm_stats['oil_rate_std'] + denorm_stats['oil_rate_mean']
                target_water = target_water * denorm_stats['water_rate_std'] + denorm_stats['water_rate_mean']

            all_pred_oil.append(pred_oil)
            all_pred_water.append(pred_water)
            all_target_oil.append(target_oil)
            all_target_water.append(target_water)

    # Concatenate all batches
    pred_oil = np.concatenate(all_pred_oil, axis=0)
    pred_water = np.concatenate(all_pred_water, axis=0)
    target_oil = np.concatenate(all_target_oil, axis=0)
    target_water = np.concatenate(all_target_water, axis=0)

    # Compute metrics
    oil_metrics = compute_metrics(pred_oil, target_oil)
    water_metrics = compute_metrics(pred_water, target_water)

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"\nOil Rate Predictions:")
    print(f"  MAPE: {oil_metrics['mape']:.2f}%")
    print(f"  R²: {oil_metrics['r2']:.4f}")
    print(f"  MAE: {oil_metrics['mae']:.2f} STB/day")
    print(f"  RMSE: {oil_metrics['rmse']:.2f} STB/day")

    print(f"\nWater Rate Predictions:")
    print(f"  MAPE: {water_metrics['mape']:.2f}%")
    print(f"  R²: {water_metrics['r2']:.4f}")
    print(f"  MAE: {water_metrics['mae']:.2f} STB/day")
    print(f"  RMSE: {water_metrics['rmse']:.2f} STB/day")

    print("\n" + "=" * 80)

    return {
        'predictions': {'oil': pred_oil, 'water': pred_water},
        'targets': {'oil': target_oil, 'water': target_water},
        'metrics': {'oil': oil_metrics, 'water': water_metrics}
    }


def plot_predictions(results, save_dir):
    """
    Generate prediction vs. truth plots.

    Args:
        results: Dict from evaluate_model()
        save_dir: Directory to save plots
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    pred_oil = results['predictions']['oil']
    pred_water = results['predictions']['water']
    target_oil = results['targets']['oil']
    target_water = results['targets']['water']

    # 1. Scatter plot: Predicted vs. Actual (all wells, all timesteps)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Oil rates
    ax1.scatter(target_oil.flatten(), pred_oil.flatten(), alpha=0.3, s=1)
    ax1.plot([-5000, -1], [-5000, -1], 'r--', label='Perfect prediction')
    ax1.set_xlabel('Actual Oil Rate (STB/day)')
    ax1.set_ylabel('Predicted Oil Rate (STB/day)')
    ax1.set_title('Oil Rate: Predicted vs. Actual')
    ax1.set_xlim(-5000, -1)
    ax1.set_ylim(-5000, -1)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Water rates
    ax2.scatter(target_water.flatten(), pred_water.flatten(), alpha=0.3, s=1)
    ax2.plot([-10, -1], [-10, -1], 'r--', label='Perfect prediction')
    ax2.set_xlabel('Actual Water Rate (STB/day)')
    ax2.set_ylabel('Predicted Water Rate (STB/day)')
    ax2.set_title('Water Rate: Predicted vs. Actual')
    ax2.set_xlim(-10, -1)
    ax2.set_ylim(-10, -1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'scatter_predictions.png', dpi=150)
    print(f"✓ Saved: {save_dir / 'scatter_predictions.png'}")
    plt.close()

    # 2. Time series for a sample scenario
    scenario_idx = 0
    num_wells = min(5, pred_oil.shape[2])  # Plot first 5 wells

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Oil rates over time
    for well_idx in range(num_wells):
        axes[0].plot(target_oil[scenario_idx, :, well_idx],
                     label=f'Well {well_idx+1} (Actual)', linestyle='--', alpha=0.7)
        axes[0].plot(pred_oil[scenario_idx, :, well_idx],
                     label=f'Well {well_idx+1} (Predicted)', alpha=0.9)

    axes[0].set_xlabel('Timestep')
    axes[0].set_ylabel('Oil Rate (STB/day)')
    axes[0].set_title(f'Oil Rates Over Time - Scenario {scenario_idx}')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Water rates over time
    for well_idx in range(num_wells):
        axes[1].plot(target_water[scenario_idx, :, well_idx],
                     label=f'Well {well_idx+1} (Actual)', linestyle='--', alpha=0.7)
        axes[1].plot(pred_water[scenario_idx, :, well_idx],
                     label=f'Well {well_idx+1} (Predicted)', alpha=0.9)

    axes[1].set_xlabel('Timestep')
    axes[1].set_ylabel('Water Rate (STB/day)')
    axes[1].set_title(f'Water Rates Over Time - Scenario {scenario_idx}')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'timeseries_sample.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_dir / 'timeseries_sample.png'}")
    plt.close()

    # 3. Per-well performance
    oil_metrics = results['metrics']['oil']
    water_metrics = results['metrics']['water']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    wells = list(range(len(oil_metrics['per_well_mape'])))

    ax1.bar(wells, oil_metrics['per_well_mape'], alpha=0.7, label='Oil MAPE')
    ax1.set_xlabel('Well Index')
    ax1.set_ylabel('MAPE (%)')
    ax1.set_title('Per-Well Oil Rate MAPE')
    ax1.grid(True, alpha=0.3)

    ax2.bar(wells, water_metrics['per_well_mape'], alpha=0.7, label='Water MAPE', color='orange')
    ax2.set_xlabel('Well Index')
    ax2.set_ylabel('MAPE (%)')
    ax2.set_title('Per-Well Water Rate MAPE')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'per_well_performance.png', dpi=150)
    print(f"✓ Saved: {save_dir / 'per_well_performance.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate GNN-LSTM model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='results/evaluation',
                        help='Directory to save evaluation results')
    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load config
    config = load_config(args.config)

    # Load graph data
    graph_data = np.load('ml/data/preprocessed/graph_data.npz')

    # Get scenarios from scenario_list.txt
    with open('ml/data/preprocessed/scenario_list.txt') as f:
        scenario_paths = [line.strip() for line in f.readlines()]

    num_scenarios = len(scenario_paths)
    num_train = int(0.7 * num_scenarios)
    num_val = int(0.15 * num_scenarios)

    train_scenarios = scenario_paths[:num_train]
    test_scenarios = scenario_paths[num_train+num_val:]

    print(f"\nFound {num_scenarios} total scenarios")
    print(f"Test set: {len(test_scenarios)} scenarios")

    # Create datasets with normalization (model was trained with normalized data)
    # Get normalization stats from training set
    temp_train_dataset = SimpleReservoirDataset(train_scenarios, graph_data, normalize=True)
    norm_stats = temp_train_dataset.normalization_stats

    # Create test dataset with same normalization
    test_dataset = SimpleReservoirDataset(test_scenarios, graph_data, normalize=True, normalization_stats=norm_stats)
    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
    )

    print(f"\nUsing normalization stats from training set:")
    print(f"  Oil rate: {norm_stats['oil_rate_mean']:.1f} ± {norm_stats['oil_rate_std']:.1f}")
    print(f"  Water rate: {norm_stats['water_rate_mean']:.1f} ± {norm_stats['water_rate_std']:.1f}")

    # Load model
    model = load_model(args.checkpoint, config, device)

    # Evaluate with denormalization stats
    results = evaluate_model(model, test_loader, device, denorm_stats=norm_stats)

    # Save metrics
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        # Convert numpy to python types for JSON
        def convert_value(v):
            if isinstance(v, (list, np.ndarray)):
                return [float(x) for x in v]
            else:
                return float(v)

        metrics_json = {
            'oil': {k: convert_value(v) for k, v in results['metrics']['oil'].items()},
            'water': {k: convert_value(v) for k, v in results['metrics']['water'].items()},
        }
        json.dump(metrics_json, f, indent=2)

    print(f"\n✓ Metrics saved to: {metrics_path}")

    # Generate plots
    plot_predictions(results, output_dir)

    print(f"\n✓ Evaluation complete! Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
