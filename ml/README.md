# Enhanced GNN-LSTM Surrogate Model for Reservoir Simulation

A state-of-the-art deep learning surrogate model that combines Graph Neural Networks (GNN) with Long Short-Term Memory (LSTM) networks to predict well production rates in reservoir simulation. This implementation extends the methodology from SPE-215842 with enhanced edge features and improved architecture.

## 🎯 Project Overview

**Goal**: Build a surrogate model that predicts oil and water production rates for 10 producer wells given:
- Injection rates and bottomhole pressures (controls)
- Reservoir properties (permeability, porosity)
- Well connectivity and spatial relationships
- Temporal dynamics over 6 months (61 timesteps)

**Performance Target**:
- Oil rate MAPE < 5%
- Water rate MAPE < 6%
- 1000× speedup vs. full physics simulator

## ✨ Key Innovations Over Baseline Paper

| Feature | Baseline (SPE-215842) | Our Implementation |
|---------|----------------------|-------------------|
| **Edge Features** | 1D (distance only) | 10D (distance, perm, pressure, flow, geometry) |
| **Connectivity** | Voronoi-based | Voronoi + data-driven time-lagged correlation |
| **Dynamic Features** | Static edges | Pressure/saturation gradients per timestep |
| **Node Features** | Basic (BHP, k, φ) | Extended (+ cumulative, autoregressive, coords) |

## 📁 Project Structure

```
ml/
├── data/                          # Data preprocessing modules
│   ├── graph_builder.py          # ✅ Voronoi connectivity, k-nearest neighbors
│   ├── preprocessing.py          # ✅ 10-dim edge features (static + dynamic)
│   ├── normalizers.py            # ✅ Feature scaling (log, minmax, standard)
│   └── dataset.py                # ✅ PyTorch Dataset for 100 scenarios
│
├── models/                        # Neural network models
│   ├── gnn.py                    # ✅ EnhancedGNN with edge features
│   ├── lstm.py                   # ✅ TemporalLSTM, WellAwareLSTM, AttentionLSTM
│   ├── surrogate.py              # ✅ Full GNN_LSTM_Surrogate model
│   └── losses.py                 # ✅ Weighted L1/MSE + physics constraints
│
├── training/                      # Training infrastructure
│   ├── trainer.py                # ⏳ Training loop (to be completed)
│   ├── evaluator.py              # ⏳ Metrics and evaluation (to be completed)
│   └── config.yaml               # ✅ Hyperparameters
│
├── utils/                         # Utility functions
│   ├── helpers.py                # ✅ Checkpointing, device setup, early stopping
│   └── visualization.py          # ⏳ Plotting functions (to be completed)
│
├── scripts/                       # Standalone scripts
│   ├── preprocess_all.py         # ⏳ Preprocessing pipeline
│   ├── train.py                  # ⏳ Main training entry point
│   └── evaluate.py               # ⏳ Evaluation script
│
├── notebooks/                     # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_edge_features.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_results_analysis.ipynb
│
└── README.md                      # This file
```

**Legend**: ✅ Implemented | ⏳ To be completed | 📝 Planned

## 🏗️ Architecture Details

### 1. Data Pipeline

**Input Features** (per timestep):
- **Producer nodes** (10-dim): BHP, k, φ, depth, x, y, cum_oil, cum_water, prev_oil_rate, prev_water_rate
- **Injector nodes** (8-dim): rate, k, φ, depth, x, y, cum_inj, prev_rate
- **Edge features** (10-dim):
  ```
  [0] Inverse distance: 1/d(i,j)
  [1] Log permeability (avg): log(k_avg)
  [2] Permeability contrast: log(k_i/k_j)
  [3-4] Direction: cos(angle), sin(angle)
  [5] Transmissibility: k_avg/distance
  [6] Time-lagged correlation (data-driven)
  [7] Pressure difference: ΔP_ij(t)
  [8] Saturation difference: ΔSw_ij(t)
  [9] Drainage overlap (geometric)
  ```

**Graphs**:
- **Producer-to-Producer (P2P)**: Voronoi connectivity (~30-40 edges)
- **Injector-to-Producer (I2P)**: Full bipartite or k-nearest (50 edges)

### 2. Model Architecture

```python
GNN_LSTM_Surrogate(
    # Spatial Encoding (GNN)
    DualGraphGNN(
        gnn_p2p: EnhancedGNN(128-dim, 3 layers),
        gnn_i2p: EnhancedGNN(128-dim, 3 layers),
        fusion: FC(256 → 128)
    ),

    # Temporal Encoding (LSTM)
    WellAwareLSTM(
        input_dim: 10×128 = 1280,
        hidden_dim: 256,
        num_layers: 2
    ),

    # Rate Decoder
    RateDecoder(
        FC(256 → 64 → 20)  # 10 oil + 10 water rates
        activation: softplus (ensure positive)
    )
)
```

**Model Size**: ~2-3M parameters

### 3. Loss Function

```python
L_total = β·L_oil + α·L_water + λ_cum·L_cumulative + λ_phys·L_physics

where:
  β = 80.0 (oil is economically more important)
  α = 1.0
  λ_cum = 0.1 (cumulative production accuracy)
  λ_phys = 0.01 (physics constraints: monotonicity, smoothness)
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements-ml.txt

# Verify PyTorch Geometric installation
python -c "import torch_geometric; print(torch_geometric.__version__)"
```

### 2. Data Preprocessing

```bash
# Build well graphs and compute static edge features
python ml/scripts/preprocess_all.py \
    --data_dir results/training_data \
    --output_dir ml/data/preprocessed \
    --num_scenarios 100
```

This will:
- Load 100 simulation NPZ files
- Build Voronoi connectivity graphs
- Compute static edge features (distance, permeability, transmissibility)
- Compute time-lagged correlations from data
- Save preprocessed data for fast loading

### 3. Training

```bash
# Train with default config
python ml/scripts/train.py --config ml/training/config.yaml

# Custom experiment
python ml/scripts/train.py \
    --config ml/training/config.yaml \
    --exp_name "gnn_lstm_enhanced" \
    --num_epochs 150 \
    --batch_size 8 \
    --learning_rate 1e-4
```

Training outputs:
- Checkpoints saved to `results/ml_experiments/{exp_name}/checkpoints/`
- Logs saved to `results/ml_experiments/{exp_name}/logs/`
- Best model: `best_model.pth`

### 4. Evaluation

```bash
# Evaluate on test set
python ml/scripts/evaluate.py \
    --checkpoint results/ml_experiments/gnn_lstm_enhanced/best_model.pth \
    --data_dir results/training_data \
    --output_dir results/ml_experiments/gnn_lstm_enhanced/evaluation

# Generate visualizations
python ml/scripts/evaluate.py \
    --checkpoint best_model.pth \
    --visualize \
    --num_samples 15
```

## 📊 Expected Performance

Based on paper SPE-215842 and our enhancements:

| Metric | Target | Notes |
|--------|--------|-------|
| **Oil Rate MAPE** | < 5% | Mean Absolute Percentage Error |
| **Water Rate MAPE** | < 6% | Slightly higher due to smaller magnitudes |
| **Cumulative Oil Error** | < 3% | Total oil production accuracy |
| **R² Score** | > 0.95 | Coefficient of determination |
| **Inference Speed** | 0.05 s | vs. 50 s for simulator (1000× speedup) |

## 🔬 Implemented Modules - Detailed Description

### Data Preprocessing

**`graph_builder.py`** (✅ Complete)
- `build_well_graphs()`: Constructs P2P and I2P graphs
- `compute_voronoi_connectivity()`: Voronoi diagram-based well adjacency
- `knearest_connectivity()`: k-nearest neighbor graphs
- `visualize_graphs()`: Plot well connectivity

**`preprocessing.py`** (✅ Complete)
- `compute_static_edge_features()`: 7-dim static features (distance, perm, etc.)
- `compute_dynamic_edge_features()`: 3-dim dynamic features (ΔP, ΔSw)
- `compute_time_lagged_correlation()`: Data-driven connectivity from historical rates
- `interpolate_to_wells()`: Bilinear interpolation of grid data to well locations
- `sample_line()`: Sample permeability/porosity along well pairs

**`normalizers.py`** (✅ Complete)
- `FeatureNormalizer`: Flexible scaler (minmax, standard, log, log1p, robust)
- `create_normalizers()`: Factory for all reservoir features
- `save_normalizers()` / `load_normalizers()`: Persistence for inference

**`dataset.py`** (✅ Complete)
- `ReservoirDataset`: PyTorch Dataset for NPZ files
  - Loads 100 scenarios with full spatiotemporal data
  - Extracts node features (producers, injectors)
  - Computes edge features per timestep
  - Applies normalization
  - Returns batched tensors for training
- `create_dataloaders()`: Train/val/test split with DataLoader

### Neural Network Models

**`gnn.py`** (✅ Complete)
- `EdgeConv`: Message passing layer with edge features
  - Combines source node, target node, and edge in message
  - Learnable aggregation and update functions
- `EnhancedGNN`: Multi-layer GNN with residual connections
  - Encodes spatial well connectivity
  - 3 message passing layers with PReLU activation
- `DualGraphGNN`: Processes P2P and I2P graphs separately
  - Fuses embeddings from both graphs
  - Outputs producer embeddings (128-dim)

**`lstm.py`** (✅ Complete)
- `TemporalLSTM`: Basic LSTM for sequences
- `WellAwareLSTM`: Processes well embeddings in parallel
  - Flattens (num_wells × embedding_dim) per timestep
  - Captures temporal dynamics across all wells
- `AttentionLSTM`: Attention mechanism over wells (optional)
  - Dynamically weights well contributions
  - Returns attention weights for visualization

**`surrogate.py`** (✅ Complete)
- `RateDecoder`: Maps LSTM output to production rates
  - MLP: 256 → 64 → 20 (10 oil + 10 water)
  - Softplus activation (ensure positive rates)
- `GNN_LSTM_Surrogate`: Full end-to-end model
  - **Forward pass**:
    1. For each timestep: GNN encodes well connectivity
    2. Stack GNN outputs into sequence (batch, T, num_prod, 128)
    3. LSTM models temporal evolution
    4. Decoder predicts oil/water rates
  - **Model size**: ~2-3M parameters

**`losses.py`** (✅ Complete)
- `weighted_l1_loss()`: Weighted L1 for oil/water (β=80, α=1)
- `cumulative_production_loss()`: MSE on cumulative production
- `physics_loss()`: Constraints on monotonicity and smoothness
- `relative_error_loss()`: MAPE-style loss for better scaling
- `combined_loss()`: Flexible combination of all components

### Utilities

**`helpers.py`** (✅ Complete)
- `set_seed()`: Reproducibility
- `get_device()`: Auto-detect GPU/CPU
- `save_checkpoint()` / `load_checkpoint()`: Model persistence
- `AverageMeter`: Track training metrics
- `EarlyStopping`: Prevent overfitting

## 🎓 Training Strategy

### Hyperparameters (from `config.yaml`)

```yaml
# Model
gnn_hidden_dim: 128
gnn_num_layers: 3
lstm_hidden_dim: 256
lstm_num_layers: 2

# Training
batch_size: 8
learning_rate: 1e-4
num_epochs: 150
optimizer: Adam
weight_decay: 1e-5
gradient_clip: 1.0

# LR Scheduler
scheduler: StepLR
step_size: 50 epochs
gamma: 0.8

# Loss weights
beta (oil): 80.0
alpha (water): 1.0
cumulative_weight: 0.1
physics_weight: 0.01
```

### Data Split

- **Training**: 70 scenarios (70%)
- **Validation**: 15 scenarios (15%)
- **Test**: 15 scenarios (15%)

### Training Time Estimates

- **1 epoch**: ~2-3 minutes (RTX 2060 GPU, batch_size=8)
- **150 epochs**: ~5-7 hours
- **Preprocessing** (one-time): ~30 minutes for 100 scenarios

## 📈 Evaluation Metrics

### Implemented Metrics

1. **MAPE** (Mean Absolute Percentage Error):
   ```
   MAPE = (1/N) Σ |predicted - actual| / |actual|
   ```

2. **R²** (Coefficient of Determination):
   ```
   R² = 1 - (SS_res / SS_tot)
   ```

3. **Cumulative Error**:
   ```
   Cum_Error = |Σ predicted - Σ actual| / |Σ actual|
   ```

4. **Per-Well Breakdown**: Individual metrics for each of 10 producers

5. **Per-Timestep Error**: Early vs. late time accuracy

## 🔧 Customization Guide

### Adding New Edge Features

Edit `ml/data/preprocessing.py`:

```python
def compute_static_edge_features(...):
    # ... existing features ...

    # Add custom feature (e.g., well spacing pattern)
    spacing_feature = compute_spacing_metric(well_coords)

    edge_features = np.concatenate([
        # ... existing 10 features ...
        spacing_feature[:, None],  # Feature 11
    ], axis=1)

    return edge_features  # Now 11-dim
```

Update `config.yaml`:
```yaml
edge_features:
  dimension: 11  # Changed from 10
```

### Switching LSTM Variants

In `ml/models/surrogate.py`:

```python
# Replace WellAwareLSTM with AttentionLSTM
self.lstm = AttentionLSTM(
    num_wells=num_producers,
    well_embedding_dim=gnn_hidden_dim,
    hidden_dim=lstm_hidden_dim,
    num_layers=lstm_num_layers,
    dropout=lstm_dropout,
)
```

### Changing Graph Connectivity

In `config.yaml`:

```yaml
graph:
  p2p_mode: 'knearest'  # Change from 'voronoi'
  k_p2p: 5
  max_distance_p2p: 3000.0  # feet
```

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**:
   - Reduce `batch_size` (try 4 instead of 8)
   - Reduce `gnn_hidden_dim` (try 64 instead of 128)
   - Process timesteps in chunks

2. **Slow Training**:
   - Ensure GPU is being used (`device='cuda'`)
   - Increase `num_workers` in DataLoader (try 8)
   - Precompute static edge features

3. **Poor Convergence**:
   - Reduce learning rate (try 5e-5)
   - Increase `gradient_clip` (try 0.5)
   - Check data normalization

4. **NaN Loss**:
   - Check for zeros in denominators (normalization)
   - Reduce learning rate
   - Enable gradient clipping

## 📚 References

1. **Huang, H., Gong, B., & Sun, W.** (2023). "A Deep-Learning-Based Graph Neural Network-Long-Short-Term Memory Model for Reservoir Simulation and Optimization With Varying Well Controls." *SPE Journal*, SPE-215842-PA.

2. **PyTorch Geometric**: https://pytorch-geometric.readthedocs.io/

3. **Reservoir Simulation**: Aziz, K., & Settari, A. (1979). "Petroleum Reservoir Simulation."

## 👥 Contributors

- Enhanced GNN-LSTM architecture design
- 10-dimensional edge feature engineering
- Data-driven connectivity from time-lagged correlations
- PyTorch implementation with PyTorch Geometric

## 📝 Next Steps

### To Complete Implementation:

1. **Trainer Module** (`ml/training/trainer.py`):
   - Implement full training loop
   - Add W&B/TensorBoard logging
   - Validation loop with metrics

2. **Evaluator Module** (`ml/training/evaluator.py`):
   - Compute all metrics (MAPE, R², cumulative error)
   - Generate comparison plots
   - Per-well and per-timestep analysis

3. **Preprocessing Script** (`ml/scripts/preprocess_all.py`):
   - Load permeability/porosity fields
   - Compute time-lagged correlations
   - Save preprocessed data

4. **Training Script** (`ml/scripts/train.py`):
   - Parse CLI arguments
   - Load config
   - Initialize model, optimizer, dataloaders
   - Call trainer.fit()

5. **Visualization** (`ml/utils/visualization.py`):
   - Rate comparison plots
   - Error heatmaps
   - Attention visualization (if using AttentionLSTM)

### Enhancements:

1. Generate more scenarios (500 total, like the paper)
2. Extend simulation time to 1-2 years
3. Multiple reservoir realizations for generalization
4. Hyperparameter tuning with Optuna
5. Production optimization with PSO using trained surrogate

## 📄 License

This project is part of academic research for reservoir surrogate modeling.

---

**Status**: Core implementation complete ✅ | Training infrastructure in progress ⏳

For questions or issues, please refer to the codebase documentation or create an issue in the repository.
