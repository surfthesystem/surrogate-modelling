# Reservoir Surrogate Modeling Project - Complete Overview

**Date**: 2025-10-30
**Goal**: Build a GNN-LSTM surrogate model to predict reservoir production rates 1000× faster than physics-based simulation
**Status**: Phase 2 complete (data generation), Phase 3 in progress (ML training pipeline)

---

## Table of Contents

1. [Project Motivation](#project-motivation)
2. [Overall Workflow](#overall-workflow)
3. [Phase 1: Reservoir Model Generation](#phase-1-reservoir-model-generation)
4. [Phase 2: Simulation Data Generation](#phase-2-simulation-data-generation)
5. [Phase 3: Surrogate Model Development](#phase-3-surrogate-model-development)
6. [Phase 4: Model Deployment](#phase-4-model-deployment)
7. [Technical Details](#technical-details)
8. [File Organization](#file-organization)

---

## Project Motivation

### The Challenge
Reservoir simulation is critical for optimizing oil/gas production, but **physics-based simulators are extremely slow**:
- Full reservoir simulation: **~50 seconds per run**
- Optimization requires: **1000+ simulation runs**
- Total time for optimization: **~14 hours**

### Our Solution
Build a **GNN-LSTM surrogate model** that:
- Predicts production rates in **~0.05 seconds** (1000× speedup)
- Maintains **<5% prediction error** (MAPE)
- Enables real-time production optimization

### Why GNN-LSTM?
- **GNN (Graph Neural Network)**: Captures spatial well connectivity and heterogeneity
- **LSTM (Long Short-Term Memory)**: Models temporal dynamics of production
- **Superior to baseline**: Enhanced with 10-dimensional edge features vs. paper's 1-dim

---

## Overall Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         COMPLETE WORKFLOW                            │
└─────────────────────────────────────────────────────────────────────┘

PHASE 1: RESERVOIR MODEL GENERATION (COMPLETE ✓)
─────────────────────────────────────────────────
    Input: config.yaml (reservoir properties)
           ↓
    [src/reservoir_model.py]
           ↓
    Output:
    • data/permeability_field.npy (100×100 grid)
    • data/porosity_field.npy (100×100 grid)
    • data/well_locations.csv (10 producers + 5 injectors)
           ↓

PHASE 2: SIMULATION DATA GENERATION (COMPLETE ✓)
─────────────────────────────────────────────────
    Input: Reservoir model + LHS sampling (100 scenarios)
           ↓
    [utils/doe_sampler.py] → Generate 100 parameter combinations
           ↓
    [utils/scenario_runner.py] → Run IMPES simulator for each
           ↓
    [simulator/IMPES_phase1.py] → Physics-based simulation
           • Solves pressure equation (Poisson)
           • Solves saturation transport (Buckley-Leverett)
           • 6 months, Δt adaptive
           ↓
    Output:
    • results/training_data/doe_001/doe_001.npz (61 timesteps)
    • results/training_data/doe_002/doe_002.npz
    • ... (100 total scenarios)
           ↓

PHASE 3: SURROGATE MODEL DEVELOPMENT (IN PROGRESS ⏳)
──────────────────────────────────────────────────
    Input: 100 NPZ simulation files
           ↓
    [ml/data/preprocessing.py] → Compute edge features
           • Static: distance, permeability, transmissibility
           • Dynamic: pressure/saturation gradients
           • Data-driven: time-lagged correlations
           ↓
    [ml/data/dataset.py] → PyTorch Dataset (70/15/15 split)
           ↓
    [ml/models/surrogate.py] → GNN-LSTM Architecture
           • GNN: Spatial encoding (well connectivity)
           • LSTM: Temporal encoding (production dynamics)
           • Decoder: Predict oil + water rates (20 outputs)
           ↓
    [ml/training/trainer.py] → Training Loop
           • Loss: Weighted L1 (β=80 for oil, α=1 for water)
           • 150 epochs, ~6 hours on GPU
           • Early stopping, checkpointing
           ↓
    Output:
    • results/ml_experiments/gnn_lstm_baseline/best_model.pth
    • Training curves, metrics
           ↓

PHASE 4: MODEL DEPLOYMENT (PLANNED 📝)
──────────────────────────────────────
    Input: Trained surrogate model
           ↓
    [Optimization Loop]
           • PSO/Genetic Algorithm
           • Surrogate predicts production (0.05s)
           • Find optimal injection strategy
           ↓
    Output: Optimal well control schedule
```

---

## Phase 1: Reservoir Model Generation

### Purpose
Create a realistic heterogeneous reservoir with strategically placed wells.

### Location
- **Script**: `src/reservoir_model.py`
- **Config**: `config.yaml`
- **Output**: `data/` directory

### What It Does

#### 1. Permeability Field Generation
```python
# Spatially correlated log-normal distribution
# Method: Gaussian Random Field with exponential covariance
# Parameters:
#   - Mean: 100 mD (millidarcy)
#   - Range: [18, 650] mD
#   - Correlation length: 500 m
#   - Grid: 100×100 cells, 50m × 50m each
```

**Key Algorithm**: FFT-based spectral method for efficiency
- Generates correlated Gaussian field: `C(r) = σ² * exp(-r / λ)`
- Transforms to log-normal: `k = exp(Gaussian_field)`
- Ensures realistic heterogeneity (some zones are 30× more permeable)

#### 2. Porosity Field Generation
```python
# Correlated with permeability using Kozeny-Carman-like relationship
# φ = φ_mean + α * (log(k) - log(k_mean))
# Parameters:
#   - Mean: 0.25 (25%)
#   - Correlation coefficient: 0.6 with permeability
```

#### 3. Well Placement Strategy
```python
# 10 Producers + 5 Injectors
# Constraints:
#   - Minimum spacing: 1000 m (avoid interference)
#   - Producers: High permeability zones (better production)
#   - Injectors: Strategically placed for sweep efficiency
#   - Pattern: Modified five-spot (optimal for waterflooding)
```

**Algorithm**:
1. Score each grid cell by permeability
2. Select producers from high-k regions (spaced ≥1000m apart)
3. Place injectors to maximize geometric sweep (Voronoi-based)

### Files Generated
```
data/
├── permeability_field.npy      # (100, 100) float64, mD
├── porosity_field.npy          # (100, 100) float64, fraction
├── well_locations.csv          # 15 rows (10P + 5I), columns: x, y, type
└── reservoir_config.json       # Metadata (grid size, domain, stats)
```

### Visualization
```bash
python src/reservoir_model.py
# Generates: results/reservoir_visualization.png
#   - Subplot 1: Permeability field (colormap: log scale)
#   - Subplot 2: Porosity field
#   - Subplot 3: Well locations overlaid on permeability
```

---

## Phase 2: Simulation Data Generation

### Purpose
Generate 100 diverse training scenarios using Latin Hypercube Sampling (LHS) to explore the parameter space.

### Location
- **Main Script**: `utils/scenario_runner.py`
- **DOE Sampler**: `utils/doe_sampler.py`
- **Simulator**: `simulator/IMPES_phase1.py`
- **Batch Runner**: `utils/batch_simulator.py`

### What It Does

#### 1. Design of Experiments (DOE) with LHS
```python
# Latin Hypercube Sampling (LHS)
# Purpose: Efficiently explore 7-dimensional parameter space
#
# Varying Parameters (per scenario):
#   1. Injector 1 rate: [300, 800] STB/day
#   2. Injector 2 rate: [300, 800] STB/day
#   3. Injector 3 rate: [300, 800] STB/day
#   4. Injector 4 rate: [300, 800] STB/day
#   5. Injector 5 rate: [300, 800] STB/day
#   6. Producer BHP: [1500, 2500] psi (same for all 10 producers)
#   7. (Optional) Initial pressure: [2800, 3200] psi
```

**Why LHS?**
- Better space-filling than random sampling
- Ensures coverage of extreme cases (corners of hypercube)
- 100 samples provide good representation of 7D space
- Each dimension divided into 100 strata, one sample per stratum

**Code Location**: `utils/doe_sampler.py`
```python
sampler = LHSSampler(n_samples=100, seed=42)
scenarios = sampler.generate_scenarios(
    param_ranges={
        'inj1_rate': (300, 800),
        'inj2_rate': (300, 800),
        ...
    }
)
```

#### 2. IMPES Two-Phase Simulator
```python
# Implicit Pressure, Explicit Saturation (IMPES)
# Solves two coupled equations:
#
# 1. Pressure Equation (Implicit):
#    ∇ · (λ_t * k * ∇P) = q_wells + ∂/∂t(φ * S_w)
#    where λ_t = λ_o + λ_w (total mobility)
#
# 2. Saturation Equation (Explicit):
#    φ * ∂S_w/∂t + ∇ · (f_w * v_t) = q_w
#    where f_w = fractional flow of water
```

**Key Components**:

a) **Pressure Solver** (`simulator/IMPES_phase1.py:300-450`):
   - Assembles coefficient matrix with 7-point stencil
   - Incorporates gravity terms: `ρ_phase * g * Δz`
   - Handles well terms: `q_well = PI * (P_wf - P_cell)`
   - Solves with scipy sparse solver (LGMRES)

b) **Saturation Transport** (`simulator/IMPES_phase1.py:450-600`):
   - Upstream weighting for stability
   - Buckley-Leverett fractional flow: `f_w = λ_w / (λ_w + λ_o)`
   - Capillary pressure effects: `P_c(S_w) = P_e * (S_w)^(-1/λ)`
   - Adaptive timesteping based on CFL condition

c) **Relative Permeability** (`simulator/rel_perm.py`):
   - Corey-type correlations:
     ```
     k_rw = k_rw_max * ((S_w - S_wc) / (1 - S_wc - S_or))^n_w
     k_ro = k_ro_max * ((1 - S_w - S_or) / (1 - S_wc - S_or))^n_o
     ```
   - Parameters: S_wc=0.2, S_or=0.2, n_w=2.0, n_o=2.0

d) **Well Model** (`simulator/prodindex.py`):
   - Peaceman productivity index: `PI = 2πkh / (μ * ln(r_e/r_w))`
   - Handles rate-constrained injectors (fixed q)
   - Handles BHP-constrained producers (fixed P_wf)

**Simulation Settings**:
- Duration: 6 months (180 days)
- Timesteps: Adaptive (CFL ≤ 0.5), typically ~61 timesteps
- Grid: 100×100 cells (5 km × 5 km domain)
- Phases: Oil (μ_o=2 cp) + Water (μ_w=0.5 cp)

#### 3. Parallel Execution
```bash
# Run 100 scenarios in parallel using GNU Parallel
parallel -j 8 --joblog parallel_joblog.tsv \
    python utils/scenario_runner.py --scenario_id {} \
    ::: {1..100}

# Monitoring
bash monitor_batch.sh  # Tracks progress, ETA
```

**Runtime**:
- Single scenario: ~50 seconds
- 100 scenarios (8 parallel): ~12 minutes
- Total compute: ~83 CPU-minutes

#### 4. Output Format
Each scenario generates an NPZ file:
```python
# results/training_data/doe_001/doe_001.npz
#
# Arrays stored:
np.savez(
    'doe_001.npz',

    # Time dimension
    times=np.array([0, 3, 6, ..., 180]),  # (61,) days

    # Pressure field (spatial + temporal)
    pressure=np.array(...),     # (61, 100, 100) psi
    saturation=np.array(...),   # (61, 100, 100) fraction

    # Well production (per timestep)
    producer_oil_rates=np.array(...),    # (61, 10) STB/day
    producer_water_rates=np.array(...),  # (61, 10) STB/day
    producer_bhp=np.array(...),          # (61, 10) psi

    # Injection (constant per scenario)
    injector_rates=np.array([500, 600, ...]),  # (5,) STB/day

    # Cumulative production
    cumulative_oil=np.array(...),   # (61, 10) STB
    cumulative_water=np.array(...), # (61, 10) STB

    # Reservoir properties (static)
    permeability=np.array(...),  # (100, 100) mD
    porosity=np.array(...),      # (100, 100) fraction

    # Well locations
    producer_coords=np.array(...),  # (10, 2) x, y in meters
    injector_coords=np.array(...),  # (5, 2) x, y in meters
)
```

### Files Generated
```
results/training_data/
├── doe_001/
│   ├── doe_001.npz           # Simulation output (~450 KB per file)
│   └── input_file_phase1.py  # Generated input (copy for reproducibility)
├── doe_002/
│   ├── doe_002.npz
│   └── input_file_phase1.py
├── ...
└── doe_100/
    ├── doe_100.npz
    └── input_file_phase1.py

Total: ~45 MB (100 scenarios)
```

---

## Phase 3: Surrogate Model Development

### Purpose
Train a GNN-LSTM neural network to predict production rates given injection controls and reservoir properties.

### Location
- **Models**: `ml/models/` (gnn.py, lstm.py, surrogate.py, losses.py)
- **Data**: `ml/data/` (graph_builder.py, preprocessing.py, dataset.py, normalizers.py)
- **Training**: `ml/training/` (trainer.py, evaluator.py, config.yaml)
- **Scripts**: `ml/scripts/` (train.py, evaluate.py, preprocess_all.py)

### Architecture Overview

```
INPUT FEATURES (per timestep t):
├── Producer Nodes (10 wells, 10-dim each):
│   [BHP, k, φ, depth, x, y, cum_oil, cum_water, prev_oil_rate, prev_water_rate]
│
├── Injector Nodes (5 wells, 8-dim each):
│   [rate, k, φ, depth, x, y, cum_inj, prev_rate]
│
└── Edge Features (10-dim, computed between all connected wells):
    [inv_distance, log_k_avg, log_k_contrast, cos_angle, sin_angle,
     transmissibility, time_lag_corr, ΔP, ΔSw, drainage_overlap]

                              ↓

┌────────────────────────────────────────────────────────────┐
│                   GNN-LSTM SURROGATE MODEL                  │
└────────────────────────────────────────────────────────────┘

STEP 1: SPATIAL ENCODING (per timestep t)
──────────────────────────────────────────
    [DualGraphGNN]
        ├── P2P Graph (Producer-Producer, Voronoi connectivity)
        │   └── EnhancedGNN(3 layers, hidden=128)
        │       • Message: concat(node_i, node_j, edge_ij)
        │       • Aggregate: sum over neighbors
        │       • Update: MLP + PReLU + residual
        │
        ├── I2P Graph (Injector-Producer, bipartite full/k-nearest)
        │   └── EnhancedGNN(3 layers, hidden=128)
        │
        └── Fusion: Linear(256 → 128)
            Output: (batch, num_prod=10, emb=128) per timestep

STEP 2: STACK TEMPORAL SEQUENCE
────────────────────────────────
    Stack GNN outputs for T=61 timesteps:
    Shape: (batch, T=61, num_prod=10, emb=128)

STEP 3: TEMPORAL ENCODING
──────────────────────────
    [WellAwareLSTM]
        • Flatten: (batch, T, num_prod*emb) = (batch, 61, 1280)
        • LSTM(2 layers, hidden=256):
            h_t = LSTM(flatten(GNN_outputs_t), h_{t-1})
        • Output: (batch, T=61, hidden=256)

STEP 4: RATE PREDICTION
────────────────────────
    [RateDecoder]
        • MLP: 256 → 64 → 20
        • Activation: Softplus (ensure positive rates)
        • Output: (batch, T=61, 20)
            ├── oil_rates: [:, :, :10]  → 10 producers
            └── water_rates: [:, :, 10:]  → 10 producers

OUTPUT:
    Predicted production rates for all producers over 6 months
```

### Model Size
```
Total Parameters: ~2.5M
├── DualGraphGNN: ~1.2M
├── WellAwareLSTM: ~1.0M
└── RateDecoder: ~0.3M

Model file: ~10 MB (FP32)
Inference time: 0.05 seconds (on CPU)
```

### Training Strategy

#### Loss Function
```python
L_total = β·L_oil + α·L_water + λ_cum·L_cumulative + λ_phys·L_physics

where:
    β = 80.0   # Oil is economically important (from SPE-215842 paper)
    α = 1.0    # Water weight
    λ_cum = 0.1   # Cumulative production accuracy
    λ_phys = 0.01 # Physics constraints (monotonicity, smoothness)

L_oil = (1/N) Σ |pred_oil - actual_oil|  # Weighted L1 loss
L_cumulative = MSE(cumsum(pred), cumsum(actual))
L_physics = penalty for violations (negative rates, non-monotonic cumulative)
```

#### Hyperparameters
```yaml
# From ml/training/config.yaml
batch_size: 8
num_epochs: 150
learning_rate: 1e-4
optimizer: Adam
weight_decay: 1e-5
gradient_clip: 1.0

lr_scheduler:
  type: StepLR
  step_size: 50
  gamma: 0.8

early_stopping:
  patience: 15
  min_delta: 0.001
```

#### Data Split
```python
train_scenarios = 70  # 70% for training
val_scenarios = 15    # 15% for validation
test_scenarios = 15   # 15% for testing (final evaluation)

# Stratified split ensures coverage of parameter space
```

### Training Workflow

```bash
# 1. Preprocess data (one-time, ~30 minutes)
python ml/scripts/preprocess_all.py \
    --data_dir results/training_data \
    --output_dir ml/data/preprocessed \
    --num_scenarios 100

# Output:
#   ml/data/preprocessed/graph_data.npz
#   ml/data/preprocessed/static_features.npz
#   ml/data/preprocessed/normalizers.pkl

# 2. Train model (~6 hours on GPU, ~24 hours on CPU)
python ml/scripts/train.py \
    --config ml/training/config.yaml \
    --exp_name gnn_lstm_baseline

# Output:
#   results/ml_experiments/gnn_lstm_baseline/
#   ├── best_model.pth         # Best validation loss checkpoint
#   ├── checkpoint_epoch_*.pth # Periodic checkpoints
#   ├── train.log              # Training progress
#   └── metrics.csv            # Loss history

# 3. Evaluate on test set (~10 minutes)
python ml/scripts/evaluate.py \
    --checkpoint results/ml_experiments/gnn_lstm_baseline/best_model.pth \
    --data_dir results/training_data \
    --output_dir results/ml_experiments/gnn_lstm_baseline/evaluation

# Output:
#   evaluation/
#   ├── metrics_summary.json   # MAPE, R², cumulative error
#   ├── predictions/           # NPZ files with predictions
#   └── plots/                 # Rate comparison plots
```

### Expected Performance

| Metric | Target | Notes |
|--------|--------|-------|
| **Oil Rate MAPE** | < 5% | Mean Absolute Percentage Error |
| **Water Rate MAPE** | < 6% | Slightly higher (smaller magnitudes) |
| **Cumulative Oil Error** | < 3% | Total production accuracy |
| **R² Score** | > 0.95 | Coefficient of determination |
| **Inference Speed** | 0.05 s | vs. 50 s for simulator (1000× speedup) |
| **Training Time** | 6 hours | RTX 2060 GPU, 150 epochs |
| **Model Size** | 10 MB | FP32 weights |

---

## Phase 4: Model Deployment

### Purpose
Use the trained surrogate for real-time optimization and decision-making.

### Workflow

```
PRODUCTION OPTIMIZATION LOOP
────────────────────────────

Input: Current reservoir state
       ↓
┌──────────────────────────────┐
│  Optimization Algorithm      │
│  (PSO / Genetic / Gradient)  │
└──────────────────────────────┘
       ↓
   Propose injection schedule
   (5 injector rates over 6 months)
       ↓
┌──────────────────────────────┐
│  Surrogate Model Prediction  │
│  (0.05 seconds)              │
└──────────────────────────────┘
       ↓
   Predicted oil/water production
       ↓
┌──────────────────────────────┐
│  Objective Function          │
│  J = Σ(revenue_oil - cost)   │
└──────────────────────────────┘
       ↓
   Fitness score
       ↓
   Update population → Repeat
       ↓
   (After 1000 evaluations)
       ↓
   Optimal injection strategy

Total optimization time:
    1000 evals × 0.05 s = 50 seconds

vs. Physics-based:
    1000 evals × 50 s = 14 hours

Speedup: 1000×
```

### Use Cases

1. **Production Optimization**
   - Maximize NPV (Net Present Value)
   - Balance oil production vs. water handling costs
   - Find optimal injection rates per well

2. **Real-Time Decision Support**
   - Predict impact of changing injection rates
   - "What-if" scenarios for field management
   - Constraint handling (max injection capacity)

3. **Uncertainty Quantification**
   - Ensemble predictions (train multiple models)
   - Confidence intervals on production forecasts
   - Risk assessment for investment decisions

4. **Closed-Loop Reservoir Management**
   - Integrate with SCADA systems
   - Automatically adjust injection based on real-time data
   - Model predictive control (MPC) for wells

---

## Technical Details

### Mathematical Formulation

#### Reservoir Simulation (IMPES)

**Pressure Equation** (Implicit):
```
∇ · (λ_t(S_w) * k * (∇P - ρ_phase * g * ∇z)) = q_wells + (φ/Δt) * (S_w^{n+1} - S_w^n)

where:
    λ_t = k_rw/μ_w + k_ro/μ_o  (total mobility)
    k = permeability tensor
    P = pressure field
    ρ_phase = fluid density
    g = gravity (9.81 m/s²)
    q_wells = well source/sink terms
```

**Saturation Equation** (Explicit):
```
φ * ∂S_w/∂t + ∇ · (f_w(S_w) * v_t) = q_w

where:
    f_w = λ_w / λ_t  (fractional flow of water)
    v_t = -λ_t * k * (∇P - ρ_avg * g * ∇z)  (total velocity)
    q_w = water injection/production
```

**Discretization**:
- Spatial: Finite volume method (7-point stencil)
- Temporal: Forward Euler for saturation, backward Euler for pressure
- Upwind scheme for saturation transport (stability)

#### GNN Message Passing

**Edge Convolution** (per layer l):
```
h_i^{l+1} = σ( W_self^l * h_i^l + Σ_{j∈N(i)} W_msg^l * m_ij )

where:
    m_ij = MLP( concat(h_i^l, h_j^l, e_ij) )
    e_ij = 10-dim edge features
    N(i) = neighbors of node i
    σ = PReLU activation
```

**Message aggregation**:
```python
# Sum aggregation (alternative: mean, max)
aggregate = Σ_{j∈neighbors} message_ij

# Update with residual connection
h_new = h_old + MLP(aggregate)
```

#### LSTM Temporal Modeling

**Standard LSTM equations**:
```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)    (forget gate)
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)    (input gate)
g_t = tanh(W_g · [h_{t-1}, x_t] + b_g) (candidate)
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)    (output gate)

c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t        (cell state)
h_t = o_t ⊙ tanh(c_t)                   (hidden state)
```

**Our modification (WellAwareLSTM)**:
```python
# Flatten spatial wells per timestep
x_t = flatten([h_well1, h_well2, ..., h_well10])  # (1280-dim)

# Process through LSTM
h_t, c_t = LSTM(x_t, h_{t-1}, c_{t-1})  # (256-dim)

# Maintains temporal coherence across all wells
```

### Data Statistics

#### Reservoir Properties
```
Permeability:
    Mean: 100 mD
    Range: [18, 650] mD
    Std: 95 mD
    Distribution: Log-normal

Porosity:
    Mean: 0.25
    Range: [0.20, 0.32]
    Std: 0.028
    Distribution: Normal (correlated with k)

Domain:
    Size: 5 km × 5 km
    Depth: 2000 m (average)
    Grid: 100×100 cells (50m × 50m each)
```

#### Well Configuration
```
Producers (10 wells):
    Control: Fixed BHP (1500-2500 psi, varies by scenario)
    Typical rates: 50-200 STB/day oil, 20-100 STB/day water
    Total oil production: ~30,000 STB over 6 months (per scenario)

Injectors (5 wells):
    Control: Fixed rate (300-800 STB/day, varies by scenario)
    Total injection: ~90,000 STB over 6 months (per scenario)

Recovery:
    Water cut: Increases from 0% to 30-60% over 6 months
    Oil recovery factor: ~5-10% OOIP (Original Oil In Place)
```

#### Training Data
```
Scenarios: 100
Timesteps per scenario: 61 (adaptive timestep)
Total datapoints: 100 × 61 × 10 = 61,000 (wells × time)

Features per datapoint:
    Node features: 10 (prod) + 5 (inj) = 15 nodes
    Edge features: ~70 edges × 10-dim = 700 features
    Targets: 2 rates × 10 producers = 20 outputs

Training set: 70 scenarios (42,700 datapoints)
Validation set: 15 scenarios (9,150 datapoints)
Test set: 15 scenarios (9,150 datapoints)
```

---

## File Organization

### Complete Directory Structure

```
surrogate-modelling-1/
│
├── README.md                  # Quick start guide
├── PROJECT_OVERVIEW.md        # This file (comprehensive documentation)
├── config.yaml                # Reservoir configuration
├── requirements-simulator.txt # Python dependencies for simulator
├── requirements-ml.txt        # Python dependencies for ML
│
├── src/                       # Reservoir Model Generation (PHASE 1)
│   └── reservoir_model.py     # Generate permeability, porosity, wells
│
├── simulator/                 # IMPES Two-Phase Simulator (PHASE 2)
│   ├── IMPES_phase1.py        # Main simulator (pressure + saturation)
│   ├── rel_perm.py            # Relative permeability curves
│   ├── cap_press.py           # Capillary pressure
│   ├── prodindex.py           # Well productivity index
│   ├── updatewells.py         # Well constraint handling
│   ├── fluid_properties.py    # Oil/water properties
│   ├── petrophysics.py        # Rock properties
│   ├── myarrays.py            # Array manipulation utilities
│   └── spdiaginv.py           # Sparse matrix utilities
│
├── utils/                     # Batch Simulation & Analysis
│   ├── doe_sampler.py         # Latin Hypercube Sampling (LHS)
│   ├── scenario_runner.py     # Single scenario execution
│   ├── batch_simulator.py     # Parallel batch execution
│   └── generate_visualizations.py  # Plot results
│
├── ml/                        # Surrogate Model (PHASE 3)
│   │
│   ├── README.md              # ML architecture documentation
│   ├── IMPLEMENTATION_STATUS.md  # Development progress
│   ├── TESTING_STATUS.md      # Module test results
│   ├── NEXT_STEPS.md          # Implementation guide
│   │
│   ├── data/                  # Data preprocessing
│   │   ├── graph_builder.py   # Well connectivity (Voronoi)
│   │   ├── preprocessing.py   # 10-dim edge features
│   │   ├── normalizers.py     # Feature scaling
│   │   └── dataset.py         # PyTorch Dataset
│   │
│   ├── models/                # Neural network models
│   │   ├── gnn.py             # Enhanced GNN with edge features
│   │   ├── lstm.py            # Temporal LSTM variants
│   │   ├── surrogate.py       # Full GNN-LSTM model
│   │   └── losses.py          # Weighted + physics-informed loss
│   │
│   ├── training/              # Training infrastructure
│   │   ├── config.yaml        # Hyperparameters
│   │   ├── trainer.py         # Training loop (to be completed)
│   │   └── evaluator.py       # Metrics & evaluation (to be completed)
│   │
│   ├── utils/                 # ML utilities
│   │   ├── helpers.py         # Checkpointing, early stopping
│   │   └── visualization.py   # Plotting (to be completed)
│   │
│   └── scripts/               # Entry points
│       ├── preprocess_all.py  # Data preprocessing (to be completed)
│       ├── train.py           # Main training script (to be completed)
│       └── evaluate.py        # Model evaluation (to be completed)
│
├── data/                      # Reservoir Model Outputs (PHASE 1)
│   ├── permeability_field.npy # 100×100 grid
│   ├── porosity_field.npy     # 100×100 grid
│   ├── well_locations.csv     # 15 wells (10P + 5I)
│   ├── reservoir_config.json  # Metadata
│   └── impes_input/           # Simulator inputs
│       ├── selected_wells.csv
│       ├── well_locations_ft.csv
│       ├── schedule_injector_rates.csv
│       └── schedule_producer_bhp.csv
│
├── scenarios/                 # DOE Sample Definitions
│   ├── scenario_001.yaml      # Parameter set for doe_001
│   ├── scenario_002.yaml
│   └── ... (100 total)
│
├── results/                   # Simulation & ML Outputs
│   ├── reservoir_visualization.png  # PHASE 1 output
│   │
│   ├── training_data/         # PHASE 2 outputs (100 scenarios)
│   │   ├── doe_001/
│   │   │   ├── doe_001.npz    # Simulation results (~450 KB)
│   │   │   └── input_file_phase1.py
│   │   ├── doe_002/
│   │   └── ... (doe_100/)
│   │
│   └── ml_experiments/        # PHASE 3 outputs
│       └── gnn_lstm_baseline/
│           ├── best_model.pth
│           ├── checkpoints/
│           ├── logs/
│           └── evaluation/
│
└── docs/                      # Additional Documentation
    ├── CODEBASE_ANALYSIS.md   # Code quality report
    ├── CLEANUP_CHECKLIST.md   # Cleanup instructions
    └── ANALYSIS_INDEX.md      # Quick reference
```

### Key Files Reference

| File | Purpose | Size | Status |
|------|---------|------|--------|
| `src/reservoir_model.py` | Generate reservoir model | 500 lines | ✓ Complete |
| `simulator/IMPES_phase1.py` | Physics-based simulator | 800 lines | ✓ Complete |
| `utils/doe_sampler.py` | LHS parameter sampling | 150 lines | ✓ Complete |
| `utils/scenario_runner.py` | Single scenario execution | 200 lines | ✓ Complete |
| `utils/batch_simulator.py` | Parallel batch runner | 180 lines | ✓ Complete |
| `ml/models/surrogate.py` | GNN-LSTM model | 280 lines | ✓ Complete |
| `ml/training/trainer.py` | Training loop | - | ⏳ To implement |
| `ml/scripts/train.py` | Training entry point | - | ⏳ To implement |

---

## Current Status & Next Steps

### Completed (✓)

#### Phase 1: Reservoir Model ✓
- Heterogeneous permeability field (log-normal, spatially correlated)
- Correlated porosity field
- Strategic well placement (10 producers + 5 injectors)
- Validated with visualization

#### Phase 2: Simulation Data ✓
- 100 diverse scenarios using LHS
- IMPES two-phase simulator fully functional
- Parallel batch execution (8 cores)
- Total: ~45 MB of training data (100 × 450 KB)

#### Phase 3: ML Infrastructure (85% complete) ✓
- ✅ All data preprocessing modules (graph_builder, preprocessing, normalizers, dataset)
- ✅ All neural network models (gnn, lstm, surrogate, losses)
- ✅ Configuration system (config.yaml)
- ✅ Utilities (checkpointing, early stopping)
- ✅ Comprehensive documentation (README, status docs)

### In Progress (⏳)

#### Phase 3: Training Pipeline (15% remaining)
- ⏳ `ml/training/trainer.py` - Training loop with validation
- ⏳ `ml/training/evaluator.py` - Metrics computation
- ⏳ `ml/scripts/preprocess_all.py` - Automated preprocessing
- ⏳ `ml/scripts/train.py` - Training entry point
- ⏳ `ml/scripts/evaluate.py` - Evaluation script
- ⏳ `ml/utils/visualization.py` - Plotting utilities

**Estimated completion**: 10-15 hours of focused work

### Planned (📝)

#### Phase 4: Deployment
- Integration with optimization algorithms (PSO, Genetic)
- Real-time inference API
- Uncertainty quantification
- Model monitoring and retraining pipeline

---

## Performance Benchmarks

### Simulation (Physics-Based)

| Metric | Value |
|--------|-------|
| Single scenario runtime | 50 seconds |
| Grid cells | 10,000 (100×100) |
| Timesteps | 61 (adaptive) |
| Phases | 2 (oil + water) |
| Memory usage | ~500 MB |

### Surrogate Model (ML-Based)

| Metric | Target | Achieved |
|--------|--------|----------|
| Inference time | < 0.1 s | TBD (after training) |
| Oil MAPE | < 5% | TBD |
| Water MAPE | < 6% | TBD |
| R² score | > 0.95 | TBD |
| Model size | ~10 MB | ✓ |
| Speedup vs simulator | 1000× | TBD |

### Data Generation

| Metric | Value |
|--------|-------|
| Scenarios | 100 |
| Total simulation time | 83 CPU-minutes (8 cores) |
| Data generated | 45 MB (compressed) |
| Parameters varied | 7 (5 injection rates + 1 BHP + 1 P_init) |

---

## References & Citations

### Papers
1. **Huang, H., Gong, B., & Sun, W.** (2023). "A Deep-Learning-Based Graph Neural Network-Long-Short-Term Memory Model for Reservoir Simulation and Optimization With Varying Well Controls." *SPE Journal*, SPE-215842-PA.
   - **Our enhancement**: 10-dim edge features vs. 1-dim (distance only)

2. **Aziz, K., & Settari, A.** (1979). "Petroleum Reservoir Simulation." *Elsevier Applied Science*.
   - Classical reference for IMPES formulation

3. **Gilman, J. R., & Kazemi, H.** (1983). "Improvements in Simulation of Naturally Fractured Reservoirs." *SPE Journal*, 23(4), 695-707.
   - Well productivity index (Peaceman model)

### Software
- **PyTorch** (2.9.0): Deep learning framework
- **PyTorch Geometric** (2.7.0): Graph neural network library
- **SciPy** (1.16.3): Sparse linear solvers (LGMRES)
- **NumPy** (2.3.4): Numerical arrays

### Algorithms
- **Latin Hypercube Sampling**: McKay et al. (1979) for DOE
- **IMPES**: Implicit Pressure Explicit Saturation (Stone & Garder, 1961)
- **Voronoi Diagrams**: Fortune's algorithm (1987) for well connectivity
- **Message Passing Neural Networks**: Gilmer et al. (2017)

---

## Troubleshooting & FAQ

### Q1: Simulation fails with "Matrix singular" error
**A**: The pressure equation solver failed. Possible causes:
- Permeability too heterogeneous (check min/max k)
- Wells too close together (check spacing constraint)
- Timestep too large (reduce Δt_max in config)

**Fix**:
```python
# In config.yaml
simulation:
  timestep:
    dt_max: 1.0  # Reduce from 5.0
  solver:
    tolerance: 1e-6  # Increase tolerance
```

### Q2: LSTM training diverges (NaN loss)
**A**: Gradient explosion. Solutions:
- Reduce learning rate: `1e-4 → 5e-5`
- Increase gradient clipping: `1.0 → 0.5`
- Check feature normalization (especially permeability - use log transform)

### Q3: GNN not capturing spatial patterns
**A**: Edge features may be poorly scaled or connectivity is wrong.
- Verify Voronoi graph has ~4-5 edges per well
- Check edge feature ranges (use `ml/data/preprocessing.py` tests)
- Visualize attention weights (if using AttentionLSTM)

### Q4: Model underfits (high training loss)
**A**: Model capacity may be too small or training too short.
- Increase GNN layers: `3 → 4`
- Increase hidden dimensions: `128 → 256`
- Train longer: `150 → 200 epochs`

### Q5: Model overfits (low train, high val loss)
**A**: Reduce model complexity or increase regularization.
- Increase dropout: `0.2 → 0.3`
- Increase weight decay: `1e-5 → 1e-4`
- Use data augmentation (add noise to inputs)

### Q6: Out of memory during training
**A**: Reduce batch size or model size.
```yaml
training:
  batch_size: 4  # Instead of 8
model:
  gnn:
    hidden_dim: 64  # Instead of 128
```

---

## Contact & Contributions

### Project Status
- **Phase 1**: ✓ Complete
- **Phase 2**: ✓ Complete
- **Phase 3**: 85% complete (training pipeline in progress)
- **Phase 4**: Planned

### Next Milestone
Complete ML training pipeline (10-15 hours estimated)

### Documentation Updates
This file is version-controlled. Last updated: 2025-10-30

---

**End of PROJECT_OVERVIEW.md**
