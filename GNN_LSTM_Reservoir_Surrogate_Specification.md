# GNN-LSTM Reservoir Surrogate Model - Project Specification

## Overview
This project implements a Graph Neural Network - Long Short-Term Memory (GNN-LSTM) surrogate model for fast reservoir simulation, following the methodology from SPE-215842-PA (Huang et al., 2023). The surrogate model will predict oil and water production rates under varying well controls, achieving 100-1000x speedup over traditional reservoir simulation.

---

## Project Objectives

### Primary Goals
1. **Build a 2D heterogeneous reservoir model** with realistic geology
   - Grid size: ~100x100 cells (10,000 active cells)
   - 40 producer wells
   - 20 injector wells
   - Heterogeneous permeability and porosity fields
   
2. **Generate training data** through full physics simulations
   - Target: 200-300 simulation cases
   - Varying well controls (injection rates, producer BHPs)
   - Multi-year production forecasts (10-15 timesteps)
   
3. **Develop GNN-LSTM surrogate model** 
   - Two GNN models: producer-producer and injector-producer connectivity
   - LSTM for temporal evolution
   - Target accuracy: <10% average relative error on oil/water rates
   
4. **Visualization and validation**
   - Reservoir model visualization
   - Training convergence plots
   - Surrogate vs. full physics comparison
   - Production rate predictions over time

---

## Phase 1: Reservoir Model Development

### 1.1 Repository Setup
```bash
# Clone the Reservoir-Simulator repository
git clone https://github.com/mashadab/Reservoir-Simulator.git
cd Reservoir-Simulator

# Install dependencies
pip install numpy scipy matplotlib pandas
```

### 1.2 Reservoir Specifications

**Grid Properties:**
- Dimensions: 100 x 100 x 1 (2D single layer)
- Cell size: 50m x 50m (5km x 5km reservoir)
- Active cells: ~8,000-9,000 (after applying irregular boundary)

**Rock Properties (Heterogeneous):**
- **Permeability**: Log-normal distribution
  - Mean: 100 mD
  - Range: 10-500 mD
  - Spatial correlation using Gaussian random field (correlation length: 500m)
  
- **Porosity**: Correlated with permeability
  - Mean: 0.20
  - Range: 0.15-0.30
  - Correlation: porosity = 0.15 + 0.0005 * sqrt(permeability)

**Fluid Properties:**
- Three-phase flow: oil, water, gas
- Initial pressure: 3000 psi
- Initial water saturation: 0.25
- Initial oil saturation: 0.65
- Initial gas saturation: 0.10
- Oil viscosity: 2 cp
- Water viscosity: 0.5 cp
- Oil density: 850 kg/m³
- Water density: 1000 kg/m³

**Relative Permeability Curves:**
- Use standard Corey-type curves
- Oil-water: Sorw = 0.2, Swc = 0.25
- Oil-gas: Sorg = 0.1, Sgc = 0.05

### 1.3 Well Configuration

**Producer Wells (40 wells):**
- Placement: Distributed across reservoir using stratified sampling
  - Divide reservoir into 8x5 regions
  - Place 1 well per region with random offset
- Control: Constant bottomhole pressure (BHP)
  - BHP range: 1500-2500 psi (will vary for training data)
- Completion: Full reservoir thickness

**Injector Wells (20 wells):**
- Placement: Distributed to provide good sweep
  - Located between producer clusters
  - Avoid reservoir edges
- Control: Constant water injection rate
  - Rate range: 100-1000 STB/day (will vary for training data)
- Completion: Full reservoir thickness

**Well Pattern:**
```
Strategy: Modified 5-spot pattern with some irregular placement
- Maintain minimum well spacing: 500m
- Avoid areas with very low permeability (<20 mD)
- Ensure each injector influences 2-3 producers
```

### 1.4 Implementation Tasks

**Task 1.1:** Create heterogeneous permeability field
```python
# Use scipy to generate Gaussian random field
# Apply log-normal transformation
# Create spatial correlation structure
# Output: permeability array (100x100)
```

**Task 1.2:** Generate porosity field
```python
# Correlate with permeability
# Apply reasonable bounds
# Output: porosity array (100x100)
```

**Task 1.3:** Place wells strategically
```python
# Generate producer locations (40 wells)
# Generate injector locations (20 wells)
# Ensure minimum spacing constraints
# Output: well coordinate lists
```

**Task 1.4:** Configure reservoir model
```python
# Set up grid structure
# Assign rock properties
# Define fluid properties
# Configure relative permeability curves
# Add wells with initial controls
```

**Task 1.5:** Validate single simulation
```python
# Run baseline case (10 years, 1-year timesteps)
# Check mass balance
# Verify production rates are reasonable
# Generate visualization of results
```

### 1.6 Deliverables for Phase 1
- [ ] `reservoir_model.py` - Main reservoir class
- [ ] `permeability_field.npy` - Saved permeability field
- [ ] `porosity_field.npy` - Saved porosity field
- [ ] `well_locations.csv` - Producer and injector coordinates
- [ ] `baseline_simulation_results.pkl` - Reference case results
- [ ] `phase1_visualization.png` - Permeability field with well locations

---

## Phase 2: Training Data Generation

### 2.1 Sampling Strategy

**Well Control Variations:**

For each simulation case, randomly sample:

**Injector Controls:**
- Water injection rate: Uniform [100, 1000] STB/day per well
- Total field injection constraint: 8,000-12,000 STB/day
- Control changes: Yearly (10 timesteps over 10 years)

**Producer Controls:**
- Bottomhole pressure: Uniform [1500, 2500] psi
- Control changes: Yearly (synchronized with injectors)

**Sampling Method:**
```python
# For each of 200-300 cases:
# 1. Generate random initial controls for all wells
# 2. At each year, perturb controls by ±10-20%
# 3. Maintain physical constraints
# 4. Run full physics simulation
```

### 2.2 Data Collection

**For each simulation, record:**

**Static Data (per well):**
- Well location (x, y)
- Grid block permeability (kx, ky)
- Grid block porosity
- Well type (producer/injector)

**Dynamic Data (per timestep):**
- Producer BHP (control)
- Injector rate (control)
- Cumulative water injected (injectors)
- Oil production rate (output)
- Water production rate (output)
- Gas production rate (output - optional)

**Data Structure:**
```python
simulation_data = {
    'case_id': int,
    'static_features': {
        'producer_locations': np.array (40, 2),
        'injector_locations': np.array (20, 2),
        'producer_perm_x': np.array (40,),
        'producer_perm_y': np.array (40,),
        'producer_porosity': np.array (40,),
        'injector_perm_x': np.array (20,),
        'injector_perm_y': np.array (20,),
        'injector_porosity': np.array (20,)
    },
    'dynamic_data': {
        'timesteps': np.array (10,),  # years
        'producer_bhp': np.array (40, 10),
        'injector_rate': np.array (20, 10),
        'injector_cum_volume': np.array (20, 10),
        'producer_oil_rate': np.array (40, 10),
        'producer_water_rate': np.array (40, 10),
        'producer_gas_rate': np.array (40, 10)  # optional
    }
}
```

### 2.3 Implementation Tasks

**Task 2.1:** Create control sampling function
```python
def generate_control_schedule(n_producers, n_injectors, n_timesteps):
    # Sample initial controls
    # Generate temporal variations
    # Apply constraints
    # Return control arrays
```

**Task 2.2:** Set up simulation loop
```python
def run_training_simulations(n_cases=250):
    for case_id in range(n_cases):
        # Generate controls
        # Configure reservoir
        # Run simulation
        # Extract and save results
        # Log progress
```

**Task 2.3:** Implement parallel execution (optional)
```python
# If simulations are slow, parallelize
# Use multiprocessing or joblib
# Target: <6 hours total for all cases
```

**Task 2.4:** Data validation and quality checks
```python
# Check for failed simulations
# Verify all wells producing/injecting
# Check for unphysical results
# Filter outliers if necessary
```

### 2.4 Deliverables for Phase 2
- [ ] `generate_training_data.py` - Main data generation script
- [ ] `training_data/` - Directory with all simulation results
  - [ ] `case_000.pkl` to `case_249.pkl` - Individual case results
  - [ ] `training_manifest.csv` - Summary of all cases
- [ ] `data_statistics.json` - Statistics of training data
- [ ] `phase2_visualization.png` - Sample of control variations

---

## Phase 3: GNN-LSTM Surrogate Model

### 3.1 Model Architecture

Following the paper (SPE-215842-PA), implement:

**3.1.1 Graph Construction**

**Producer-Producer Graph:**
- Nodes: 40 producers
- Edges: Based on Voronoi diagram adjacency
- Node features (per timestep):
  - BHP (1 value)
  - Permeability x,y (2 values)
  - Porosity (1 value)
  - **Total: 4 features → project to 128-dim**

**Injector-Producer Graph:**
- Nodes: 20 injectors + 40 producers
- Edges: Injector → Producer connections (Voronoi based)
- Injector node features:
  - Injection rate (1 value)
  - Permeability x,y (2 values)
  - Cumulative injection (1 value)
  - Porosity (1 value)
  - **Total: 5 features → project to 128-dim**
- Producer node features: Same as above (4 features)

**Edge Features:**
- Weight = 1/distance between wells
- 1 feature per edge

**3.1.2 GNN Architecture**

```
For Producer-Producer Graph:
Input: Producer features (40, 4)
    ↓
Neural Net: FC(64) + PReLU + FC(128)
    ↓ 
GNN Aggregation: (40, 128)
    ↓
Output: Producer embeddings (40, 128)

For Injector-Producer Graph:
Input: Injector features (20, 5) + Producer features (40, 4)
    ↓
Neural Net: FC(64) + PReLU + FC(128) for each
    ↓
GNN Aggregation (injector → producer): (40, 128)
    ↓
Output: Producer embeddings from injection (40, 128)

Combine both:
    Concat[Producer-Producer, Injector-Producer] → (40, 256)
    ↓
    FC(128) + PReLU + FC(128)
    ↓
    Final producer features: (40, 128) per timestep
```

**3.1.3 LSTM Architecture**

```
Input: Producer features over time (40, n_timesteps, 128)
    ↓
LSTM Layer: hidden_size=128
    ↓
Output: Evolved features (40, n_timesteps, 128)
    ↓
FC(64) + PReLU + FC(2)  [for oil and water rates]
    ↓
Predictions: (40, n_timesteps, 2)  # oil rate, water rate
```

**3.1.4 Loss Function**

Following the paper:
```python
# Oil loss
E_oil = (1/T) * (1/M) * sum(|q_oil_predicted - q_oil_actual|)

# Water loss
E_water = (1/T) * (1/M) * sum(|q_water_predicted - q_water_actual|)

# Total loss
Loss = α * E_water + β * E_oil

# Where T = timesteps, M = number of producers
# Paper uses: α = 80, β = 1
```

### 3.2 Implementation Tasks

**Task 3.1:** Data preprocessing
```python
class ReservoirDataset:
    def __init__(self, data_dir, train=True):
        # Load simulation cases
        # Normalize inputs (min-max normalization)
        # Create graph structures
        # Split train/test (80/20)
        
    def __getitem__(self, idx):
        # Return: graphs, features, targets
```

**Task 3.2:** Graph construction
```python
def construct_graphs(well_locations, n_producers, n_injectors):
    # Create Voronoi diagram
    # Build adjacency matrices
    # Compute edge weights (1/distance)
    # Return PyTorch Geometric Data objects
```

**Task 3.3:** Implement GNN layers
```python
class ProducerProducerGNN(nn.Module):
    def __init__(self):
        # Feature projection layers
        # Graph convolution layers
        # Aggregation function
        
    def forward(self, node_features, edge_index, edge_weight):
        # Project features to high dimension
        # Apply graph convolution
        # Aggregate neighbor information
        # Return updated node embeddings
```

**Task 3.4:** Implement full GNN-LSTM model
```python
class ReservoirGNNLSTM(nn.Module):
    def __init__(self):
        self.producer_gnn = ProducerProducerGNN()
        self.injector_producer_gnn = InjectorProducerGNN()
        self.feature_combine = nn.Sequential(...)
        self.lstm = nn.LSTM(...)
        self.output_head = nn.Sequential(...)
        
    def forward(self, producer_features, injector_features, graphs):
        # Process through GNNs
        # Combine features
        # Pass through LSTM
        # Generate predictions
```

**Task 3.5:** Training loop
```python
def train_model(model, train_loader, val_loader, n_epochs=100):
    optimizer = torch.optim.Adam(lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(step_size=50, gamma=0.8)
    
    for epoch in range(n_epochs):
        # Training
        # Validation
        # Logging
        # Checkpoint saving
```

### 3.3 Training Configuration

**Hyperparameters:**
```python
config = {
    'batch_size': 16,
    'learning_rate': 1e-4,
    'n_epochs': 100,
    'hidden_dim': 128,
    'lstm_hidden': 128,
    'dropout': 0.1,
    'weight_decay': 1e-5,
    'alpha': 80,  # water loss weight
    'beta': 1,    # oil loss weight
}
```

**Hardware:**
- GPU recommended (but CPU okay for 60 wells)
- Expected training time: 1-2 hours on modern GPU
- RAM requirement: 8-16 GB

### 3.4 Deliverables for Phase 3
- [ ] `gnn_lstm_model.py` - Model architecture
- [ ] `dataset.py` - Data loading and preprocessing
- [ ] `train.py` - Training script
- [ ] `config.yaml` - Hyperparameters
- [ ] `models/` - Saved model checkpoints
- [ ] `training_logs/` - Tensorboard logs
- [ ] `phase3_results.json` - Training metrics

---

## Phase 4: Validation and Visualization

### 4.1 Performance Metrics

**Prediction Accuracy:**
```python
# For each test case, compute:
1. Mean Absolute Error (MAE):
   MAE_oil = mean(|q_oil_pred - q_oil_true|)
   MAE_water = mean(|q_water_pred - q_water_true|)

2. Relative Error:
   RE_oil = MAE_oil / mean(|q_oil_true|) * 100%
   RE_water = MAE_water / mean(|q_water_true|) * 100%

3. R² Score:
   R2_oil = 1 - sum((q_pred - q_true)²) / sum((q_true - q_mean)²)

4. Well-wise accuracy:
   - Individual well predictions
   - Identify worst/best performing wells
```

**Computational Performance:**
```python
# Compare:
- Time for full physics simulation
- Time for GNN-LSTM prediction
- Speedup factor
- Memory usage
```

### 4.2 Visualization Requirements

**Visualization 1: Reservoir Model**
```python
# Create figure showing:
- Permeability field (heatmap)
- Well locations (producers=red, injectors=blue)
- Grid structure
- Scale bar and legends
```

**Visualization 2: Training Convergence**
```python
# Plot:
- Training loss vs. epoch
- Validation loss vs. epoch
- Oil error vs. epoch
- Water error vs. epoch
# With proper labels and legends
```

**Visualization 3: Prediction Comparison - Time Series**
```python
# For 6 selected wells (3 high producers, 3 average):
- Plot oil rate over time: True vs. Predicted
- Plot water rate over time: True vs. Predicted
- Show confidence intervals if available
- Subplot for each well
```

**Visualization 4: Prediction Comparison - Scatter**
```python
# All wells, all timesteps:
- Scatter plot: Predicted vs. True oil rates
- Scatter plot: Predicted vs. True water rates
- Add 45° line
- Color points by timestep
- Show R² value
```

**Visualization 5: Spatial Error Distribution**
```python
# Map view of reservoir:
- Color each well by average prediction error
- Size by total production
- Show which areas are well-predicted vs. poorly predicted
```

**Visualization 6: Example Simulation Run**
```python
# Animated or multi-panel showing:
- Pressure field evolution
- Oil saturation evolution
- Well production rates
- For one interesting test case
```

**Visualization 7: Performance Comparison**
```python
# Bar chart showing:
- Full physics simulation time
- GNN-LSTM prediction time
- Speedup factor
# Include error bars if multiple runs
```

### 4.3 Implementation Tasks

**Task 4.1:** Evaluation pipeline
```python
def evaluate_model(model, test_loader):
    # Run predictions on all test cases
    # Compute all metrics
    # Generate per-well statistics
    # Save results to JSON/CSV
```

**Task 4.2:** Create all visualizations
```python
# Implement plotting functions
# Use matplotlib/seaborn
# Save high-resolution figures
# Create summary dashboard
```

**Task 4.3:** Speed benchmarking
```python
def benchmark_performance():
    # Time full physics runs
    # Time GNN-LSTM inference
    # Report statistics
```

**Task 4.4:** Generate report
```python
# Create markdown report with:
# - Project overview
# - Model architecture summary
# - Performance metrics
# - All visualizations
# - Conclusions and recommendations
```

### 4.4 Deliverables for Phase 4
- [ ] `evaluate.py` - Evaluation script
- [ ] `visualizations.py` - Plotting functions
- [ ] `results/` - Directory with all outputs:
  - [ ] `reservoir_visualization.png`
  - [ ] `training_convergence.png`
  - [ ] `prediction_timeseries.png`
  - [ ] `prediction_scatter.png`
  - [ ] `spatial_error_map.png`
  - [ ] `simulation_animation.gif` or `.mp4`
  - [ ] `performance_comparison.png`
  - [ ] `metrics_summary.json`
- [ ] `FINAL_REPORT.md` - Comprehensive project report

---

## Success Criteria

### Minimum Acceptable Performance
- ✅ Oil rate relative error < 15%
- ✅ Water rate relative error < 20%
- ✅ Speedup > 50x
- ✅ Model trains successfully
- ✅ All visualizations generated

### Target Performance (Paper-like)
- 🎯 Oil rate relative error < 10%
- 🎯 Water rate relative error < 10%
- 🎯 Speedup > 100x
- 🎯 R² > 0.9 for both phases

### Stretch Goals
- 🌟 Oil rate relative error < 5%
- 🌟 Water rate relative error < 5%
- 🌟 Speedup > 500x
- 🌟 Demonstrate optimization capability

---

## Implementation Timeline

### Estimated Effort (with Claude Code assistance):

**Phase 1: Reservoir Model** - 2-4 hours
- Setup and configuration: 30 min
- Permeability/porosity generation: 1 hour
- Well placement: 1 hour
- Initial validation: 1-1.5 hours

**Phase 2: Training Data** - 4-8 hours
- Sampling strategy implementation: 1 hour
- Data generation setup: 1 hour
- Running simulations: 2-5 hours (depends on hardware)
- Data processing and validation: 1 hour

**Phase 3: GNN-LSTM Model** - 6-10 hours
- Graph construction: 2 hours
- Model architecture: 2-3 hours
- Training pipeline: 2 hours
- Model training: 2-3 hours

**Phase 4: Validation** - 2-4 hours
- Evaluation metrics: 1 hour
- Visualization creation: 2 hours
- Report generation: 1 hour

**Total: 14-26 hours** (including debugging and iterations)

---

## File Structure

```
reservoir-surrogate-project/
│
├── README.md
├── requirements.txt
├── config.yaml
│
├── data/
│   ├── reservoir_config.json
│   ├── permeability_field.npy
│   ├── porosity_field.npy
│   ├── well_locations.csv
│   └── training_data/
│       ├── case_000.pkl
│       ├── case_001.pkl
│       └── ...
│
├── src/
│   ├── __init__.py
│   ├── reservoir_model.py       # Reservoir setup and simulation
│   ├── data_generator.py        # Training data generation
│   ├── dataset.py               # PyTorch Dataset class
│   ├── graph_builder.py         # Graph construction utilities
│   ├── gnn_lstm_model.py        # Model architecture
│   ├── train.py                 # Training script
│   ├── evaluate.py              # Evaluation script
│   └── visualizations.py        # Plotting functions
│
├── notebooks/
│   ├── 01_reservoir_exploration.ipynb
│   ├── 02_data_analysis.ipynb
│   └── 03_model_results.ipynb
│
├── models/
│   ├── best_model.pth
│   └── checkpoint_epoch_100.pth
│
├── results/
│   ├── metrics_summary.json
│   ├── reservoir_visualization.png
│   ├── training_convergence.png
│   ├── prediction_timeseries.png
│   ├── prediction_scatter.png
│   ├── spatial_error_map.png
│   └── simulation_animation.mp4
│
├── logs/
│   └── training_logs/
│
└── FINAL_REPORT.md
```

---

## Dependencies

```python
# requirements.txt
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.4.0
pandas>=1.3.0
torch>=2.0.0
torch-geometric>=2.3.0
scikit-learn>=1.0.0
tqdm>=4.62.0
tensorboard>=2.9.0
pyyaml>=6.0
pillow>=9.0.0
```

---

## Key References

1. **Primary Paper**: 
   - Huang, H., Gong, B., & Sun, W. (2023). "A Deep-Learning-Based Graph Neural Network-Long-Short-Term Memory Model for Reservoir Simulation and Optimization With Varying Well Controls." SPE Journal, 28(06), 2898-2916. DOI: 10.2118/215842-PA

2. **Reservoir Simulator**:
   - https://github.com/mashadab/Reservoir-Simulator
   - Supports 2D single/two-phase flow with wells

3. **PyTorch Geometric Documentation**:
   - https://pytorch-geometric.readthedocs.io/
   - Graph neural network library

---

## Notes and Recommendations

### Scaling Considerations:
- **If training is too slow**: Reduce to 150 cases initially
- **If model doesn't converge**: Reduce model complexity or adjust learning rate
- **If predictions are poor**: 
  - Check data normalization
  - Verify graph construction
  - Increase training data
  - Tune loss weights (alpha, beta)

### Alternative Approaches:
- Could use simpler ML models (Random Forest, XGBoost) as baseline
- Could implement capacitance-resistance model (CRM) for comparison
- Could use physics-informed neural networks (PINN) instead of pure data-driven

### Extension Opportunities:
- Add optimization module (like paper's PSO approach)
- Implement uncertainty quantification
- Extend to 3D reservoirs
- Add more complex physics (compositional, thermal)

---

## Getting Started Command

```bash
# Quick start for Claude Code:
# 1. Clone reservoir simulator
# 2. Install dependencies  
# 3. Start with Phase 1 - reservoir model setup
# 4. Follow spec sequentially through all phases

# Begin implementation with:
python src/reservoir_model.py --setup
```

---

**End of Specification**

This document serves as a comprehensive blueprint for implementing a GNN-LSTM reservoir surrogate model. Follow the phases sequentially, validating each stage before proceeding to the next.
