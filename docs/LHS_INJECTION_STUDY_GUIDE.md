# Latin Hypercube Sampling for Injection Scenarios

Complete guide for running design of experiments on injection strategies.

---

## What is Latin Hypercube Sampling?

Latin Hypercube Sampling (LHS) is an efficient method to explore a multidimensional parameter space with fewer samples than full factorial design.

### Simple Example:

**Problem**: Test how injection rate and permeability affect oil recovery

**Traditional approach**: Full factorial
- Injection rates: [100, 200, 300, 400] bbl/day (4 values)
- Permeability: [50, 100, 150, 200] mD (4 values)
- Total runs: 4 × 4 = **16 simulations**

**LHS approach**:
- Same parameter ranges
- Total runs: **4 simulations** (or any number you choose)
- But **better coverage** than random sampling!

### How LHS Works:

1. **Divide each parameter into intervals**: Split the range into N equal probability intervals
2. **Sample once per interval**: Ensure each interval is sampled exactly once
3. **Randomize pairing**: Randomly pair values from different parameters

**Result**: Evenly distributed samples across the entire parameter space

### Visual Comparison:

```
Grid Sampling (16 points):          LHS (4 points):
┌─┬─┬─┬─┐                           ┌───────┐
├─┼─┼─┼─┤  All combinations         │   x   │  Stratified
├─┼─┼─┼─┤  Systematic               │ x     │  Random
└─┴─┴─┴─┘  Expensive                │     x │  Efficient
                                     │  x    │
                                     └───────┘
```

---

## How Your Simulator Handles Wells

### Key Concept: Well Constraint Types

Your simulator has **two operating modes** for each well:

#### **Type 1: Rate-Constrained** (Specify flow rate)

**Configuration**:
```python
well.type[i][0] = 1
well.constraint[i][0] = 500 * 5.61  # 500 bbl/day water injection
```

**What Happens**:
- **You control**: Flow rate (fixed)
- **Simulator calculates**: Bottom-hole pressure (BHP)
- **Physics**: BHP adjusts to maintain the specified rate

**Equation**:
```
q = J × (P_reservoir - P_wf)  →  P_wf = P_reservoir - q/J
```

Where:
- q = flow rate (specified by you)
- J = productivity index (calculated from rock/fluid properties)
- P_reservoir = pressure at grid block
- P_wf = wellbore flowing pressure (calculated)

#### **Type 2: BHP-Constrained** (Specify pressure)

**Configuration**:
```python
well.type[i][0] = 2
well.constraint[i][0] = 300  # 300 psi BHP
```

**What Happens**:
- **You control**: Bottom-hole pressure (fixed)
- **Simulator calculates**: Flow rate (varies with time)
- **Physics**: Flow rate responds to reservoir pressure changes

**Equation**:
```
q = J × (P_reservoir - P_wf)
```

Where:
- P_wf = wellbore flowing pressure (specified by you)
- q = flow rate (calculated by simulator)

---

## Will Different Injection Rates Show Different Production Rates?

### Short Answer: **YES, if you set it up correctly!**

### The Key: Use the Right Constraint Combinations

#### **Scenario A: Rate/Rate** (Both Fixed)

```python
Injector: Type 1, Rate = 500 bbl/day  (fixed)
Producer: Type 1, Rate = -300 bbl/day (fixed)
```

**Result**:
- Injection: Always 500 bbl/day (unchanged)
- Production: Always 300 bbl/day (unchanged)
- Reservoir pressure: Increases over time (500 > 300)

**For LHS**: Changing injection rate to 700 bbl/day:
- Injection: 700 bbl/day
- Production: **Still 300 bbl/day** (fixed!)
- Difference: Pressure builds up faster

**Problem**: Production doesn't respond to injection changes. Not realistic.

#### **Scenario B: Rate/BHP** (Recommended!)

```python
Injector: Type 1, Rate = 500 bbl/day  (fixed)
Producer: Type 2, BHP = 300 psi       (fixed)
```

**Result**:
- Injection: 500 bbl/day (fixed)
- Producer BHP: 300 psi (fixed)
- Production rate: **VARIES** based on reservoir pressure
- Reservoir pressure: Increases over time

**For LHS**: Changing injection rate to 700 bbl/day:
- Injection: 700 bbl/day
- Reservoir pressure: Rises faster (more water injected)
- Pressure at producer: Higher
- Production rate: **INCREASES AUTOMATICALLY** (q = J × ΔP)

**Benefit**: Production responds realistically to injection strategy!

#### **Scenario C: BHP/BHP** (Most Realistic)

```python
Injector: Type 2, BHP = 800 psi  (fixed)
Producer: Type 2, BHP = 300 psi  (fixed)
```

**Result**:
- Injector BHP: 800 psi (fixed)
- Producer BHP: 300 psi (fixed)
- Injection rate: Varies (depends on reservoir pressure)
- Production rate: Varies (depends on reservoir pressure)

**Benefit**: Both wells respond to reservoir conditions. Most realistic field operations.

---

## Understanding the Physics: Pressure-Rate Coupling

### Time Evolution Example:

**Setup**:
- Injector: 500 bbl/day (Type 1)
- Producer: 300 psi BHP (Type 2)
- Initial reservoir pressure: 500 psi

**Time = 0 days**:
```
Reservoir pressure: 500 psi (uniform)
Producer drawdown: 500 - 300 = 200 psi
Production rate: J × 200 = 60 bbl/day (example)
```

**Time = 100 days**:
```
Reservoir pressure: 550 psi (increased from injection)
Producer drawdown: 550 - 300 = 250 psi
Production rate: J × 250 = 75 bbl/day (increased!)
```

**Time = 500 days**:
```
Reservoir pressure: 600 psi (water reached producer)
Producer drawdown: 600 - 300 = 300 psi
Production rate: J × 300 = 90 bbl/day (maximum)
Water cut: 60% (producing water + oil)
```

### Key Insight:

With **BHP-constrained producers**, production rate **automatically responds** to:
1. Reservoir pressure changes (from injection)
2. Fluid mobility changes (oil vs water)
3. Permeability variations (reservoir heterogeneity)

This is **realistic field behavior**!

---

## Parameters for LHS Study

### 1. Well Parameters (Primary Focus)

| Parameter | Symbol | Type | Range | Units | Impact |
|-----------|--------|------|-------|-------|--------|
| **Injector 1 rate** | q_inj1 | Type 1 | 100-600 | bbl/day | Water injection |
| **Injector 2 rate** | q_inj2 | Type 1 | 100-600 | bbl/day | Water injection |
| **Producer BHP** | P_wf | Type 2 | 250-400 | psi | Drawdown control |
| **Injector BHP** | P_inj | Type 2 | 600-1000 | psi | Injection pressure |

**Note**: For injection study, recommend **Type 1 (rate) for injectors**, **Type 2 (BHP) for producers**.

### 2. Reservoir Properties (Secondary)

| Parameter | Symbol | Range | Units | Impact |
|-----------|--------|-------|-------|--------|
| Permeability | k | 50-300 | mD | Flow capacity |
| Porosity | φ | 0.18-0.28 | - | Storage capacity |
| Initial pressure | P₀ | 400-600 | psi | Energy |
| Anisotropy ratio | k_y/k_x | 0.1-0.5 | - | Flow direction |

### 3. Fluid Properties

| Parameter | Symbol | Range | Units | Impact |
|-----------|--------|-------|-------|--------|
| Oil viscosity | μ_o | 2-10 | cp | Flow resistance |
| Water viscosity | μ_w | 0.3-1.0 | cp | Flow resistance |
| Oil compressibility | c_o | 3E-6-1E-5 | psi⁻¹ | Pressure response |

### 4. Rock-Fluid Properties

| Parameter | Symbol | Range | Units | Impact |
|-----------|--------|-------|-------|--------|
| Residual oil sat. | S_or | 0.25-0.4 | - | Trapped oil |
| Residual water sat. | S_wr | 0.15-0.25 | - | Irreducible water |
| Water kr exponent | n_w | 2.0-4.0 | - | Water mobility |
| Oil kr exponent | n_o | 2.0-4.0 | - | Oil mobility |

### 5. Simulation Parameters

| Parameter | Symbol | Range | Units | Impact |
|-----------|--------|-------|-------|--------|
| Simulation time | t_final | 365-3650 | days | 1-10 years |
| Time step | dt | 0.5-5 | days | Stability |

---

## Recommended LHS Designs

### Design 1: Simple Injection Rate Study (Beginner)

**Objective**: How do injection rates affect oil recovery?

**Fixed Parameters**:
- Producers: BHP = 300 psi (Type 2)
- Reservoir: Use Phase 1 generated field
- Fluids: Fixed properties

**Variable Parameters** (5 dimensions):
```python
PARAMETERS = {
    'injector1_rate': [200, 600],      # bbl/day
    'injector2_rate': [200, 600],      # bbl/day
    'producer_bhp': [250, 400],        # psi
    'simulation_time': [730, 1825],    # 2-5 years
    'oil_viscosity': [3, 8],           # cp
}
```

**Number of samples**: 20-30
**Simulation time**: ~1-2 hours (depending on hardware)

**Analysis**: Plot recovery factor vs. injection rates

---

### Design 2: Comprehensive Injection Study (Intermediate)

**Objective**: Full sensitivity analysis

**Variable Parameters** (10 dimensions):
```python
PARAMETERS = {
    # Wells
    'injector1_rate': [200, 600],
    'injector2_rate': [200, 600],
    'producer_bhp': [250, 400],

    # Reservoir
    'perm_multiplier': [0.5, 2.0],     # Multiply existing field
    'poro_multiplier': [0.9, 1.1],     # Multiply existing field
    'initial_pressure': [450, 550],

    # Fluids
    'oil_viscosity': [3, 10],
    'Sor': [0.25, 0.4],

    # Simulation
    'simulation_time': [730, 1825],
    'time_step': [0.5, 2.0],
}
```

**Number of samples**: 50-100
**Simulation time**: ~5-10 hours

**Analysis**:
- Sensitivity analysis (Sobol indices)
- Response surfaces
- Optimization

---

### Design 3: Well Placement Study (Advanced)

**Objective**: Optimize well locations AND injection rates

**Variable Parameters** (12 dimensions):
```python
PARAMETERS = {
    # Injector 1
    'inj1_x': [1000, 5000],           # ft
    'inj1_y': [1000, 5000],           # ft
    'inj1_rate': [200, 600],          # bbl/day

    # Injector 2
    'inj2_x': [7000, 11000],          # ft
    'inj2_y': [1000, 5000],           # ft
    'inj2_rate': [200, 600],          # bbl/day

    # Producers (5 wells)
    'prod_bhp_mean': [300, 400],      # psi (average)
    'prod_bhp_std': [10, 50],         # psi (variation)

    # Reservoir
    'perm_multiplier': [0.7, 1.5],
    'oil_viscosity': [3, 8],

    # Simulation
    'simulation_time': [1095, 2190],  # 3-6 years
}
```

**Number of samples**: 100-200
**Simulation time**: ~10-20 hours
**Analysis**: Multi-objective optimization (recovery vs. cost)

---

## Implementation Steps

### Step 1: Install Required Packages

```bash
pip install numpy scipy pandas matplotlib seaborn
```

### Step 2: Modify Input File Template

Create a function to programmatically modify your input file:

```python
def modify_input_file(params, template='simulator/input_file_phase1.py'):
    """Modify input file with LHS parameters."""

    with open(template, 'r') as f:
        content = f.read()

    # Example modifications:

    # Injection rates (convert bbl/day to scf/day)
    inj1_scf = params['injector1_rate'] * 5.61
    inj2_scf = params['injector2_rate'] * 5.61

    # Find the well.constraint line and replace
    # This depends on your specific input file format!

    # For example:
    # Original: well.constraint = [[-120*5.61], [-90*5.61], ...]
    # Replace with LHS values

    # Producer BHP
    prod_bhp = params['producer_bhp']
    # Replace all producer BHP values

    # Simulation time
    t_final = params['simulation_time']
    content = content.replace(
        'tfinal = 10',
        f'tfinal = {t_final}'
    )

    return content
```

### Step 3: Run LHS Study

```bash
python run_lhs_study.py
```

This will:
1. Generate LHS design (saves to `lhs_results/lhs_design.csv`)
2. Create modified input files for each sample
3. Run simulations
4. Extract results
5. Create summary CSV for analysis

### Step 4: Analyze Results

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load results
df = pd.read_csv('lhs_results/lhs_study_summary.csv')

# Plot: Recovery factor vs. Injection rate
plt.figure(figsize=(10, 6))
plt.scatter(
    df['injector1_rate'] + df['injector2_rate'],
    df['recovery_factor'],
    c=df['producer_bhp'],
    cmap='viridis',
    s=100
)
plt.colorbar(label='Producer BHP (psi)')
plt.xlabel('Total Injection Rate (bbl/day)')
plt.ylabel('Recovery Factor')
plt.title('Oil Recovery vs. Injection Strategy')
plt.grid(True, alpha=0.3)
plt.savefig('recovery_vs_injection.png', dpi=300)
plt.show()

# Sensitivity analysis
from scipy.stats import spearmanr

correlations = {}
for param in ['injector1_rate', 'injector2_rate', 'producer_bhp']:
    corr, _ = spearmanr(df[param], df['recovery_factor'])
    correlations[param] = corr

print("\nSensitivity Analysis (Spearman correlation):")
for param, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True):
    print(f"  {param}: {corr:.3f}")
```

---

## Expected Outputs

### 1. LHS Design Matrix

File: `lhs_results/lhs_design.csv`

```
sample_id,injector1_rate,injector2_rate,producer_bhp,simulation_time
0,234.5,487.2,324.1,1456
1,512.3,156.8,278.9,2103
2,389.1,621.4,391.2,945
...
```

### 2. Simulation Results

File: `lhs_results/lhs_study_summary.csv`

```
sample_id,injector1_rate,injector2_rate,producer_bhp,recovery_factor,water_cut,final_pressure
0,234.5,487.2,324.1,0.342,0.65,523.4
1,512.3,156.8,278.9,0.289,0.52,487.2
2,389.1,621.4,391.2,0.418,0.78,556.1
...
```

### 3. Visualizations

- Recovery factor vs. injection rate
- Water cut evolution
- Sensitivity tornado chart
- Response surfaces (2D slices)

---

## Key Metrics to Track

### Production Metrics:
1. **Recovery Factor** = Cumulative oil produced / Original oil in place
2. **Water Cut** = Water production / Total fluid production
3. **Oil Rate** = Instantaneous oil production rate (bbl/day)
4. **Cumulative Oil** = Total oil produced (STB)

### Injection Metrics:
5. **Injectivity** = Injection rate / (BHP - Reservoir pressure)
6. **Voidage Replacement** = Total injection / Total production
7. **Sweep Efficiency** = Contacted reservoir volume / Total volume

### Pressure Metrics:
8. **Average Pressure** = Mean reservoir pressure
9. **Pressure Support** = Maintenance of pressure over time
10. **Pressure Gradient** = Spatial pressure variation

### Economic Metrics:
11. **NPV** = Net present value (if you add costs)
12. **Water Handling Cost** = Cost of produced water disposal
13. **Recovery per Injection** = Oil recovered / Water injected

---

## Tips for Success

### 1. Start Small
- Begin with 10-20 samples
- Use 2-3 parameters
- Short simulation times (1 year)
- Validate results before scaling up

### 2. Check Material Balance
After each run, verify:
```python
Volume_in - Volume_out ≈ Change_in_pore_volume
```

If material balance fails, check:
- Time step too large
- Boundary condition issues
- Numerical instability

### 3. Use Dimensional Analysis
Non-dimensionalize parameters:
```python
# Pore volumes injected
PVI = (Total_injection * dt) / Pore_volume

# Mobility ratio
M = (k_rw / μ_w) / (k_ro / μ_o)

# Dimensionless time
t_D = k * t / (φ * μ * c * L²)
```

### 4. Validate Against Known Cases
Before LHS study:
- Run baseline case (no injection)
- Run symmetric case (equal injection/production)
- Compare to analytical solutions (if available)

### 5. Save Everything
For each simulation:
```python
- Input parameters (JSON)
- Simulator log file
- Final .npz result file
- Key metrics (CSV)
- Plots (PNG)
```

---

## Troubleshooting

### Issue: All results look the same

**Possible causes**:
- Parameter ranges too narrow
- Parameters not actually changing in input file
- Simulation time too short to see effects

**Solution**:
- Check that input files are actually different
- Increase parameter range
- Run longer simulations

### Issue: Some simulations fail

**Possible causes**:
- Extreme parameter combinations
- Numerical instability
- Memory issues

**Solution**:
- Add parameter constraints (e.g., min/max ratios)
- Reduce time step for unstable cases
- Implement automatic time step adaptation

### Issue: Results don't make physical sense

**Possible causes**:
- Material balance violation
- Incorrect unit conversions
- Wrong well constraint type

**Solution**:
- Check material balance each time step
- Verify all unit conversions (bbl/day ↔ scf/day)
- Print diagnostics for pressure/saturation bounds

---

## Next Steps: Surrogate Modeling

After LHS study, you'll have:
- N simulation runs
- Input parameters (X)
- Output metrics (Y)

Use this data to train surrogate models:

### 1. Polynomial Response Surface
```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
model = LinearRegression().fit(X_poly, y)
```

### 2. Gaussian Process
```python
from sklearn.gaussian_process import GaussianProcessRegressor

gp = GaussianProcessRegressor()
gp.fit(X, y)
```

### 3. Neural Network
```python
from sklearn.neural_network import MLPRegressor

nn = MLPRegressor(hidden_layers=(50, 50))
nn.fit(X, y)
```

These surrogates can then **replace expensive simulations** for optimization!

---

## Summary

### Latin Hypercube Sampling:
- **Efficient** parameter space exploration
- **Stratified** sampling ensures coverage
- **Flexible** number of samples

### Well Constraints:
- **Type 1 (Rate)**: Fix flow rate, calculate BHP
- **Type 2 (BHP)**: Fix BHP, calculate flow rate
- **Recommended**: Type 1 injectors + Type 2 producers

### Key Insight:
With **BHP-constrained producers**, production rates **automatically respond** to injection rate changes through reservoir pressure coupling. This is the realistic field behavior you want!

### Parameters to Vary:
1. **Injector rates** (primary control variable)
2. **Producer BHPs** (operational constraint)
3. **Reservoir properties** (uncertainty)
4. **Fluid properties** (uncertainty)
5. **Simulation time** (forecast horizon)

### Workflow:
1. Generate LHS design
2. Modify input files
3. Run simulations
4. Extract metrics
5. Analyze sensitivity
6. Build surrogate model
7. Optimize!

---

**Good luck with your injection scenario study!**
