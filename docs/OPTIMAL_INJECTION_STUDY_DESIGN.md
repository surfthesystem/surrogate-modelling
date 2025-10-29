# Optimal Water Injection Strategy Study Design

## Objective

**Find the optimal water injection strategy that maximizes oil recovery over 3 months (90 days) with a total injection constraint.**

---

## Research Findings: Typical Field Operations

### Water Injection Rates (Based on Literature)

For reservoirs with **100-500 mD permeability** (moderate permeability):

| Source | Rate Range | Notes |
|--------|-----------|-------|
| Standard operations | 700-2,500 STB/day/well | Typical field operations |
| Smaller operations | 500-1,000 STB/day/well | Conservative approach |
| Moderate operations | 1,000-2,000 STB/day/well | Most common |
| Aggressive operations | 2,000-3,000 STB/day/well | High permeability zones |

**Considerations**:
- Lower injection rates in low-permeability zones (< 100 mD)
- Higher rates possible in high-permeability zones (> 300 mD)
- Must consider reservoir pressure limits
- Formation fracture pressure typically limits max injection

### Oil Production Rates (Based on Literature)

For waterflood operations with **100-500 mD permeability**:

| Source | Rate Range | Notes |
|--------|-----------|-------|
| Typical producers | 100-350 STB/day | Most common range |
| Low-end producers | 50-100 STB/day | Lower permeability or depleted |
| High-end producers | 300-500 STB/day | High permeability, good pressure support |
| Economic limit | 10 STB/day | Shut-in threshold |
| Water cut limit | WOR > 80% | Typical abandonment criterion |

---

## Your Reservoir Characteristics

Based on `data/impes_input/permeability.txt` and Phase 1 configuration:

| Property | Value | Notes |
|----------|-------|-------|
| Grid size | 100 × 100 cells | 10,000 cells |
| Domain | 5 km × 5 km | 16,404 ft × 16,404 ft |
| Thickness | 50 ft | Uniform |
| Permeability | 50-500 mD | Mean ~120 mD, heterogeneous |
| Porosity | 0.15-0.16 | Mean ~0.155 |
| Initial pressure | 4500 psi | Well above bubble point |
| Bubble point | 502.5 psi | Undersaturated oil |
| Total wells | 58 | 40 producers + 18 injectors |

### Pore Volume Calculation

```
PV = Domain_area × Thickness × Porosity_avg
   = (5000m × 5000m) × 15.24m × 0.155
   = 5.815 × 10^7 m³
   = 365 million barrels
```

### Voidage Considerations

For 3-month operation (90 days):

**Total injection capacity** (18 injectors):
- Conservative: 18 × 700 × 90 = **1.13 million STB**
- Moderate: 18 × 1500 × 90 = **2.43 million STB**
- Aggressive: 18 × 2500 × 90 = **4.05 million STB**

**Percentage of pore volume**:
- Conservative: 0.3% PV injected
- Moderate: 0.7% PV injected
- Aggressive: 1.1% PV injected

This is reasonable for a 90-day waterflood study!

---

## Constraint: Maximum Daily Injection

Based on typical operations and reservoir size, set constraints:

### Total Injection Constraint Options

| Scenario | Total Daily Injection | Per Injector (18 wells) | Basis |
|----------|----------------------|-------------------------|-------|
| **Low** | 12,600 STB/day | 700 STB/day/well | Conservative |
| **Medium** | 27,000 STB/day | 1,500 STB/day/well | Standard |
| **High** | 45,000 STB/day | 2,500 STB/day/well | Aggressive |

**Recommendation**: Use **27,000 STB/day total** (medium scenario) as the constraint.

This gives flexibility to:
- Allocate more to high-perm zones
- Allocate less to low-perm zones
- Test different spatial distributions
- Stay within realistic operational limits

---

## LHS Study Design

### Objective Function

**Maximize**: Cumulative oil production over 90 days

**Subject to**: Total injection ≤ 27,000 STB/day

### Decision Variables (18 injectors)

Allow each injector to have independent rate, but constrained by total:

```python
# Individual injector rates
inj_rate[i] ∈ [200, 3000] STB/day  # Per injector bounds

# Global constraint
Σ(inj_rate[i]) ≤ 27,000 STB/day  # Total injection limit
```

### Implementation Strategy

Since LHS doesn't naturally handle sum constraints, use this approach:

1. **Generate 18 unconstrained samples** in [0, 1]
2. **Normalize to satisfy constraint**:
   ```python
   # Generate weights that sum to 1
   weights = lhs_sample / sum(lhs_sample)

   # Allocate total budget
   inj_rates = weights * 27000  # STB/day
   ```
3. **Clip to individual bounds**: [200, 3000] STB/day

This ensures:
- Total injection = exactly 27,000 STB/day
- Each injector gets realistic rate
- LHS stratification is maintained

### Producer Configuration

Keep producers at **BHP-constrained (Type 2)**:
- All 40 producers at 1000 psi BHP
- Allows production to respond to injection strategy
- Realistic field operation

Or vary producer BHP as additional parameter:
```python
producer_bhp ∈ [800, 1200] psi  # All producers same BHP
```

---

## LHS Parameter Space

### Option 1: Injection Rates Only (18D)

**Parameters**: 18 injector rates (individual control)

```python
PARAMETERS = {
    'inj_000_fraction': [0.01, 0.15],  # Fraction of total budget
    'inj_001_fraction': [0.01, 0.15],
    # ... 18 injectors total
}

# Post-process to satisfy constraint:
fractions = normalize(samples)  # Sum to 1
rates = fractions * 27000  # STB/day total
```

**Samples needed**: 90-180 (5-10× dimensions)

---

### Option 2: Zonal Control + Global (5D)

Simplify by grouping injectors geographically:

```python
PARAMETERS = {
    # Spatial allocation (fractions sum to 1)
    'south_zone_fraction': [0.15, 0.35],   # 6 injectors
    'central_zone_fraction': [0.30, 0.50], # 8 injectors
    'north_zone_fraction': [0.15, 0.35],   # 4 injectors

    # Producer constraint
    'producer_bhp': [900, 1100],  # psi

    # Simulation time
    'sim_days': [85, 95],  # Around 90 days
}
```

**Samples needed**: 25-50 (5-10× dimensions)

**Benefit**: Faster, easier to interpret results

---

### Option 3: Uniform + Perturbation (3D) - Recommended for Initial Study

Baseline uniform + random variations:

```python
PARAMETERS = {
    # Base injection rate (distributed uniformly)
    'base_injection_rate': [1000, 2000],  # STB/day/well

    # Random perturbation factor
    'perturbation_std': [0.1, 0.4],  # Relative std dev

    # Producer BHP
    'producer_bhp': [900, 1100],  # psi
}
```

**Process**:
1. Set base rate for all injectors
2. Add random perturbations (maintaining total ≤ 27,000)
3. Clip individual rates to [200, 3000]

**Samples needed**: 15-30 (5-10× dimensions)

**Benefit**: Very fast, good for initial exploration

---

## Recommendation

### Phase 1: Start Simple (Option 3)

Use **uniform + perturbation** approach:
- 30 LHS samples
- 3 parameters
- Total runtime: ~10-15 hours on Google VM (e2-standard-4)
- Easy to analyze results

### Phase 2: Detailed Optimization (Option 2)

Use **zonal control**:
- 50 LHS samples
- 5 parameters
- Test spatial allocation strategies
- Better optimization potential

### Phase 3: Full Control (Option 1)

Use **individual injector rates**:
- 100-200 LHS samples
- 18 parameters
- Complete flexibility
- Requires more computation

---

## Expected Results

### Metrics to Track

For each simulation, record:

1. **Cumulative oil production** (STB) - PRIMARY OBJECTIVE
2. **Oil recovery factor** (%)
3. **Water cut** (%) at end
4. **Average reservoir pressure** (psi)
5. **Sweep efficiency** (fraction of reservoir contacted)
6. **Individual well performance** (40 producers + 18 injectors)

### Visualization Plan

1. **Pressure evolution** (10 snapshots over 90 days)
2. **Saturation evolution** (10 snapshots over 90 days)
3. **Cumulative production per well** (interactive plot)
4. **Total recovery comparison** (all simulations on one plot)
5. **Sensitivity analysis** (tornado chart)

---

## Computational Estimates

### Per Simulation

- Grid: 100 × 100 = 10,000 cells
- Time steps: ~90 days (depends on dt)
- Wells: 58 wells
- **Estimated time**: 15-30 minutes per simulation (on e2-standard-4)

### Total LHS Study

| Samples | Time (serial) | Time (4-core parallel) |
|---------|---------------|------------------------|
| 30 | 7.5-15 hours | 2-4 hours |
| 50 | 12.5-25 hours | 3-6 hours |
| 100 | 25-50 hours | 6-12 hours |

**Note**: Can run in parallel on multi-core VM or multiple VMs

---

## Implementation Files

Will create:

1. **`run_optimal_injection_lhs.py`** - Main script
   - Generate LHS samples
   - Create input files for each sample
   - Run simulations
   - Extract results

2. **`plot_evolution.py`** - Evolution plots
   - Pressure snapshots
   - Saturation snapshots
   - Save as PNG/MP4

3. **`plot_production.py`** - Production analysis
   - Per-well cumulative plots (interactive HTML)
   - Total recovery comparison
   - Sensitivity analysis

4. **`utils/input_file_generator.py`** - Input file creation
   - Modify well rates
   - Set up all 58 wells
   - Handle constraints

5. **`README_GOOGLE_VM.md`** - Deployment guide
   - How to upload and run on Google VM
   - Batch submission
   - Results download

---

## Summary

**Goal**: Maximize oil recovery in 90 days with ≤ 27,000 STB/day total injection

**Approach**: Latin Hypercube Sampling with injection rate allocation

**Wells**: All 58 wells (40 producers + 18 injectors)

**Constraint**: Total injection ≤ 27,000 STB/day (realistic field operation)

**Expected outcome**: Optimal spatial allocation of injection that maximizes sweep and oil recovery

**Next step**: Implement the LHS framework and run the study!
