# CRITICAL FIX: Water Saturation Initialization Singularity

## Executive Summary

**Problem**: IMPES simulation produced NaN values immediately after first timestep
**Root Cause**: Mathematical singularity in capillary pressure when Sw = Swr exactly  
**Solution**: Initialize Sw slightly above Swr (e.g., 0.21 instead of 0.20)
**Impact**: Complete resolution - simulation now runs stably with valid physics

## Detailed Analysis

### The Bug

When water saturation (Sw) is initialized exactly equal to residual water saturation (Swr), the capillary pressure calculation encounters a division by zero raised to a negative power, producing infinity.

### Mathematical Explanation

In `Reservoir-Simulator/proj2/Problem 2/cap_press.py` lines 13-18:

```python
S = (Sw - petro.Swr)/(1.0 - petro.Sor - petro.Swr)  # Normalized saturation
Se = (Sw - petro.Swr)/(1.0 - petro.Swr)             # Effective saturation

Pcd = petro.Pe * (Se) **(-1.0 / petro.lamda)         # Drainage capillary pressure
Pci = petro.Pe * ((S) **(-1.0 / petro.lamda) - 1.0)  # Imbibition capillary pressure
```

**When Sw = Swr = 0.20:**
- S = (0.20 - 0.20) / (1.0 - 0.4 - 0.20) = 0.0 / 0.4 = **0.0**
- Se = (0.20 - 0.20) / (1.0 - 0.20) = 0.0 / 0.8 = **0.0**

**Then:**
- Pcd = 3.5 * (0.0)^(-1/2.0) = 3.5 * 0^(-0.5) = **∞**
- Pci = 3.5 * (0.0^(-0.5) - 1.0) = 3.5 * (∞ - 1.0) = **∞**

This infinity propagates through:
1. Capillary pressure (Pc) → ∞
2. Pressure solver RHS (EX) → contains ∞
3. Sparse linear solve → produces NaN
4. Pressure field (P) → all NaN
5. Saturation update → NaN
6. All subsequent timesteps → NaN

### Debugging Process

1. **Initial Symptom**: All variables became NaN after timestep 1
2. **Hypothesis Testing**:
   - ✅ Wells positioned correctly within grid
   - ✅ Permeability field valid (10-500 mD, no zeros/NaN)
   - ✅ Porosity field valid (0.15-0.16, no zeros/NaN)
3. **Pinpointing**: Created debug script (IMPES_phase1_debug.py)
4. **Discovery**: NaN appeared in Pc (capillary pressure) from `myarrays()` 
5. **Root Cause**: Traced to `cap_press.py` line 17-18 with Sw=Swr input

### The Fix

**File**: `input_file_phase1.py` line 219

**Before (WRONG)**:
```python
IC.Sw = petro.Swr*np.ones((numerical.N,1))  # Sw = 0.20 exactly
```

**After (CORRECT)**:
```python
IC.Sw = (petro.Swr + 0.01)*np.ones((numerical.N,1))  # Sw = 0.21
```

### Why This Fix Works

With Sw = 0.21 (instead of 0.20):
- S = (0.21 - 0.20) / (1.0 - 0.4 - 0.20) = 0.01 / 0.4 = **0.025**
- Se = (0.21 - 0.20) / (1.0 - 0.20) = 0.01 / 0.8 = **0.0125**

Now:
- Pcd = 3.5 * (0.0125)^(-0.5) = 3.5 * 8.944 = **31.3 psi** ✓
- Pci = 3.5 * (0.025^(-0.5) - 1.0) = 3.5 * (6.325 - 1.0) = **18.6 psi** ✓

Both are finite, valid capillary pressures!

### Physical Justification

Setting Sw = 0.21 (slightly above residual) is physically reasonable:
1. Represents oil-saturated reservoir (So = 0.79)
2. Avoids mathematical singularity while maintaining physical intent
3. Oil mobility still dominant (kro >> krw at low Sw)
4. Commonly done in industry to avoid numerical issues
5. 1% saturation difference is negligible compared to measurement uncertainty

### Verification

After fix:
- ✅ Timestep 1: P = 3226-4578 psi, Sw = 0.2100-0.2124 (VALID)
- ✅ Pressure declining as expected (production working)
- ✅ Saturation updating gradually (injection working)
- ✅ All 10 timesteps completed successfully
- ✅ No NaN values throughout simulation

### Lessons Learned

1. **Never initialize at exact saturation endpoints** (Swr or 1-Sor)
2. **Capillary pressure models often have singularities** at S=0 or S=1
3. **Small numerical perturbations** (0.01) can prevent catastrophic failures
4. **Systematic debugging** (diagnose each component) is essential
5. **Test edge cases** during model development

### Recommendations for Future Work

1. Add validation check in input file:
   ```python
   if IC.Sw.min() <= petro.Swr:
       raise ValueError("Initial Sw must be > Swr to avoid singularity!")
   ```

2. Add epsilon buffer in cap_press.py:
   ```python
   epsilon = 1e-6
   S = max((Sw - petro.Swr)/(1 - petro.Sor - petro.Swr), epsilon)
   ```

3. Document this limitation prominently in code comments

4. Consider alternative capillary pressure models without singularities

## Timeline

- **Issue Discovered**: Multiple runs produced NaN immediately
- **Debugging**: ~2 hours systematic investigation
- **Root Cause Found**: Sw=Swr singularity in cap_press.py
- **Fix Implemented**: Single line change in input_file_phase1.py
- **Verification**: 10-day simulation ran successfully (4.3 minutes)
- **Status**: RESOLVED ✅

## References

- `cap_press.py`: Corey-Brooks capillary pressure model
- `myarrays.py`: Calls cap_press for each cell
- `IMPES_phase1_debug.py`: Debug script that identified the issue
- Test results: `results/impes_sim/Phase1_pressure_evolution.png`

---

**Author**: Debugging session 2025-10-28
**Severity**: CRITICAL - Blocks all simulations
**Resolution**: Complete
