# Well Selection Explanation

## The Discrepancy

You noticed that:
- **Visualization** shows: **40 producers + 18 injectors = 58 wells**
- **Current simulator** uses: **10 producers + 5 injectors = 15 wells**

## What Happened?

### Phase 1 Generation Created 58 Wells

When you ran `src/reservoir_model.py`, it generated:

**File**: `data/well_locations.csv`
- **40 producers** (PROD_000 to PROD_039)
- **18 injectors** (INJ_000 to INJ_017)
- **Total: 58 wells**

These wells were placed uniformly across the 5km × 5km domain.

### But Only 15 Wells Were Selected for Simulation

The file `data/impes_input/selected_wells.csv` contains only:
- **10 producers**: PROD_000, PROD_004, PROD_008, PROD_012, PROD_016, PROD_020, PROD_024, PROD_028, PROD_032, PROD_036
- **5 injectors**: INJ_000, INJ_004, INJ_008, INJ_012, INJ_016

**Pattern**: Every 4th well was selected (wells ending in 000, 004, 008, 012, 016, etc.)

### Why?

This was likely done to:
1. **Reduce computational cost** for initial testing
2. **Maintain spatial coverage** (selecting every 4th well still covers the domain)
3. **Create a manageable test case** (15 wells vs 58 wells)

---

## Visualization vs Simulation Mismatch

### The Visualization (`results/reservoir_visualization.png`)

This was created by `src/reservoir_model.py` and shows **ALL 58 generated wells**:
- Red circles: 40 producers
- Cyan triangles: 18 injectors

### The Simulation (`simulator/input_file_phase1.py`)

This uses only **15 selected wells** from `selected_wells.csv`:
- 10 producers (every 4th producer)
- 5 injectors (every 4th injector)

---

## Which Wells Are You Currently Simulating?

Based on `simulator/input_file_phase1.py` lines 116-170:

### Producers (10 wells):

| Index | Well ID | X (m) | Y (m) | X (ft) | Y (ft) |
|-------|---------|-------|-------|--------|--------|
| 0 | PROD_000 | 292.6 | 702.1 | 960.0 | 2303.5 |
| 1 | PROD_004 | 3179.3 | 284.4 | 10430.7 | 933.2 |
| 2 | PROD_008 | 917.0 | 1259.0 | 3008.4 | 4130.6 |
| 3 | PROD_012 | 4023.3 | 1212.1 | 13199.9 | 3976.7 |
| 4 | PROD_016 | 1954.8 | 1992.6 | 6413.3 | 6537.4 |
| 5 | PROD_020 | 4782.4 | 1813.6 | 15690.2 | 5950.0 |
| 6 | PROD_024 | 2634.1 | 2955.5 | 8642.1 | 9696.4 |
| 7 | PROD_028 | 249.0 | 3535.4 | 817.0 | 11599.2 |
| 8 | PROD_032 | 3416.2 | 3774.9 | 11207.9 | 12384.9 |
| 9 | PROD_036 | 1029.0 | 4407.3 | 3375.8 | 14459.8 |

### Injectors (5 wells):

| Index | Well ID | X (m) | Y (m) | X (ft) | Y (ft) |
|-------|---------|-------|-------|--------|--------|
| 10 | INJ_000 | 2144.6 | 1077.7 | 7036.2 | 3535.7 |
| 11 | INJ_004 | 2606.1 | 783.3 | 8550.2 | 2570.0 |
| 12 | INJ_008 | 2660.7 | 1956.7 | 8729.3 | 6419.6 |
| 13 | INJ_012 | 1393.6 | 1038.0 | 4572.1 | 3405.6 |
| 14 | INJ_016 | 2991.7 | 4043.7 | 9815.2 | 13266.8 |

---

## Options Going Forward

### Option 1: Keep Current 15 Wells (Recommended for Testing)

**Pros**:
- ✅ Faster simulations (~4x faster)
- ✅ Easier to debug
- ✅ Good for initial LHS studies
- ✅ Still reasonable spatial coverage

**Cons**:
- ❌ Less detailed response
- ❌ Coarser well spacing
- ❌ May miss local heterogeneity effects

**Best for**: Initial testing, parameter studies, surrogate model development

---

### Option 2: Use All 58 Wells (Full Configuration)

**Pros**:
- ✅ More detailed reservoir response
- ✅ Better spatial coverage
- ✅ Matches your visualization
- ✅ More realistic field case

**Cons**:
- ❌ ~4x slower simulations
- ❌ More complex well management
- ❌ Higher computational cost for LHS

**Best for**: Final production runs, detailed forecasts

---

### Option 3: Custom Selection

Pick specific wells based on:
- Permeability distribution (target high/low perm areas)
- Porosity patterns
- Strategic locations for waterflood
- Economic considerations

---

## How to Switch to 58 Wells

If you want to use all 58 wells, you need to modify `simulator/input_file_phase1.py`:

### Step 1: Read All Wells

Instead of using `selected_wells.csv`, read from `well_locations.csv`:

```python
# Read all wells (not just selected ones)
wells_df = pd.read_csv(base_path + "../well_locations.csv")

# Separate producers and injectors
producers = wells_df[wells_df['well_type'] == 'producer']
injectors = wells_df[wells_df['well_type'] == 'injector']

# Convert to feet
producers['x_ft'] = producers['x_m'] * 3.28084
producers['y_ft'] = producers['y_m'] * 3.28084
injectors['x_ft'] = injectors['x_m'] * 3.28084
injectors['y_ft'] = injectors['y_m'] * 3.28084
```

### Step 2: Update Well Arrays

```python
# Producers (40 wells)
well.x_prod = [[x] for x in producers['x_ft'].values]
well.y_prod = [[y] for y in producers['y_ft'].values]

# Injectors (18 wells)
well.x_inj = [[x] for x in injectors['x_ft'].values]
well.y_inj = [[y] for y in injectors['y_ft'].values]

# Combine
well.x = well.x_prod + well.x_inj
well.y = well.y_prod + well.y_inj

# Well types: 40 producers (Type 2) + 18 injectors (Type 1)
well.type = [[2]]*40 + [[1]]*18

# Constraints
well.constraint = [[1000.0]]*40 + [[500.0]]*18  # 40 BHP + 18 rate

# Other properties
well.rw = [[0.25]]*58
well.skin = [[0]]*58
well.direction = [['v']]*58
```

### Step 3: Update Print Statements

```python
print(f"\nWells: {len(well.x)} total")
print(f"  Producers: 40 (BHP control @ 1000 psi)")
print(f"  Injectors: 18 (rate control @ 500 STB/day)")
```

---

## Material Balance Impact

### Current (15 wells):
- **Injection**: 5 × 500 = 2,500 STB/day
- **Production**: ~10 × 180 = ~1,800 STB/day (at equilibrium)
- **Net**: +700 STB/day → Pressure increases

### Full (58 wells):
- **Injection**: 18 × 500 = 9,000 STB/day
- **Production**: ~40 × 180 = ~7,200 STB/day (at equilibrium)
- **Net**: +1,800 STB/day → Pressure increases faster

**Note**: With 58 wells, you may need to adjust rates to maintain reasonable pressure:
- Option A: Reduce injector rates to 200-300 STB/day each
- Option B: Increase producer drawdown (lower BHP to 800-900 psi)
- Option C: Use fewer active injectors (turn some off)

---

## Recommendation

### For Your LHS Study:

**Start with 15 wells** for these reasons:

1. **Speed**: Run 50-100 LHS samples in reasonable time
2. **Learning**: Understand system behavior first
3. **Development**: Test your LHS script and surrogate model
4. **Economics**: Cheaper computational cost

### Later:

**Scale up to 58 wells** for:
- Final validation runs
- Detailed sensitivity analysis
- Production forecasts
- Real field application

---

## Summary

| Aspect | Current (15 wells) | Full (58 wells) |
|--------|-------------------|-----------------|
| **Producers** | 10 | 40 |
| **Injectors** | 5 | 18 |
| **Total** | 15 | 58 |
| **Spacing** | Every 4th well | All wells |
| **Simulation time** | Baseline | ~4x longer |
| **Coverage** | Good | Excellent |
| **Use case** | Testing, LHS | Production |

**Your visualization shows all 58 wells that were generated in Phase 1, but your simulator is currently configured to use only 15 selected wells for faster computation.**

This is actually a **good strategy** for initial development and testing!

---

## Action Items

1. **Keep 15 wells** for now (already configured)
2. Run your LHS study with 15 wells
3. Develop and test your surrogate model
4. **Later**: Scale up to 58 wells for final validation

Or, if you want to use all 58 wells immediately, I can help you modify the input file.

---

**Files to check**:
- All 58 wells: `data/well_locations.csv` (generated by Phase 1)
- Selected 15 wells: `data/impes_input/selected_wells.csv` (used by simulator)
- Current config: `simulator/input_file_phase1.py` lines 113-170
