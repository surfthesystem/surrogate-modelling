# Well Configuration Summary

Based on your Phase 1 configuration file: `simulator/input_file_phase1.py`

---

## Overview

**Total Wells: 15**
- **10 Producers** (Wells 0-9)
- **5 Injectors** (Wells 10-14)

---

## Producer Details (10 Wells)

All producers are **BHP-constrained (Type 2)** at **1000 psi**.

| Well ID | Name | X (ft) | Y (ft) | Type | BHP (psi) | Notes |
|---------|------|--------|--------|------|-----------|-------|
| 0 | PROD_000 | 960.04 | 2303.54 | 2 (BHP) | 1000 | Near SW corner |
| 1 | PROD_004 | 10430.73 | 933.22 | 2 (BHP) | 1000 | Near S edge |
| 2 | PROD_008 | 3008.45 | 4130.63 | 2 (BHP) | 1000 | West-central |
| 3 | PROD_012 | 13199.87 | 3976.69 | 2 (BHP) | 1000 | East-central |
| 4 | PROD_016 | 6413.32 | 6537.44 | 2 (BHP) | 1000 | Central |
| 5 | PROD_020 | 15690.18 | 5950.01 | 2 (BHP) | 1000 | Near E edge |
| 6 | PROD_024 | 8642.12 | 9696.44 | 2 (BHP) | 1000 | North-central |
| 7 | PROD_028 | 816.98 | 11599.21 | 2 (BHP) | 1000 | NW region |
| 8 | PROD_032 | 11207.92 | 12384.90 | 2 (BHP) | 1000 | NE region |
| 9 | PROD_036 | 3375.83 | 14459.76 | 2 (BHP) | 1000 | Near N edge |

**Configuration**:
- **Type 2**: BHP-constrained
- **BHP**: 1000 psi (fixed)
- **Production rate**: Calculated by simulator (varies with time)
- **Formula**: `q_prod = J × (P_reservoir - 1000)`

**Behavior**:
- As reservoir pressure increases (from injection), production rate increases
- As water breaks through, water cut increases
- Production continues until pressure drops below 1000 psi

---

## Injector Details (5 Wells)

All injectors are **Rate-constrained (Type 1)** at **500 STB/day**.

| Well ID | Name | X (ft) | Y (ft) | Type | Rate (STB/day) | Notes |
|---------|------|--------|--------|------|----------------|-------|
| 10 | INJ_000 | 7036.20 | 3535.69 | 1 (Rate) | 500 | South-central |
| 11 | INJ_004 | 8550.18 | 2569.95 | 1 (Rate) | 500 | South-central |
| 12 | INJ_008 | 8729.29 | 6419.55 | 1 (Rate) | 500 | Central |
| 13 | INJ_012 | 4572.10 | 3405.63 | 1 (Rate) | 500 | West-central |
| 14 | INJ_016 | 9815.16 | 13266.84 | 1 (Rate) | 500 | Near N edge |

**Configuration**:
- **Type 1**: Rate-constrained
- **Rate**: 500 STB/day (stock tank barrels per day) water injection
- **BHP**: Calculated by simulator (varies with time)
- **Formula**: `BHP = P_reservoir + q_inj / J`

**Behavior**:
- Injection rate is fixed at 500 STB/day
- BHP increases as reservoir pressure increases
- BHP may increase significantly if permeability is low or reservoir is pressured up

---

## Total Injection vs Production

### Rates:

**Total Injection**:
- 5 injectors × 500 STB/day = **2500 STB/day** (fixed)

**Total Production** (initial estimate):
- 10 producers × ~60 STB/day = **~600 STB/day** (varies)
- Depends on: reservoir pressure, permeability, productivity index

**Net Balance**:
- Initially: **+1900 STB/day** (more injection than production)
- Reservoir pressure will **increase** over time
- Production rates will **increase** as pressure builds
- Eventually reaches quasi-steady state

### Volumes (after 10 days of simulation):

**Injected**:
- 2500 STB/day × 10 days = **25,000 STB**

**Produced** (estimate):
- Varies based on pressure response
- Likely **10,000-15,000 STB** initially

**Net stored in reservoir**:
- **10,000-15,000 STB** causes pressure increase

---

## Spatial Distribution

### Domain:
- **Size**: 16,404 ft × 16,404 ft (5 km × 5 km)
- **Grid**: 100 × 100 cells
- **Cell size**: 164 ft × 164 ft

### Well Pattern:

```
     0    2000  4000  6000  8000  10000 12000 14000 16000 ft
  0  ┌─────────────────────────────────────────────────────┐
     │                                                      │
2000 │  PROD0                                               │
     │              INJ12                                   │
4000 │       PROD8        INJ10 INJ11                       │
     │                           PROD12                     │
6000 │                   PROD4     INJ8                     │
     │                                      PROD20          │
8000 │                                                      │
     │                       PROD24                         │
10000│                                                      │
     │  PROD28                                              │
12000│                         PROD32                       │
     │                                             INJ14    │
14000│           PROD36                                     │
     │                                                      │
16000└─────────────────────────────────────────────────────┘
```

**Pattern Type**: Somewhat scattered
- Injectors: Mostly south and central
- Producers: Distributed across entire domain
- Not a regular pattern (5-spot, line drive, etc.)

---

## Well Properties

### Common to All Wells:

| Property | Value | Units | Notes |
|----------|-------|-------|-------|
| Radius (rw) | 0.25 | ft | Small radius (3 inch diameter) |
| Skin factor | 0 | - | No formation damage |
| Direction | Vertical | - | All wells are vertical |
| Thickness | 50 | ft | Reservoir thickness |

### Productivity Index Calculation:

The simulator calculates productivity index using **Peaceman's well model**:

```
J = 0.00633 × (2π × k × kr × h) / (μ × B × (ln(r_eq/r_w) + S))
```

Where:
- k = permeability (varies by location)
- kr = relative permeability (varies with saturation)
- h = 50 ft (reservoir thickness)
- μ = viscosity (0.383 cp water, 2.47 cp oil)
- B = formation volume factor (1.012 water, 1.046 oil)
- r_eq = equivalent radius (calculated from grid block size)
- r_w = 0.25 ft (wellbore radius)
- S = 0 (skin factor)

**Typical J values**:
- For k = 100 mD, kr = 1.0: J ≈ 2-5 STB/day/psi
- For k = 500 mD, kr = 1.0: J ≈ 10-25 STB/day/psi

---

## For Latin Hypercube Sampling

### Current Configuration Summary:

```python
# Producers (10 wells)
producer_type = 2              # BHP-constrained
producer_bhp = 1000 psi       # Current value
producer_bhp_range = [800, 1200]  # Suggested LHS range

# Injectors (5 wells)
injector_type = 1             # Rate-constrained
injector_rate = 500 STB/day   # Current value (per well)
injector_rate_range = [200, 800]  # Suggested LHS range (per well)
```

### Recommended LHS Parameters:

#### **Individual Well Control** (Maximum flexibility):

```python
PARAMETERS = {
    # Injector rates (5 wells)
    'inj_000_rate': [200, 800],  # INJ_000
    'inj_004_rate': [200, 800],  # INJ_004
    'inj_008_rate': [200, 800],  # INJ_008
    'inj_012_rate': [200, 800],  # INJ_012
    'inj_016_rate': [200, 800],  # INJ_016

    # Producer BHPs (10 wells)
    'prod_000_bhp': [800, 1200],  # PROD_000
    'prod_004_bhp': [800, 1200],  # PROD_004
    # ... (all 10 producers)

    # Total: 15 parameters (5 injectors + 10 producers)
}
```

**Pros**: Maximum control, test asymmetric patterns
**Cons**: 15-dimensional space, need ~150-300 samples

#### **Group Control** (Recommended for initial study):

```python
PARAMETERS = {
    # All injectors same rate
    'injector_rate_all': [200, 800],      # Applied to all 5 injectors

    # All producers same BHP
    'producer_bhp_all': [800, 1200],      # Applied to all 10 producers

    # Reservoir uncertainty
    'perm_multiplier': [0.5, 2.0],        # Multiply existing perm field
    'initial_pressure': [4000, 5000],     # Initial reservoir pressure

    # Simulation time
    'simulation_time': [365, 1825],       # 1-5 years

    # Total: 5 parameters
}
```

**Pros**: Simple, manageable, only need ~25-50 samples
**Cons**: Less flexibility, symmetric injection pattern

#### **Regional Control** (Intermediate):

```python
PARAMETERS = {
    # South injectors (INJ_000, INJ_004, INJ_012)
    'south_injector_rate': [200, 800],

    # Central/North injectors (INJ_008, INJ_016)
    'north_injector_rate': [200, 800],

    # All producers same BHP
    'producer_bhp_all': [800, 1200],

    # Total: 3 parameters
}
```

**Pros**: Balance of control and simplicity
**Cons**: Need to define regions carefully

---

## Material Balance Check

With your current configuration:

**Input**:
- 5 injectors × 500 STB/day = 2500 STB/day

**Output**:
- 10 producers × (varies) STB/day

**Accumulation**:
- Net injection - Net production = Pressure increase

**Important**: With BHP-constrained producers, the system will **self-regulate**:
1. More injection → Higher pressure
2. Higher pressure → More production
3. Eventually reaches equilibrium

**Typical equilibrium**:
- Total production ≈ 1800-2200 STB/day
- Net accumulation ≈ 300-700 STB/day
- Pressure stabilizes at ~4600-4800 psi

---

## Visualization

To visualize well locations:

```python
import matplotlib.pyplot as plt
import numpy as np

# Well coordinates from input file
prod_x = [960.04, 10430.73, 3008.45, 13199.87, 6413.32,
          15690.18, 8642.12, 816.98, 11207.92, 3375.83]
prod_y = [2303.54, 933.22, 4130.63, 3976.69, 6537.44,
          5950.01, 9696.44, 11599.21, 12384.90, 14459.76]

inj_x = [7036.20, 8550.18, 8729.29, 4572.10, 9815.16]
inj_y = [3535.69, 2569.95, 6419.55, 3405.63, 13266.84]

plt.figure(figsize=(10, 10))
plt.scatter(prod_x, prod_y, s=200, c='blue', marker='o',
            label='Producers (10)', edgecolors='black', linewidths=2)
plt.scatter(inj_x, inj_y, s=200, c='red', marker='^',
            label='Injectors (5)', edgecolors='black', linewidths=2)

# Add well labels
for i, (x, y) in enumerate(zip(prod_x, prod_y)):
    plt.text(x+200, y+200, f'P{i}', fontsize=8)
for i, (x, y) in enumerate(zip(inj_x, inj_y)):
    plt.text(x+200, y+200, f'I{i}', fontsize=8)

plt.xlim(0, 16404)
plt.ylim(0, 16404)
plt.xlabel('X (ft)')
plt.ylabel('Y (ft)')
plt.title('Well Configuration\n10 Producers + 5 Injectors')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()
plt.savefig('well_locations.png', dpi=300)
plt.show()
```

---

## Summary

### Configuration:
- **15 total wells**: 10 producers + 5 injectors
- **Producers**: BHP-constrained @ 1000 psi (Type 2)
- **Injectors**: Rate-constrained @ 500 STB/day (Type 1)

### Key Features:
- ✅ Producers will **respond automatically** to injection rate changes
- ✅ Material balance is maintained through pressure response
- ✅ Good setup for Latin Hypercube Sampling studies

### For LHS Study:
- **Primary variables**: Injector rates (5 wells)
- **Secondary variables**: Producer BHPs (10 wells)
- **Response variables**: Oil recovery, water cut, pressure

### Recommendation:
Start with **grouped control** (all injectors same rate, all producers same BHP) to explore the parameter space efficiently with ~25-50 samples.

---

**File**: `simulator/input_file_phase1.py` lines 114-170
