"""
Diagnose Permeability Field for IMPES Simulation
Check for zero or problematic permeability values
"""

import numpy as np
import matplotlib.pyplot as plt

# Load permeability field
perm_field = np.load('data/permeability_field.npy')  # mD

print("="*70)
print("PERMEABILITY FIELD DIAGNOSTIC")
print("="*70)

print(f"\nField Shape: {perm_field.shape}")
print(f"Total cells: {perm_field.size}")

print(f"\nStatistics:")
print(f"  Minimum: {perm_field.min():.6f} mD")
print(f"  Maximum: {perm_field.max():.6f} mD")
print(f"  Mean: {perm_field.mean():.6f} mD")
print(f"  Median: {np.median(perm_field):.6f} mD")
print(f"  Std Dev: {perm_field.std():.6f} mD")

print(f"\nPotential Problems:")

# Check for zero or negative values
zero_count = np.sum(perm_field == 0)
negative_count = np.sum(perm_field < 0)
print(f"  Zero values: {zero_count}")
print(f"  Negative values: {negative_count}")

# Check for very low values
very_low_count = np.sum(perm_field < 0.01)
low_count = np.sum(perm_field < 1.0)
print(f"  Values < 0.01 mD: {very_low_count}")
print(f"  Values < 1.0 mD: {low_count}")

# Check for NaN or Inf
nan_count = np.sum(np.isnan(perm_field))
inf_count = np.sum(np.isinf(perm_field))
print(f"  NaN values: {nan_count}")
print(f"  Inf values: {inf_count}")

# Distribution
print(f"\nDistribution:")
percentiles = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
for p in percentiles:
    val = np.percentile(perm_field, p)
    print(f"  {p:3d}th percentile: {val:10.4f} mD")

# Check after IMPES-style processing (from input_file_phase1.py line 97)
perm_processed = perm_field.copy()
perm_processed[perm_processed <= 1E-6] = 1E-16
print(f"\nAfter IMPES processing (values <= 1e-6 set to 1e-16):")
print(f"  Values modified: {np.sum(perm_field <= 1E-6)}")
print(f"  New minimum: {perm_processed.min():.2e} mD")

# Check if this could cause numerical issues
print(f"\nNumerical Stability Check:")
if perm_processed.min() < 1e-10:
    print(f"  WARNING: Minimum permeability ({perm_processed.min():.2e}) is very small")
    print(f"           This could cause numerical instability")
else:
    print(f"  OK: Minimum permeability is reasonable")

if perm_processed.max() / perm_processed.min() > 1e8:
    print(f"  WARNING: Permeability ratio ({perm_processed.max()/perm_processed.min():.2e}) is very large")
    print(f"           This could cause ill-conditioned matrices")
else:
    print(f"  OK: Permeability ratio is reasonable")

print("="*70)

# Simple histogram
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.hist(perm_field.flatten(), bins=50, edgecolor='black')
ax.set_xlabel('Permeability (mD)')
ax.set_ylabel('Frequency')
ax.set_title('Permeability Distribution')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/permeability_histogram.png', dpi=150)
print(f"\nHistogram saved to: results/permeability_histogram.png")
