"""
Convert Phase 1 Data to IMPES Simulator Format

Converts NumPy arrays and well locations to text files compatible with
the validated Reservoir-Simulator IMPES code.

Conversions:
- Permeability: .npy -> tab-separated text (mD units)
- Porosity: .npy -> tab-separated text (fraction)
- Depth: Generate flat reservoir at -7500 ft
- Well locations: meters -> feet (multiply by 3.28084)
"""

import numpy as np
import pandas as pd
import os

print("="*70)
print("PHASE 1 DATA CONVERSION FOR IMPES SIMULATOR")
print("="*70)

# Load Phase 1 data
print("\n1. Loading Phase 1 generated data...")
perm_field = np.load('data/permeability_field.npy')  # mD, shape (100, 100)
poro_field = np.load('data/porosity_field.npy')      # fraction, shape (100, 100)
wells_df = pd.read_csv('data/well_locations.csv')

print(f"   Permeability: {perm_field.shape}, range {perm_field.min():.2f}-{perm_field.max():.2f} mD")
print(f"   Porosity: {poro_field.shape}, range {poro_field.min():.4f}-{poro_field.max():.4f}")
print(f"   Wells: {len(wells_df)} total ({len(wells_df[wells_df['well_type']=='producer'])} producers, "
      f"{len(wells_df[wells_df['well_type']=='injector'])} injectors)")

# Create output directory
output_dir = 'data/impes_input'
os.makedirs(output_dir, exist_ok=True)
print(f"\n2. Created output directory: {output_dir}")

# Convert permeability to text format
# IMPES simulator expects tab-separated values
print("\n3. Converting permeability field to text format...")
perm_file = os.path.join(output_dir, 'permeability.txt')
np.savetxt(perm_file, perm_field, fmt='%.6f', delimiter='\t')
print(f"   Saved: {perm_file}")
print(f"   Format: 100x100 tab-separated, units=mD")

# Convert porosity to text format
print("\n4. Converting porosity field to text format...")
poro_file = os.path.join(output_dir, 'porosity.txt')
np.savetxt(poro_file, poro_field, fmt='%.6f', delimiter='\t')
print(f"   Saved: {poro_file}")
print(f"   Format: 100x100 tab-separated, dimensionless")

# Create depth file (flat reservoir)
# Phase 1: 5000m x 5000m domain = 16404 ft x 16404 ft
# Assume reservoir top at -7500 ft subsea
print("\n5. Creating depth file (flat reservoir)...")
depth_value = -7500.0  # ft subsea
depth_field = np.ones((100, 100)) * depth_value
depth_file = os.path.join(output_dir, 'depth.txt')
np.savetxt(depth_file, depth_field, fmt='%.2f', delimiter='\t')
print(f"   Saved: {depth_file}")
print(f"   Format: 100x100 tab-separated, constant depth = {depth_value} ft")

# Convert well locations from meters to feet
print("\n6. Converting well locations (meters -> feet)...")
m_to_ft = 3.28084

wells_ft = wells_df.copy()
wells_ft['x_ft'] = wells_df['x_m'] * m_to_ft
wells_ft['y_ft'] = wells_df['y_m'] * m_to_ft

# Save converted well locations
wells_ft_file = os.path.join(output_dir, 'well_locations_ft.csv')
wells_ft.to_csv(wells_ft_file, index=False)
print(f"   Saved: {wells_ft_file}")
print(f"   Conversion factor: {m_to_ft} ft/m")

# Display domain size in feet
domain_m = 5000.0
domain_ft = domain_m * m_to_ft
print(f"   Domain size: {domain_m} m x {domain_m} m = {domain_ft:.1f} ft x {domain_ft:.1f} ft")

# For initial test, select subset of wells
# Choose 10 producers and 5 injectors based on spatial distribution
print("\n7. Selecting wells for initial simulation...")

producers = wells_ft[wells_ft['well_type'] == 'producer']
injectors = wells_ft[wells_ft['well_type'] == 'injector']

# Select wells spread across domain
# For producers: select every 4th well (40/4 = 10)
# For injectors: select every ~3.6th well (18/5 = 3.6)
selected_producers = producers.iloc[::4].head(10).copy()
selected_injectors = injectors.iloc[::4].head(5).copy()

selected_wells = pd.concat([selected_producers, selected_injectors], ignore_index=True)

print(f"   Selected {len(selected_producers)} producers + {len(selected_injectors)} injectors = {len(selected_wells)} total")
print(f"   Producer IDs: {selected_producers['well_id'].tolist()}")
print(f"   Injector IDs: {selected_injectors['well_id'].tolist()}")

# Save selected wells
selected_wells_file = os.path.join(output_dir, 'selected_wells.csv')
selected_wells.to_csv(selected_wells_file, index=False)
print(f"   Saved: {selected_wells_file}")

# Create well configuration for IMPES input file
print("\n8. Generating well configuration for IMPES...")

# Prepare well data in IMPES format
# IMPES expects: well.x, well.y, well.type, well.value
# well.type: 1 = rate control, 2 = BHP control
# Producers: BHP control at 1000 psi
# Injectors: Rate control at 500 STB/day

well_config = {
    'x_ft': [],
    'y_ft': [],
    'type': [],  # 1=rate, 2=BHP
    'value': [],  # rate (STB/day) or BHP (psi)
    'well_id': [],
    'well_type': []
}

# Add producers (BHP control)
for _, prod in selected_producers.iterrows():
    well_config['x_ft'].append(prod['x_ft'])
    well_config['y_ft'].append(prod['y_ft'])
    well_config['type'].append(2)  # BHP control
    well_config['value'].append(1000.0)  # 1000 psi
    well_config['well_id'].append(prod['well_id'])
    well_config['well_type'].append('producer')

# Add injectors (rate control)
for _, inj in selected_injectors.iterrows():
    well_config['x_ft'].append(inj['x_ft'])
    well_config['y_ft'].append(inj['y_ft'])
    well_config['type'].append(1)  # Rate control
    well_config['value'].append(500.0)  # 500 STB/day
    well_config['well_id'].append(inj['well_id'])
    well_config['well_type'].append('injector')

well_config_df = pd.DataFrame(well_config)
well_config_file = os.path.join(output_dir, 'well_configuration.csv')
well_config_df.to_csv(well_config_file, index=False)
print(f"   Saved: {well_config_file}")
print(f"   Producers: BHP control @ 1000 psi")
print(f"   Injectors: Rate control @ 500 STB/day")

# Summary statistics
print("\n" + "="*70)
print("CONVERSION SUMMARY")
print("="*70)
print(f"Grid: 100 x 100 cells")
print(f"Domain: {domain_ft:.1f} ft x {domain_ft:.1f} ft")
print(f"Cell size: {domain_ft/100:.1f} ft x {domain_ft/100:.1f} ft")
print(f"\nReservoir Properties:")
print(f"  Permeability: {perm_field.mean():.2f} mD (range: {perm_field.min():.2f}-{perm_field.max():.2f})")
print(f"  Porosity: {poro_field.mean():.4f} (range: {poro_field.min():.4f}-{poro_field.max():.4f})")
print(f"  Depth: {depth_value} ft (constant)")
print(f"\nWells:")
print(f"  Total available: {len(wells_df)} ({len(producers)} producers, {len(injectors)} injectors)")
print(f"  Selected for simulation: {len(selected_wells)} ({len(selected_producers)} producers, {len(selected_injectors)} injectors)")
print(f"\nOutput Files:")
print(f"  {perm_file}")
print(f"  {poro_file}")
print(f"  {depth_file}")
print(f"  {wells_ft_file}")
print(f"  {selected_wells_file}")
print(f"  {well_config_file}")
print("="*70)

print("\nData conversion complete! Ready for IMPES simulator.")
