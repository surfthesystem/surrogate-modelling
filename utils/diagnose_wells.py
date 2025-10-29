"""
Diagnose Well Positioning for IMPES Simulation
Check if wells fall within grid boundaries
"""

import numpy as np
import pandas as pd

# Load well configuration
wells_df = pd.read_csv('data/impes_input/well_configuration.csv')

# Grid parameters (from input_file_phase1.py)
Nx = 100
Ny = 100
L = 16404.2  # ft
W = 16404.2  # ft

# Calculate cell parameters
dx = L / Nx  # 164.042 ft
dy = W / Ny  # 164.042 ft

# Cell center positions
xc = np.empty((Nx, 1))
xc[0,0] = 0.5 * dx
for i in range(1, Nx):
    xc[i,0] = xc[i-1,0] + dx

yc = np.empty((Ny, 1))
yc[0,0] = 0.5 * dy
for j in range(1, Ny):
    yc[j,0] = yc[j-1,0] + dy

print("="*70)
print("WELL POSITIONING DIAGNOSTIC")
print("="*70)
print(f"\nGrid Configuration:")
print(f"  Nx = {Nx}, Ny = {Ny}")
print(f"  Domain: {L:.2f} ft x {W:.2f} ft")
print(f"  Cell size: {dx:.3f} ft x {dy:.3f} ft")
print(f"  X range: 0 to {L:.2f} ft")
print(f"  Y range: 0 to {W:.2f} ft")
print(f"  First cell center: ({xc[0,0]:.3f}, {yc[0,0]:.3f})")
print(f"  Last cell center: ({xc[-1,0]:.3f}, {yc[-1,0]:.3f})")

print(f"\nWell Locations:")
print(f"  Total wells: {len(wells_df)}")

problems = []
well_blocks = []

for idx, row in wells_df.iterrows():
    well_x = row['x_ft']
    well_y = row['y_ft']
    well_id = row['well_id']
    well_type = row['well_type']

    # Check if well is within domain
    if well_x < 0 or well_x > L:
        problems.append(f"  {well_id}: X coordinate {well_x:.2f} is OUTSIDE domain [0, {L:.2f}]")
    if well_y < 0 or well_y > W:
        problems.append(f"  {well_id}: Y coordinate {well_y:.2f} is OUTSIDE domain [0, {W:.2f}]")

    # Find grid block (same logic as updatewells.py)
    iblock = 0
    for i in range(Nx):
        if well_x < (xc[i,0] + dx/2) and well_x >= (xc[i,0] - dx/2):
            iblock = i
            break

    jblock = 0
    for j in range(Ny):
        if well_y < (yc[j,0] + dy/2) and well_y >= (yc[j,0] - dy/2):
            jblock = j
            break

    kblock = iblock + jblock * Nx

    well_blocks.append({
        'well_id': well_id,
        'well_type': well_type,
        'x_ft': well_x,
        'y_ft': well_y,
        'iblock': iblock,
        'jblock': jblock,
        'kblock': kblock,
        'cell_center_x': xc[iblock,0],
        'cell_center_y': yc[jblock,0],
        'distance_to_center': np.sqrt((well_x - xc[iblock,0])**2 + (well_y - yc[jblock,0])**2)
    })

    print(f"  {well_id:12s} ({well_type:8s}): ({well_x:8.2f}, {well_y:8.2f}) -> Cell ({iblock:3d}, {jblock:3d}) = Block {kblock:5d}")

# Check for duplicate blocks
blocks_df = pd.DataFrame(well_blocks)
duplicate_blocks = blocks_df[blocks_df.duplicated(subset=['kblock'], keep=False)]

print(f"\n{'='*70}")
print("DIAGNOSTIC RESULTS")
print("="*70)

if problems:
    print(f"\nPROBLEMS FOUND:")
    for p in problems:
        print(p)
else:
    print(f"\n✓ All wells are within domain boundaries")

if len(duplicate_blocks) > 0:
    print(f"\nWARNING: {len(duplicate_blocks)} wells assigned to duplicate blocks:")
    print(duplicate_blocks[['well_id', 'well_type', 'iblock', 'jblock', 'kblock']])
else:
    print(f"\n✓ No duplicate well blocks")

print(f"\nWell Block Summary:")
print(f"  Minimum block index: {blocks_df['kblock'].min()}")
print(f"  Maximum block index: {blocks_df['kblock'].max()}")
print(f"  Total grid blocks: {Nx * Ny}")
print(f"  Unique well blocks: {blocks_df['kblock'].nunique()}")

# Save diagnostic results
blocks_df.to_csv('data/impes_input/well_blocks_diagnostic.csv', index=False)
print(f"\nDetailed well block assignments saved to: data/impes_input/well_blocks_diagnostic.csv")

print("="*70)
