"""
Test script to generate an input file and check for issues
"""
import numpy as np
import pandas as pd
import os

# Import the function from run_lhs_optimization
import sys
sys.path.insert(0, os.path.dirname(__file__))

# Configuration
N_SAMPLES = 1
SIMULATION_DAYS = 90
N_INJECTORS = 18
N_PRODUCERS = 40
N_WELLS_TOTAL = 58

def read_well_locations():
    """Read well locations from CSV."""
    wells_file = '../data/well_locations.csv'
    df = pd.read_csv(wells_file)

    # Convert to feet
    df['x_ft'] = df['x_m'] * 3.28084
    df['y_ft'] = df['y_m'] * 3.28084

    # Separate producers and injectors
    producers = df[df['well_type'] == 'producer'].reset_index(drop=True)
    injectors = df[df['well_type'] == 'injector'].reset_index(drop=True)

    return producers, injectors

# Create test directory
output_dir = 'test_input_gen'
os.makedirs(output_dir, exist_ok=True)

# Test parameters
inj_rates = np.ones(N_INJECTORS) * 1500
prod_bhp = 1000

print("Generating test input file...")

# Read template
template_file = '../simulator/input_file_phase1.py'
with open(template_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"Template has {len(lines)} lines")

# Read well locations
producers, injectors = read_well_locations()

# Create input file (same logic as run_lhs_optimization.py)
output_file = os.path.join(output_dir, 'input_sample_test.py')

with open(output_file, 'w', encoding='utf-8') as f:
    # Write header
    f.write('"""\n')
    f.write('Test Input File\n')
    f.write('"""\n\n')

    # Copy imports and class definitions (lines 1-49)
    for i in range(49):
        f.write(lines[i])

    # Start inputfile function
    f.write('\ndef inputfile(fluid,reservoir,petro,numerical,BC,IC,well):\n')

    # Write numerical parameters
    f.write(f'    numerical.dt     = 1.0\n')
    f.write(f'    numerical.tfinal = {SIMULATION_DAYS}\n')
    f.write(f'    numerical.PV_final = 1\n')
    f.write(f'    numerical.Nx     = 100\n')
    f.write(f'    numerical.Ny     = 100\n')
    f.write(f'    numerical.N  = numerical.Nx * numerical.Ny\n')
    f.write(f'    numerical.method = "IMPES"\n')
    f.write(f'    numerical.tswitch= 500\n\n')

    # Copy fluid and petro parameters (lines 61-86)
    for i in range(61, 86):
        f.write('    ' + lines[i].strip() + '\n')

    # Write reservoir parameters
    f.write('\n    # Reading Phase 1 generated files\n')
    f.write('    base_path = "../../data/impes_input/"\n')
    f.write('    depth    =-np.loadtxt(base_path + "depth.txt")\n')
    f.write('    porosity = np.loadtxt(base_path + "porosity.txt")\n')
    f.write('    permx    = np.loadtxt(base_path + "permeability.txt")\n\n')

    # Reservoir properties (lines 97-111)
    for i in range(96, 111):
        f.write('    ' + lines[i].strip() + '\n')

    # Write well configuration
    f.write('\n    # Well configuration - ALL 58 WELLS\n')
    f.write(f'    # {N_PRODUCERS} producers (BHP control) + {N_INJECTORS} injectors (rate control)\n\n')

    # Well x-coordinates
    f.write('    well.x = [\n')
    for idx, row in producers.iterrows():
        f.write(f'        [{row["x_ft"]:.2f}],\n')
    for idx, row in injectors.iterrows():
        f.write(f'        [{row["x_ft"]:.2f}],\n')
    f.write('    ]\n\n')

    # Well y-coordinates
    f.write('    well.y = [\n')
    for idx, row in producers.iterrows():
        f.write(f'        [{row["y_ft"]:.2f}],\n')
    for idx, row in injectors.iterrows():
        f.write(f'        [{row["y_ft"]:.2f}],\n')
    f.write('    ]\n\n')

    # Well types
    f.write('    well.type = [\n')
    f.write(f'        {[[2]]*N_PRODUCERS},\n')
    f.write(f'        {[[1]]*N_INJECTORS}\n')
    f.write('    ]\n')
    f.write('    well.type = [item for sublist in well.type for item in sublist]\n\n')

    # Well constraints
    f.write('    well.constraint = [\n')
    for i in range(N_PRODUCERS):
        f.write(f'        [{prod_bhp:.1f}],\n')
    for i in range(N_INJECTORS):
        f.write(f'        [{inj_rates[i]:.2f}],\n')
    f.write('    ]\n\n')

    # Other well properties
    f.write(f'    well.rw = [[0.25]]*{N_WELLS_TOTAL}\n')
    f.write(f'    well.skin = [[0]]*{N_WELLS_TOTAL}\n')
    f.write(f'    well.direction = [["v"]]*{N_WELLS_TOTAL}\n\n')

    # Copy rest of input file (grid setup, BC, IC) - WITH PROPER INDENTATION
    print(f"Copying lines 181-261 from template...")
    for i in range(181, 262):
        line = lines[i]
        if line.strip():
            f.write(line)
        else:
            f.write('\n')

    # End of function
    f.write('\n# Initialize objects\n')
    f.write('fluid = fluid()\n')
    f.write('reservoir = reservoir()\n')
    f.write('petro = petro()\n')
    f.write('numerical = numerical()\n')
    f.write('BC = BC()\n')
    f.write('IC = IC()\n')
    f.write('well = well()\n\n')
    f.write('# Call input file function\n')
    f.write('inputfile(fluid,reservoir,petro,numerical,BC,IC,well)\n')

print(f"Generated: {output_file}")

# Count lines in generated file
with open(output_file, 'r') as f:
    gen_lines = f.readlines()

print(f"Generated file has {len(gen_lines)} lines")

# Check around line 404
if len(gen_lines) >= 404:
    print(f"\nLines around 404:")
    for i in range(max(0, 400), min(len(gen_lines), 410)):
        print(f"{i+1:3d}: {gen_lines[i]}", end='')
else:
    print(f"\nGenerated file only has {len(gen_lines)} lines (less than 404)")
    print(f"\nLast 10 lines:")
    for i in range(max(0, len(gen_lines)-10), len(gen_lines)):
        print(f"{i+1:3d}: {gen_lines[i]}", end='')

# Try to import it to see if there are any syntax errors
print("\n\nTrying to parse the generated file...")
try:
    with open(output_file, 'r') as f:
        code = f.read()
    compile(code, output_file, 'exec')
    print("✓ File compiles successfully!")
except SyntaxError as e:
    print(f"✗ Syntax error at line {e.lineno}: {e.msg}")
    print(f"   {e.text}")
except IndentationError as e:
    print(f"✗ Indentation error at line {e.lineno}: {e.msg}")
    print(f"   {e.text}")
