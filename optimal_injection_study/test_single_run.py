"""
Quick test script to verify the simulator works before running full LHS study
"""
import sys
import os
import numpy as np
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Configuration
TEST_DIR = "test_run"
SIMULATION_DAYS = 10  # Short test run

# Well configuration (using all 58 wells)
N_INJECTORS = 18
N_PRODUCERS = 40

# Test parameters
BASE_INJECTION_RATE = 1500  # STB/day
PRODUCER_BHP = 1000  # psi

print("="*70)
print("TESTING SINGLE SIMULATION")
print("="*70)

# Create test directory
os.makedirs(TEST_DIR, exist_ok=True)

# Load well locations
import pandas as pd
well_data = pd.read_csv('../data/well_locations.csv')

# Separate injectors and producers
injectors = well_data[well_data['well_type'] == 'injector'].reset_index(drop=True)
producers = well_data[well_data['well_type'] == 'producer'].reset_index(drop=True)

print(f"\nWells loaded:")
print(f"  - Injectors: {len(injectors)}")
print(f"  - Producers: {len(producers)}")

# Create simple injection rates (all equal for test)
injection_rates = np.ones(N_INJECTORS) * BASE_INJECTION_RATE
producer_bhps = np.ones(N_PRODUCERS) * PRODUCER_BHP

print(f"\nTest parameters:")
print(f"  - Injection rate: {BASE_INJECTION_RATE} STB/day")
print(f"  - Producer BHP: {PRODUCER_BHP} psi")
print(f"  - Simulation time: {SIMULATION_DAYS} days")

# Create input file
input_content = f"""\"\"\"
Test Input File - Single Simulation
\"\"\"
import numpy as np

class inputfile:
    filename = 'test_run'
    name = 'Test Single Simulation'
    t_max = {SIMULATION_DAYS}
    dt = 0.1
    output_interval = 1.0

class fluid:
    mu_w = 0.5
    mu_o = 2.0
    rho_w = 62.4
    rho_o = 49.0
    c_w = 3e-6
    c_o = 1e-5
    B_w = 1.0
    B_o = 1.2

class reservoir:
    dimension = 2
    Lx = 1000
    Ly = 1000
    Nx = 100
    Ny = 100

    # Load permeability from data
    perm = np.load('../../data/permeability_field.npy')
    k = perm
    kx = perm
    ky = perm

    # Load porosity
    phi = np.load('../../data/porosity_field.npy')

    h = 50
    c_r = 1e-6

class petro:
    Swc = 0.2
    Sor = 0.2
    krw_max = 0.8
    kro_max = 1.0
    nw = 2.0
    no = 2.0

class numerical:
    tol = 1e-6
    max_iter = 100

class BC:
    left = 'noflow'
    right = 'noflow'
    top = 'noflow'
    bottom = 'noflow'

class IC:
    P = 2000.0
    Sw = 0.2

class well:
"""

# Add well locations
input_content += f"    loc_inj = {injectors[['x_m', 'y_m']].values.tolist()}\n"
input_content += f"    loc_prod = {producers[['x_m', 'y_m']].values.tolist()}\n"
input_content += f"    Nw_inj = {N_INJECTORS}\n"
input_content += f"    Nw_prod = {N_PRODUCERS}\n"

# Well constraints
input_content += f"    type = [[1]]*{N_INJECTORS} + [[2]]*{N_PRODUCERS}\n"
input_content += f"    constraint = {injection_rates.tolist()} + {producer_bhps.tolist()}\n"
input_content += f"    rw = 0.25\n"

# Save input file
input_path = os.path.join(TEST_DIR, 'input_file_phase1.py')
with open(input_path, 'w') as f:
    f.write(input_content)

print(f"\nCreated input file: {input_path}")

# Copy simulator files to test directory
simulator_files = [
    'IMPES_phase1.py',
    'myarrays.py',
    'updatewells.py',
    'rel_perm.py',
    'spdiaginv.py',
    'init_plot.py',
    'postprocess.py',
    'Thalf.py',
    'cap_press.py',
    'mobilityfun.py',
    'prodindex.py',
    'fluid_properties.py',
    'petrophysics.py'
]

print("\nCopying simulator files...")
for file in simulator_files:
    src = os.path.join('..', 'simulator', file)
    dst = os.path.join(TEST_DIR, file)
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"  - {file}")

# Run simulation
print("\n" + "="*70)
print("RUNNING TEST SIMULATION")
print("="*70)

os.chdir(TEST_DIR)
sys.path.insert(0, os.getcwd())

try:
    # Import and run
    import IMPES_phase1
    print("\n" + "="*70)
    print("TEST SIMULATION COMPLETED SUCCESSFULLY!")
    print("="*70)
except Exception as e:
    print("\n" + "="*70)
    print("TEST SIMULATION FAILED")
    print("="*70)
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
