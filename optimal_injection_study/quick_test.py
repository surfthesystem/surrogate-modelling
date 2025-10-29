"""
Quick test - Run 1 simulation for 10 days to verify everything works
"""
import numpy as np
import pandas as pd
import os
import sys
import shutil
import subprocess

# Configuration
TEST_SAMPLES = 1
TEST_DAYS = 10
OUTPUT_DIR = 'test_results'

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'simulator'))

print("="*70)
print("QUICK TEST - 1 SIMULATION FOR 10 DAYS")
print("="*70)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read well locations
wells_file = '../data/well_locations.csv'
df = pd.read_csv(wells_file)

# Convert to feet
df['x_ft'] = df['x_m'] * 3.28084
df['y_ft'] = df['y_m'] * 3.28084

# Separate producers and injectors
producers = df[df['well_type'] == 'producer'].reset_index(drop=True)
injectors = df[df['well_type'] == 'injector'].reset_index(drop=True)

N_PRODUCERS = len(producers)
N_INJECTORS = len(injectors)
N_WELLS_TOTAL = N_PRODUCERS + N_INJECTORS

print(f"\nWells: {N_WELLS_TOTAL} ({N_PRODUCERS} producers + {N_INJECTORS} injectors)")

# Simple test parameters
inj_rates = np.ones(N_INJECTORS) * 1500  # 1500 STB/day each
prod_bhp = 1000  # psi

print(f"Injection rate: 1500 STB/day per injector")
print(f"Total injection: {np.sum(inj_rates):.0f} STB/day")
print(f"Producer BHP: {prod_bhp} psi")

# Create run directory
run_dir = os.path.join(OUTPUT_DIR, 'run_001')
os.makedirs(run_dir, exist_ok=True)

print(f"\nCreating input file in: {run_dir}")

# Create input file
input_content = f'''"""
Quick Test Input File - 10 day simulation
"""
import numpy as np

from rel_perm import rel_perm
from Thalf import Thalf
from myarrays import myarrays
from fluid_properties import fluid_properties

class fluid:
    def __init__(self):
        self.mu = []
class numerical:
    def __init__(self):
        self.Bw  = []
class reservoir:
    def __init__(self):
        self.dt = []
class grid:
    def __init__(self):
        self.xmin = []
class BC:
    def __init__(self):
        self.xmin = []
class IC:
    def __init__(self):
        self.xmin = []
class petro:
    def __init__(self):
        self.xmin = []
class well:
    def __init__(self):
        self.xmin = []
        self.xblock = []

def inputfile(fluid,reservoir,petro,numerical,BC,IC,well):
    # Numerical simulation parameters
    numerical.dt     = 1.0
    numerical.tfinal = {TEST_DAYS}
    numerical.PV_final = 1
    numerical.Nx     = 100
    numerical.Ny     = 100
    numerical.N  = numerical.Nx * numerical.Ny
    numerical.method = 'IMPES'
    numerical.tswitch= 500

    # Fluid parameters
    fluid.muw = 0.383*np.ones((numerical.N, 1))
    fluid.Bw  = 1.012298811*np.ones((numerical.N, 1))
    fluid.cw  = 2.87E-6

    fluid.muo = 2.47097*np.ones((numerical.N, 1))
    fluid.Bo  = 1.04567*np.ones((numerical.N, 1))
    fluid.co  = 3.0E-6

    fluid.rhow   = 62.4
    fluid.rhoosc = 53.0
    fluid.sg     = 0.6
    fluid.BP     = 502.505
    fluid.Rs     = 90.7388
    fluid.B_BP   = 1.5

    # Petro parameters
    petro.Swr  = 0.2
    petro.Sor  = 0.4
    petro.nw   = 2.0
    petro.no   = 2.0
    petro.krw0 = 0.3
    petro.kro0 = 0.8
    petro.lamda= 2.0
    petro.Pe   = 3.5

    # Reading Phase 1 generated files
    base_path = "../../../data/impes_input/"
    depth    =-np.loadtxt(base_path + "depth.txt")
    porosity = np.loadtxt(base_path + "porosity.txt")
    permx    = np.loadtxt(base_path + "permeability.txt")

    # Reservoir parameters
    reservoir.L  = 16404.2
    reservoir.h  = 50.0
    reservoir.W  = 16404.2
    reservoir.T  = 100.0
    reservoir.phi   = np.reshape(porosity, (numerical.N,1))
    reservoir.permx = np.reshape(permx, (numerical.N,1))
    reservoir.permx[reservoir.permx <= 1E-6] = 1E-16
    reservoir.permy  = 0.15*reservoir.permx
    reservoir.permz  = 1.0*reservoir.permx
    reservoir.Dref   = 7500.0
    reservoir.alpha  = 0.0*np.pi/6.0
    reservoir.cfr    = 1e-6
    reservoir.Pref   = 4500.0
    reservoir.Pwf    = 1000.0
    reservoir.phi[reservoir.phi==0] = 1E-16

    # Well configuration - ALL {N_WELLS_TOTAL} WELLS
    # {N_PRODUCERS} producers (BHP control) + {N_INJECTORS} injectors (rate control)
'''

# Add well x-coordinates
input_content += '    well.x = [\n'
for idx, row in producers.iterrows():
    input_content += f'        [{row["x_ft"]:.2f}],  # PROD_{idx:03d}\n'
for idx, row in injectors.iterrows():
    input_content += f'        [{row["x_ft"]:.2f}],  # INJ_{idx:03d}\n'
input_content += '    ]\n\n'

# Add well y-coordinates
input_content += '    well.y = [\n'
for idx, row in producers.iterrows():
    input_content += f'        [{row["y_ft"]:.2f}],  # PROD_{idx:03d}\n'
for idx, row in injectors.iterrows():
    input_content += f'        [{row["y_ft"]:.2f}],  # INJ_{idx:03d}\n'
input_content += '    ]\n\n'

# Add well types
input_content += '    well.type = [\n'
input_content += f'        {[[2]]*N_PRODUCERS},  # Producers (Type 2 = BHP)\n'
input_content += f'        {[[1]]*N_INJECTORS}   # Injectors (Type 1 = Rate)\n'
input_content += '    ]\n'
input_content += '    well.type = [item for sublist in well.type for item in sublist]\n\n'

# Add well constraints
input_content += '    well.constraint = [\n'
for i in range(N_PRODUCERS):
    input_content += f'        [{prod_bhp:.1f}],  # PROD_{i:03d} BHP (psi)\n'
for i in range(N_INJECTORS):
    input_content += f'        [{inj_rates[i]:.2f}],  # INJ_{i:03d} rate (STB/day)\n'
input_content += '    ]\n\n'

# Add other well properties
input_content += f'    well.rw = [[0.25]]*{N_WELLS_TOTAL}\n'
input_content += f'    well.skin = [[0]]*{N_WELLS_TOTAL}\n'
input_content += f'    well.direction = [["v"]]*{N_WELLS_TOTAL}\n\n'

# Add grid and BC/IC setup (copy from template lines 182-261)
input_content += '''    # Defining numerical parameters for discretized solution
    numerical.dx1 = (reservoir.L/numerical.Nx)*np.ones((numerical.Nx, 1))
    numerical.dy1 = (reservoir.W/numerical.Ny)*np.ones((numerical.Ny, 1))
    [numerical.dX,numerical.dY] = np.meshgrid(numerical.dx1,numerical.dy1)

    numerical.dx  = np.reshape(numerical.dX, (numerical.N,1))
    numerical.dy  = np.reshape(numerical.dY, (numerical.N,1))

    # Position of the block centres x-direction
    numerical.xc = np.empty((numerical.Nx, 1))
    numerical.xc[0,0] = 0.5 * numerical.dx[0,0]
    for i in range(1,numerical.Nx):
        numerical.xc[i,0] = numerical.xc[i-1,0] + 0.5*(numerical.dx1[i-1,0] + numerical.dx1[i,0])

    # Position of the block centres y-direction
    numerical.yc = np.empty((numerical.Ny, 1))
    numerical.yc[0,0] = 0.5 * numerical.dy[0,0]
    for i in range(1,numerical.Ny):
        numerical.yc[i,0] = numerical.yc[i-1,0] + 0.5*(numerical.dy1[i-1,0] + numerical.dy1[i,0])

    [numerical.Xc,numerical.Yc] = np.meshgrid(numerical.xc,numerical.yc)

    numerical.x1  = np.reshape(numerical.Xc, (numerical.N,1))
    numerical.y1  = np.reshape(numerical.Yc, (numerical.N,1))

    # Depth vector
    numerical.D = np.reshape(depth, (numerical.N,1))

    # Boundary conditions: No-flow (Neumann) on all sides
    BC.type  = [['Neumann'],['Neumann'],['Neumann'],['Neumann']]
    BC.value = [[0],[0],[0],[0]]

    # Initial conditions - Start with oil-saturated reservoir
    IC.P     = reservoir.Pref*np.ones((numerical.N,1))
    IC.Pw    = reservoir.Pref*np.ones((numerical.N,1))
    IC.Sw    = (petro.Swr + 0.01)*np.ones((numerical.N,1))

    # Initialize fluid properties
    fluid_properties(reservoir,fluid,IC.P,IC.P)

    # Re-set fluid parameters
    fluid.muw = 0.383*np.ones((numerical.N, 1))
    fluid.Bw  = 1.012298811*np.ones((numerical.N, 1))
    fluid.cw  = 2.87E-6

    fluid.muo = 2.47097*np.ones((numerical.N, 1))
    fluid.Bo  = 1.04567*np.ones((numerical.N, 1))
    fluid.co  = 3.0E-6

    fluid.rhow   = 62.4
    fluid.rhoosc = 53.0
    fluid.sg     = 0.6
    fluid.BP     = 502.505
    fluid.Rs     = 90.7388
    fluid.B_BP   = 1.5

# Initialize objects
fluid = fluid()
reservoir = reservoir()
petro = petro()
numerical = numerical()
BC = BC()
IC = IC()
well = well()

# Call input file function
inputfile(fluid,reservoir,petro,numerical,BC,IC,well)
'''

# Save input file
input_file_path = os.path.join(run_dir, 'input_file_phase1.py')
with open(input_file_path, 'w') as f:
    f.write(input_content)

print(f"Created: {input_file_path}")

# Copy simulator files to run directory
simulator_files = [
    'IMPES_phase1.py', 'myarrays.py', 'updatewells.py',
    'cap_press.py', 'rel_perm.py', 'fluid_properties.py',
    'petrophysics.py', 'prodindex.py', 'Thalf.py', 'spdiaginv.py',
    'mobilityfun.py', 'postprocess.py', 'init_plot.py',
    'assignment1_reservior_init.py', 'petroplots.py'
]

print("\nCopying simulator files...")
for sim_file in simulator_files:
    src = os.path.join('..', 'simulator', sim_file)
    if os.path.exists(src):
        shutil.copy(src, run_dir)
        print(f"  - {sim_file}")

print("\n" + "="*70)
print("RUNNING TEST SIMULATION")
print("="*70)

# Run simulation
original_dir = os.getcwd()
os.chdir(run_dir)

try:
    result = subprocess.run(
        [sys.executable, 'IMPES_phase1.py'],
        timeout=300  # 5 minute timeout for quick test
    )

    os.chdir(original_dir)

    if result.returncode == 0:
        print("\n" + "="*70)
        print("TEST SIMULATION COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nThe code is working! You can now run the full LHS study.")
    else:
        print("\n" + "="*70)
        print("TEST SIMULATION FAILED")
        print("="*70)

except subprocess.TimeoutExpired:
    os.chdir(original_dir)
    print("\nTest timed out (took longer than 5 minutes)")
    print("This might be normal - check the results folder")
except Exception as e:
    os.chdir(original_dir)
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
