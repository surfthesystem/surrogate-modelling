"""
Phase 1 Reservoir - IMPES Simulation Input File
Customized for GNN-LSTM Surrogate Model Project

Grid: 100x100 cells
Domain: 16404.2 ft x 16404.2 ft (5000m x 5000m)
Wells: 10 producers + 5 injectors

Author: Adapted from Mohammad Afzal Shadab's IMPES code
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt

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

#fluid, reservoir and simulation parameters
def inputfile(fluid,reservoir,petro,numerical,BC,IC,well):
    # Numerical simulation parameters
    numerical.dt     = 1.0  #time step (days)
    numerical.tfinal = 10   #final time [days] - SHORT TEST RUN
    numerical.PV_final = 1  #final pore volume
    numerical.Nx     = 100  #number of grid blocks in x-direction (Phase 1)
    numerical.Ny     = 100  #number of grid blocks in y-direction (Phase 1)
    numerical.N  = numerical.Nx * numerical.Ny #Total number of grid blocks (10,000)
    numerical.method = 'IMPES' #Implicit pressure explicit saturation solver
    numerical.tswitch= 500  #time after switch occurs [days]

    # Fluid parameters - same as original for consistency
    fluid.muw = 0.383*np.ones((numerical.N, 1))   #water viscosity [centipoise]
    fluid.Bw  = 1.012298811*np.ones((numerical.N, 1))   #formation volume factor of water [rb/stb]
    fluid.cw  = 2.87E-6                          #total compressibility: rock + fluid [1/psi]

    fluid.muo = 2.47097*np.ones((numerical.N, 1)) #oil viscosity [centipoise]
    fluid.Bo  = 1.04567*np.ones((numerical.N, 1)) #formation volume factor of oil [rb/stb]
    fluid.co  = 3.0E-6     #total compressibility: rock + fluid [1/psi]

    fluid.rhow   = 62.4      #density of the water [lbm/ft^3]
    fluid.rhoosc = 53.0      #density of the stock tank oil [lbm/ft^3]
    fluid.sg     = 0.6       #specific gravity of gas
    fluid.BP     = 502.505   #bubble point pressure [psi]
    fluid.Rs     = 90.7388   #solution gas ration [scf/STB]
    fluid.B_BP   = 1.5       #formation volume factor at bubble point pressure [rb/stb]

    # Multiphase/Relative permeability values and Capillary pressure
    petro.Swr  = 0.2    #residual water saturation
    petro.Sor  = 0.4    #residual oil saturation
    petro.nw   = 2.0    #Corey-Brooks exponent (water)
    petro.no   = 2.0    #Corey-Brooks exponent (oil)
    petro.krw0 = 0.3    #Corey-Brooks endpoint (water)
    petro.kro0 = 0.8    #Corey-Brooks endpoint (oil)
    petro.lamda= 2.0    #fitting parameter for Corey-Brooks model
    petro.Pe   = 3.5    #capillary entry pressure [psi]

    #reading Phase 1 generated files - using relative paths from script location
    # Script is in: Reservoir-Simulator/proj2/Problem 2/
    # Data is in: data/impes_input/
    base_path = "../../../data/impes_input/"

    depth    =-np.loadtxt(base_path + "depth.txt")
    porosity = np.loadtxt(base_path + "porosity.txt")
    permx    = np.loadtxt(base_path + "permeability.txt")

    # Reservoir parameters - Phase 1 configuration
    reservoir.L  = 16404.2   #length of the reservoir [feet] (5000m converted)
    reservoir.h  = 50.0      #height of the reservoir [feet]
    reservoir.W  = 16404.2   #width of the reservoir [feet] (5000m converted)
    reservoir.T  = 100.0     #Temperature of the reservoir [F]
    reservoir.phi   = np.reshape(porosity, (numerical.N,1))  #porosity of the reservior vector
    reservoir.permx = np.reshape(permx, (numerical.N,1))  #fluid permeability in x direction vector [mDarcy]
    reservoir.permx[reservoir.permx <= 1E-6] = 1E-16  #avoid zero permeability
    reservoir.permy  = 0.15*reservoir.permx #fluid permeability in y direction vector [mDarcy]
    reservoir.permz  = 1.0*reservoir.permx #fluid permeability in z direction vector [mDarcy]
    reservoir.Dref   = 7500.0            #reference depth [feet] (matches our constant depth)
    reservoir.alpha  = 0.0*np.pi/6.0     #dip angle [in radians] - flat reservoir
    reservoir.cfr    = 1e-6              #formation of rock compressibility
    reservoir.Pref   = 4500.0            #pressure at reference depth [psi]
    reservoir.Pwf    = 1000.0            #reservoir BHP [psi] (producer constraint)
    reservoir.phi[reservoir.phi==0] = 1E-16  #avoid zero porosity

    # Well parameters - from Phase 1 well_configuration.csv
    # 10 producers (BHP control @ 1000 psi) + 5 injectors (rate control @ 500 STB/day)

    # Well x-coordinates [ft]
    well.x = [
        [960.04],      # PROD_000
        [10430.73],    # PROD_004
        [3008.45],     # PROD_008
        [13199.87],    # PROD_012
        [6413.32],     # PROD_016
        [15690.18],    # PROD_020
        [8642.12],     # PROD_024
        [816.98],      # PROD_028
        [11207.92],    # PROD_032
        [3375.83],     # PROD_036
        [7036.20],     # INJ_000
        [8550.18],     # INJ_004
        [8729.29],     # INJ_008
        [4572.10],     # INJ_012
        [9815.16]      # INJ_016
    ]

    # Well y-coordinates [ft]
    well.y = [
        [2303.54],     # PROD_000
        [933.22],      # PROD_004
        [4130.63],     # PROD_008
        [3976.69],     # PROD_012
        [6537.44],     # PROD_016
        [5950.01],     # PROD_020
        [9696.44],     # PROD_024
        [11599.21],    # PROD_028
        [12384.90],    # PROD_032
        [14459.76],    # PROD_036
        [3535.69],     # INJ_000
        [2569.95],     # INJ_004
        [6419.55],     # INJ_008
        [3405.63],     # INJ_012
        [13266.84]     # INJ_016
    ]

    # Well type: 1 = rate control, 2 = BHP control
    # First 10 are producers (BHP), last 5 are injectors (rate)
    well.type = [
        [2],[2],[2],[2],[2],[2],[2],[2],[2],[2],  # 10 producers
        [1],[1],[1],[1],[1]                        # 5 injectors
    ]

    # Well constraint: BHP [psi] for producers, rate [STB/day] for injectors
    # Producers: 1000 psi BHP
    # Injectors: 500 STB/day water injection (positive = injection)
    # NOTE: For water injection, use STB/day directly (not scf/day conversion)
    well.constraint = [
        [1000.0],[1000.0],[1000.0],[1000.0],[1000.0],  # 5 producers (BHP)
        [1000.0],[1000.0],[1000.0],[1000.0],[1000.0],  # 5 producers (BHP)
        [500.0],[500.0],[500.0],                       # 3 injectors (rate in STB/day)
        [500.0],[500.0]                                # 2 injectors (rate in STB/day)
    ]

    # Well radius [ft]
    well.rw = [[0.25]]*15  # 15 wells

    # Well skin factor [dimensionless]
    well.skin = [[0]]*15  # 15 wells, no skin

    # Well direction: 'v' = vertical
    well.direction = [['v']]*15  # 15 vertical wells

    #Defining numerical parameters for discretized solution
    numerical.dx1 = (reservoir.L/numerical.Nx)*np.ones((numerical.Nx, 1)) #block thickness in x vector
    numerical.dy1 = (reservoir.W/numerical.Ny)*np.ones((numerical.Ny, 1)) #block thickness in y vector
    [numerical.dX,numerical.dY] = np.meshgrid(numerical.dx1,numerical.dy1)

    numerical.dx  = np.reshape(numerical.dX, (numerical.N,1))    #building the single dx column vector
    numerical.dy  = np.reshape(numerical.dY, (numerical.N,1))    #building the single dy column vector

    #position of the block centres x-direction
    numerical.xc = np.empty((numerical.Nx, 1))
    numerical.xc[0,0] = 0.5 * numerical.dx[0,0]
    for i in range(1,numerical.Nx):
        numerical.xc[i,0] = numerical.xc[i-1,0] + 0.5*(numerical.dx1[i-1,0] + numerical.dx1[i,0])

    #position of the block centres y-direction
    numerical.yc = np.empty((numerical.Ny, 1))
    numerical.yc[0,0] = 0.5 * numerical.dy[0,0]
    for i in range(1,numerical.Ny):
        numerical.yc[i,0] = numerical.yc[i-1,0] + 0.5*(numerical.dy1[i-1,0] + numerical.dy1[i,0])

    [numerical.Xc,numerical.Yc] = np.meshgrid(numerical.xc,numerical.yc)

    numerical.x1  = np.reshape(numerical.Xc, (numerical.N,1))    #building the single X column vector
    numerical.y1  = np.reshape(numerical.Yc, (numerical.N,1))    #building the single Y column vector

    #depth vector
    numerical.D = np.reshape(depth, (numerical.N,1))

    # Boundary conditions: No-flow (Neumann) on all sides
    BC.type  = [['Neumann'],['Neumann'],['Neumann'],['Neumann']] #type of BC: left, right, bottom, top
    BC.value = [[0],[0],[0],[0]]    #value of the boundary condition: psi or ft^3/day

    # Initial conditions
    # For waterflooding: Start with oil-saturated reservoir at residual water saturation
    # IMPORTANT: Cannot set Sw exactly equal to Swr - causes singularity in capillary pressure!
    # Set Sw slightly above Swr to avoid 0^(-0.5) = inf in cap_press.py
    IC.P     = reservoir.Pref*np.ones((numerical.N,1)) #Initial Pressure [psi]
    IC.Pw    = reservoir.Pref*np.ones((numerical.N,1)) #Initial Water Pressure [psi]
    IC.Sw    = (petro.Swr + 0.01)*np.ones((numerical.N,1))  #Initial water saturation slightly above residual (0.21)

    # Initialize fluid properties
    fluid_properties(reservoir,fluid,IC.P,IC.P)

    # SKIP hydrostatic equilibrium calculation for flat reservoir
    # We want a uniform oil-saturated reservoir for waterflooding
    # NOT a water-saturated reservoir which the equilibrium calculation produces

    # Re-set fluid parameters after equilibration
    fluid.muw = 0.383*np.ones((numerical.N, 1))   #fluid viscosity [centipoise]
    fluid.Bw  = 1.012298811*np.ones((numerical.N, 1))   #formation volume factor of water [rb/stb]
    fluid.cw  = 2.87E-6                          #total compressibility: rock + fluid [1/psi]

    fluid.muo = 2.47097*np.ones((numerical.N, 1)) #fluid viscosity [centipoise]
    fluid.Bo  = 1.04567*np.ones((numerical.N, 1)) #formation volume factor of oil [rb/stb]
    fluid.co  = 3.0E-6     #total compressibility: rock + fluid [1/psi]

    fluid.rhow   = 62.4      #density of the water [lbm/ft^3]
    fluid.rhoosc = 53.0      #density of the stock tank oil [lbm/ft^3]
    fluid.sg     = 0.6       #specific gravity of gas
    fluid.BP     = 502.505   #bubble point pressure [psi]
    fluid.Rs     = 90.7388   #solution gas ration [scf/STB]
    fluid.B_BP   = 1.5       #formation volume factor at bubble point pressure [rb/stb]

    print("="*70)
    print("PHASE 1 RESERVOIR - IMPES SIMULATION CONFIGURATION")
    print("="*70)
    print(f"Grid: {numerical.Nx} x {numerical.Ny} = {numerical.N} cells")
    print(f"Domain: {reservoir.L:.1f} ft x {reservoir.W:.1f} ft")
    print(f"Cell size: {reservoir.L/numerical.Nx:.1f} ft x {reservoir.W/numerical.Ny:.1f} ft")
    print(f"Simulation time: {numerical.tfinal} days")
    print(f"Time step: {numerical.dt} day")
    print(f"\nReservoir Properties:")
    print(f"  Permeability: {reservoir.permx.mean():.2f} mD (range: {reservoir.permx.min():.2f}-{reservoir.permx.max():.2f})")
    print(f"  Porosity: {reservoir.phi.mean():.4f} (range: {reservoir.phi.min():.4f}-{reservoir.phi.max():.4f})")
    print(f"  Depth: {reservoir.Dref:.1f} ft")
    print(f"  Initial Pressure: {reservoir.Pref:.1f} psi")
    print(f"  Initial Water Saturation: {IC.Sw.mean():.3f}")
    print(f"\nWells: {len(well.x)} total")
    print(f"  Producers: 10 (BHP control @ 1000 psi)")
    print(f"  Injectors: 5 (rate control @ 500 STB/day)")
    print("="*70)

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
