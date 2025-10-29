"""
Test Input File
"""

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
    numerical.dt     = 1.0
    numerical.tfinal = 90
    numerical.PV_final = 1
    numerical.Nx     = 100
    numerical.Ny     = 100
    numerical.N  = numerical.Nx * numerical.Ny
    numerical.method = "IMPES"
    numerical.tswitch= 500

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
    

    # Reading Phase 1 generated files
    base_path = "../../data/impes_input/"
    depth    =-np.loadtxt(base_path + "depth.txt")
    porosity = np.loadtxt(base_path + "porosity.txt")
    permx    = np.loadtxt(base_path + "permeability.txt")

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

    # Well configuration - ALL 58 WELLS
    # 40 producers (BHP control) + 18 injectors (rate control)

    well.x = [
        [960.04],
        [3906.63],
        [6029.25],
        [8763.03],
        [10430.73],
        [13077.74],
        [14881.76],
        [1097.61],
        [3008.45],
        [5124.76],
        [8987.76],
        [10215.88],
        [13199.87],
        [14825.46],
        [1336.91],
        [2820.85],
        [6413.32],
        [7484.24],
        [11004.91],
        [13238.04],
        [15690.18],
        [1784.39],
        [3902.60],
        [6511.98],
        [8642.12],
        [11002.76],
        [12927.37],
        [15081.26],
        [816.98],
        [4240.21],
        [6083.73],
        [8714.48],
        [11207.92],
        [13407.75],
        [14773.42],
        [1708.31],
        [3375.83],
        [5217.25],
        [8949.42],
        [10577.26],
        [7036.20],
        [9753.39],
        [4115.90],
        [14742.91],
        [8550.18],
        [11627.22],
        [6953.26],
        [2813.42],
        [8729.29],
        [12629.35],
        [13545.91],
        [8052.14],
        [4572.10],
        [13189.05],
        [1858.19],
        [11778.70],
        [9815.16],
        [5765.51],
    ]

    well.y = [
        [2303.54],
        [1572.02],
        [1799.36],
        [769.30],
        [933.22],
        [617.93],
        [1089.33],
        [4693.61],
        [4130.63],
        [5033.77],
        [4741.86],
        [3265.07],
        [3976.69],
        [4438.75],
        [7711.68],
        [6203.38],
        [6537.44],
        [7846.77],
        [6209.06],
        [7310.93],
        [5950.01],
        [9825.31],
        [9854.94],
        [9511.38],
        [9696.44],
        [9556.22],
        [9418.56],
        [10100.12],
        [11599.21],
        [12943.38],
        [13075.00],
        [11651.88],
        [12384.90],
        [13126.17],
        [12151.69],
        [15786.73],
        [14459.76],
        [14699.77],
        [14669.82],
        [15459.03],
        [3535.69],
        [7798.34],
        [7700.00],
        [14695.73],
        [2569.95],
        [7976.76],
        [14684.10],
        [11515.41],
        [6419.55],
        [14623.74],
        [2218.00],
        [13175.26],
        [3405.63],
        [11224.96],
        [13641.15],
        [2156.20],
        [13266.84],
        [8047.71],
    ]

    well.type = [
        [[2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2]],
        [[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]]
    ]
    well.type = [item for sublist in well.type for item in sublist]

    well.constraint = [
        [1000.0],
        [1000.0],
        [1000.0],
        [1000.0],
        [1000.0],
        [1000.0],
        [1000.0],
        [1000.0],
        [1000.0],
        [1000.0],
        [1000.0],
        [1000.0],
        [1000.0],
        [1000.0],
        [1000.0],
        [1000.0],
        [1000.0],
        [1000.0],
        [1000.0],
        [1000.0],
        [1000.0],
        [1000.0],
        [1000.0],
        [1000.0],
        [1000.0],
        [1000.0],
        [1000.0],
        [1000.0],
        [1000.0],
        [1000.0],
        [1000.0],
        [1000.0],
        [1000.0],
        [1000.0],
        [1000.0],
        [1000.0],
        [1000.0],
        [1000.0],
        [1000.0],
        [1000.0],
        [1500.00],
        [1500.00],
        [1500.00],
        [1500.00],
        [1500.00],
        [1500.00],
        [1500.00],
        [1500.00],
        [1500.00],
        [1500.00],
        [1500.00],
        [1500.00],
        [1500.00],
        [1500.00],
        [1500.00],
        [1500.00],
        [1500.00],
        [1500.00],
    ]

    well.rw = [[0.25]]*58
    well.skin = [[0]]*58
    well.direction = [["v"]]*58

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
