"""
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
    numerical.tfinal = 10
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

    # Well configuration - ALL 58 WELLS
    # 40 producers (BHP control) + 18 injectors (rate control)
    well.x = [
        [960.04],  # PROD_000
        [3906.63],  # PROD_001
        [6029.25],  # PROD_002
        [8763.03],  # PROD_003
        [10430.73],  # PROD_004
        [13077.74],  # PROD_005
        [14881.76],  # PROD_006
        [1097.61],  # PROD_007
        [3008.45],  # PROD_008
        [5124.76],  # PROD_009
        [8987.76],  # PROD_010
        [10215.88],  # PROD_011
        [13199.87],  # PROD_012
        [14825.46],  # PROD_013
        [1336.91],  # PROD_014
        [2820.85],  # PROD_015
        [6413.32],  # PROD_016
        [7484.24],  # PROD_017
        [11004.91],  # PROD_018
        [13238.04],  # PROD_019
        [15690.18],  # PROD_020
        [1784.39],  # PROD_021
        [3902.60],  # PROD_022
        [6511.98],  # PROD_023
        [8642.12],  # PROD_024
        [11002.76],  # PROD_025
        [12927.37],  # PROD_026
        [15081.26],  # PROD_027
        [816.98],  # PROD_028
        [4240.21],  # PROD_029
        [6083.73],  # PROD_030
        [8714.48],  # PROD_031
        [11207.92],  # PROD_032
        [13407.75],  # PROD_033
        [14773.42],  # PROD_034
        [1708.31],  # PROD_035
        [3375.83],  # PROD_036
        [5217.25],  # PROD_037
        [8949.42],  # PROD_038
        [10577.26],  # PROD_039
        [7036.20],  # INJ_000
        [9753.39],  # INJ_001
        [4115.90],  # INJ_002
        [14742.91],  # INJ_003
        [8550.18],  # INJ_004
        [11627.22],  # INJ_005
        [6953.26],  # INJ_006
        [2813.42],  # INJ_007
        [8729.29],  # INJ_008
        [12629.35],  # INJ_009
        [13545.91],  # INJ_010
        [8052.14],  # INJ_011
        [4572.10],  # INJ_012
        [13189.05],  # INJ_013
        [1858.19],  # INJ_014
        [11778.70],  # INJ_015
        [9815.16],  # INJ_016
        [5765.51],  # INJ_017
    ]

    well.y = [
        [2303.54],  # PROD_000
        [1572.02],  # PROD_001
        [1799.36],  # PROD_002
        [769.30],  # PROD_003
        [933.22],  # PROD_004
        [617.93],  # PROD_005
        [1089.33],  # PROD_006
        [4693.61],  # PROD_007
        [4130.63],  # PROD_008
        [5033.77],  # PROD_009
        [4741.86],  # PROD_010
        [3265.07],  # PROD_011
        [3976.69],  # PROD_012
        [4438.75],  # PROD_013
        [7711.68],  # PROD_014
        [6203.38],  # PROD_015
        [6537.44],  # PROD_016
        [7846.77],  # PROD_017
        [6209.06],  # PROD_018
        [7310.93],  # PROD_019
        [5950.01],  # PROD_020
        [9825.31],  # PROD_021
        [9854.94],  # PROD_022
        [9511.38],  # PROD_023
        [9696.44],  # PROD_024
        [9556.22],  # PROD_025
        [9418.56],  # PROD_026
        [10100.12],  # PROD_027
        [11599.21],  # PROD_028
        [12943.38],  # PROD_029
        [13075.00],  # PROD_030
        [11651.88],  # PROD_031
        [12384.90],  # PROD_032
        [13126.17],  # PROD_033
        [12151.69],  # PROD_034
        [15786.73],  # PROD_035
        [14459.76],  # PROD_036
        [14699.77],  # PROD_037
        [14669.82],  # PROD_038
        [15459.03],  # PROD_039
        [3535.69],  # INJ_000
        [7798.34],  # INJ_001
        [7700.00],  # INJ_002
        [14695.73],  # INJ_003
        [2569.95],  # INJ_004
        [7976.76],  # INJ_005
        [14684.10],  # INJ_006
        [11515.41],  # INJ_007
        [6419.55],  # INJ_008
        [14623.74],  # INJ_009
        [2218.00],  # INJ_010
        [13175.26],  # INJ_011
        [3405.63],  # INJ_012
        [11224.96],  # INJ_013
        [13641.15],  # INJ_014
        [2156.20],  # INJ_015
        [13266.84],  # INJ_016
        [8047.71],  # INJ_017
    ]

    well.type = [
        [[2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2]],  # Producers (Type 2 = BHP)
        [[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]]   # Injectors (Type 1 = Rate)
    ]
    well.type = [item for sublist in well.type for item in sublist]

    well.constraint = [
        [1000.0],  # PROD_000 BHP (psi)
        [1000.0],  # PROD_001 BHP (psi)
        [1000.0],  # PROD_002 BHP (psi)
        [1000.0],  # PROD_003 BHP (psi)
        [1000.0],  # PROD_004 BHP (psi)
        [1000.0],  # PROD_005 BHP (psi)
        [1000.0],  # PROD_006 BHP (psi)
        [1000.0],  # PROD_007 BHP (psi)
        [1000.0],  # PROD_008 BHP (psi)
        [1000.0],  # PROD_009 BHP (psi)
        [1000.0],  # PROD_010 BHP (psi)
        [1000.0],  # PROD_011 BHP (psi)
        [1000.0],  # PROD_012 BHP (psi)
        [1000.0],  # PROD_013 BHP (psi)
        [1000.0],  # PROD_014 BHP (psi)
        [1000.0],  # PROD_015 BHP (psi)
        [1000.0],  # PROD_016 BHP (psi)
        [1000.0],  # PROD_017 BHP (psi)
        [1000.0],  # PROD_018 BHP (psi)
        [1000.0],  # PROD_019 BHP (psi)
        [1000.0],  # PROD_020 BHP (psi)
        [1000.0],  # PROD_021 BHP (psi)
        [1000.0],  # PROD_022 BHP (psi)
        [1000.0],  # PROD_023 BHP (psi)
        [1000.0],  # PROD_024 BHP (psi)
        [1000.0],  # PROD_025 BHP (psi)
        [1000.0],  # PROD_026 BHP (psi)
        [1000.0],  # PROD_027 BHP (psi)
        [1000.0],  # PROD_028 BHP (psi)
        [1000.0],  # PROD_029 BHP (psi)
        [1000.0],  # PROD_030 BHP (psi)
        [1000.0],  # PROD_031 BHP (psi)
        [1000.0],  # PROD_032 BHP (psi)
        [1000.0],  # PROD_033 BHP (psi)
        [1000.0],  # PROD_034 BHP (psi)
        [1000.0],  # PROD_035 BHP (psi)
        [1000.0],  # PROD_036 BHP (psi)
        [1000.0],  # PROD_037 BHP (psi)
        [1000.0],  # PROD_038 BHP (psi)
        [1000.0],  # PROD_039 BHP (psi)
        [1500.00],  # INJ_000 rate (STB/day)
        [1500.00],  # INJ_001 rate (STB/day)
        [1500.00],  # INJ_002 rate (STB/day)
        [1500.00],  # INJ_003 rate (STB/day)
        [1500.00],  # INJ_004 rate (STB/day)
        [1500.00],  # INJ_005 rate (STB/day)
        [1500.00],  # INJ_006 rate (STB/day)
        [1500.00],  # INJ_007 rate (STB/day)
        [1500.00],  # INJ_008 rate (STB/day)
        [1500.00],  # INJ_009 rate (STB/day)
        [1500.00],  # INJ_010 rate (STB/day)
        [1500.00],  # INJ_011 rate (STB/day)
        [1500.00],  # INJ_012 rate (STB/day)
        [1500.00],  # INJ_013 rate (STB/day)
        [1500.00],  # INJ_014 rate (STB/day)
        [1500.00],  # INJ_015 rate (STB/day)
        [1500.00],  # INJ_016 rate (STB/day)
        [1500.00],  # INJ_017 rate (STB/day)
    ]

    well.rw = [[0.25]]*58
    well.skin = [[0]]*58
    well.direction = [["v"]]*58

    # Defining numerical parameters for discretized solution
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
