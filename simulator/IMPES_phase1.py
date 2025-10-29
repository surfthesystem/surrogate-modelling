"""
Phase 1 Reservoir - IMPES Simulation
Simplified IMPES solver for GNN-LSTM Surrogate Model Project

Based on: Mohammad Afzal Shadab's IMPES code
Modified for: Phase 1 waterflooding simulation
"""

#import inbuilt libraries
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, identity, eye
from scipy.sparse.linalg import inv
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import time as timer
from math import floor, ceil
import warnings
warnings.filterwarnings("ignore")

#importing personal libraries
from input_file_phase1 import inputfile, fluid, reservoir, petro, numerical, BC, IC, well
from myarrays import myarrays
from updatewells import updatewells
from rel_perm import rel_perm
from spdiaginv import spdiaginv
from init_plot import initial_plot
from postprocess import postprocess

print("\n" + "="*70)
print("PHASE 1 RESERVOIR - IMPES SIMULATION STARTING")
print("="*70)

tprogstart= timer.time()

#Implicit pressure and explicit saturation for update
#looping through time
nmax  = ceil(numerical.tfinal / numerical.dt)
t = np.empty((nmax + 1))  #dimensional time
t[0]= 0
t_D = np.empty((nmax + 1))#non dimensional time
t_D[0]= 0
k     = 0
PV    = 0
P     = np.copy(IC.P)
Pw    = np.copy(IC.Pw)
Sw    = np.array(np.copy(IC.Sw))
Sw_hyst=np.zeros((numerical.N,2))
Sw_hyst[:,0]=Sw[:,0]

fw   = np.empty((nmax +1 ))          #fractional flow of wetting phase
fw[0]= 0
P_plot= np.zeros((numerical.N,nmax + 1)) #matrix to save pressure
P_plot[:,0] = IC.P[:,0]
Sw_plot= np.zeros((numerical.N, nmax + 1)) #matrix to save saturation
Sw_plot[:,0]= IC.Sw[:,0]
well.typetime       = np.kron(np.ones((nmax,1)),np.transpose(well.type))
well.constrainttime = np.kron(np.ones((nmax,1)),np.transpose(well.constraint))
well.fw             = np.zeros((len(well.x),nmax+1))
well.Qwf            = np.zeros((len(well.x),nmax+1))
well.Jwvectime      = np.zeros((len(well.x),nmax+1))
well.Jovectime      = np.zeros((len(well.x),nmax+1))
well.Jvectime       = np.zeros((len(well.x),nmax+1))
well.Q              = np.zeros((len(well.x),nmax+1))

# Track production/injection for mass balance
cumulative_water_injected = 0.0  # STB
cumulative_water_produced = 0.0  # STB
cumulative_oil_produced = 0.0    # STB

print(f"\nStarting time loop: {nmax} steps")
print(f"Progress updates at 25%, 50%, 75%, 100%")

while (t[k] < numerical.tfinal): #time marching
    # Progress reporting
    if k % max(1, nmax//10) == 0:
        progress = 100 * k / nmax
        print(f"  Step {k}/{nmax} ({progress:.1f}%) - Time = {t[k]:.2f} days")
        print(f"    Avg Pressure: {P.mean():.1f} psi, Avg Sw: {Sw.mean():.3f}")

    P_old = np.copy(P)   #Placeholdering the old array
    Sw_old= np.copy(Sw)   #Placeholdering the old array

    # NO WELL SWITCHING - wells remain fixed throughout simulation
    # This is different from original IMPES.py which had problem-specific switching

    #Calculating the arrays
    Tw, To, T, d11, d12, d21, d22, D, G, Pc, Pw = myarrays(fluid,reservoir,petro,numerical,BC,P,Sw,Sw_hyst)

    #updating the wells
    well, Qw, Qo, Jw, Jo = updatewells(reservoir,fluid,numerical,petro,P,Sw,well)

    J = -d22 @ ( spdiaginv(d12) @ Jw ) + Jo

    Q = -d22 @ ( spdiaginv(d12) @ Qw ) + Qo + reservoir.Pwf * J @ np.ones((numerical.N,1))

    if numerical.method == 'IMPES':
        IM = T + J + D          #implicit part coefficient
        EX = D @ P_old + Q + G  #explicit part or RHS

        P = np.transpose([spsolve(IM,EX)]) #solving IM*P = EX or Ax=B
        Sw = Sw + spdiaginv(d12) @ (-Tw @ (P - (fluid.rhow/144.0) * numerical.D - Pc) - d11 @ (P - P_old) + Qw + Jw @ (reservoir.Pwf - P))         #explicit saturation

    # Enforce saturation bounds
    Sw[Sw > 1.0] = 1.0
    Sw[Sw < petro.Swr] = petro.Swr  # Use residual saturation from petro object

    # Hysteresis tracking
    for i in range(0, numerical.N):
        if Sw[i,0] > Sw_old[i,0] and Sw_hyst[i,1] == 0:  # [i,1] is a flag
            Sw_hyst[i,0] = Sw[i,0]
            Sw_hyst[i,1] = 1.0
        elif Sw[i,0] < Sw_old[i,0]:
            Sw_hyst[i,0] = Sw[i,0]

    k = k+1
    P_plot[:,k] = P[:,0]
    Sw_plot[:,k]= np.array(Sw)[:,0]
    t[k]= t[k-1] + numerical.dt
    t_D[k]= well.constraint[0][0]*t[k-1]/(reservoir.L*reservoir.W*reservoir.h*reservoir.phi[0,0])

    well.Qwf[:,k]  = (Qw[well.block,0].toarray())[:,0] + (Qo[well.block,0].toarray())[:,0] + (well.Jwvec[:,0] + well.Jovec[:,0]) * reservoir.Pwf

    for i in range(0,len(well.x)):
        kblock  = well.block[i][0]
        krw,kro = rel_perm(petro,Sw[kblock,0])
        M = (kro*fluid.muw[kblock,0])/(krw*fluid.muo[kblock,0])
        well.fw[i,k] = 1/(1+M)
        well.Jwvectime[i,k] = well.Jwvec[i,0]
        well.Jovectime[i,k] = well.Jovec[i,0]
        well.Jvectime[i,k]  = J[kblock,kblock]
        well.Q[i,k]         = Q[kblock,0]

    # Track cumulative production/injection
    # Sum all well rates for this timestep
    for i in range(0, len(well.x)):
        kblock = well.block[i][0]
        # Total fluid rate at wellbore
        total_rate_rb = well.Qwf[i,k]  # rb/day
        # Calculate phase rates
        fw_well = well.fw[i,k]
        water_rate_rb = total_rate_rb * fw_well
        oil_rate_rb = total_rate_rb * (1 - fw_well)

        # Convert to stock tank barrels
        water_rate_stb = water_rate_rb / fluid.Bw[kblock,0]  # STB/day
        oil_rate_stb = oil_rate_rb / fluid.Bo[kblock,0]      # STB/day

        # Accumulate (negative = production, positive = injection)
        if water_rate_stb > 0:
            cumulative_water_injected += water_rate_stb * numerical.dt
        else:
            cumulative_water_produced += abs(water_rate_stb) * numerical.dt

        if oil_rate_stb < 0:
            cumulative_oil_produced += abs(oil_rate_stb) * numerical.dt

P_plot[np.argwhere(reservoir.permx < 0.0001)] = np.nan

tprogend= timer.time()
elapsed_time = tprogend - tprogstart

print(f"\n{'='*70}")
print("SIMULATION COMPLETE!")
print(f"{'='*70}")
print(f"Time elapsed: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
print(f"Timesteps completed: {k}")
print(f"Simulation time: {t[k]:.2f} days")
print(f"\nFinal Conditions:")
print(f"  Average Pressure: {P.mean():.1f} psi (Initial: {IC.P.mean():.1f} psi)")
print(f"  Pressure Change: {P.mean() - IC.P.mean():.1f} psi")
print(f"  Average Water Saturation: {Sw.mean():.3f} (Initial: {IC.Sw.mean():.3f})")
print(f"  Water Saturation Change: {Sw.mean() - IC.Sw.mean():.3f}")
print(f"\nMass Balance Check:")
print(f"  Cumulative Water Injected: {cumulative_water_injected:.1f} STB")
print(f"  Cumulative Water Produced: {cumulative_water_produced:.1f} STB")
print(f"  Cumulative Oil Produced: {cumulative_oil_produced:.1f} STB")
print(f"  Total Fluid Produced: {cumulative_water_produced + cumulative_oil_produced:.1f} STB")
print(f"  Injection/Production Ratio: {cumulative_water_injected/(cumulative_water_produced + cumulative_oil_produced + 1e-10):.3f}")
print(f"{'='*70}")

# Save results
output_file = f'../../../results/impes_sim/Phase1_n{numerical.N}_t{int(t[k])}_days.npz'
import os
os.makedirs('../../../results/impes_sim', exist_ok=True)
np.savez(output_file,
         P_plot = P_plot,
         Sw_plot = Sw_plot,
         Nx = numerical.Nx,
         Ny = numerical.Ny,
         fw = fw,
         t = t,
         x1 = numerical.x1,
         y1 = numerical.y1,
         well_x = well.x,
         well_y = well.y,
         well_fw = well.fw,
         well_Qwf = well.Qwf,
         cumulative_water_injected = cumulative_water_injected,
         cumulative_water_produced = cumulative_water_produced,
         cumulative_oil_produced = cumulative_oil_produced)
print(f"\nResults saved to: {output_file}")

# Post process for visualization
P_plot[np.argwhere(numerical.D==0),:] = np.nan
Sw_plot[np.argwhere(numerical.D ==0),:] = np.nan

print("\nGenerating visualization plots...")

# Create plots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(f'Phase 1 IMPES Simulation - {int(t[k])} Days', fontsize=16, fontweight='bold')

# Select timesteps to plot
if k > 5:
    plot_indices = [0, k//4, k//2, 3*k//4, k-1, k]
else:
    plot_indices = [0, k//2, k]

for idx, (ax, ti) in enumerate(zip(axes.flat, plot_indices)):
    if ti <= k:
        # Reshape pressure for plotting
        P_2D = P_plot[:,ti].reshape((numerical.Ny, numerical.Nx))

        im = ax.contourf(P_2D, levels=20, cmap='jet')
        ax.set_title(f'Pressure - Time = {t[ti]:.1f} days')
        ax.set_xlabel('X (cells)')
        ax.set_ylabel('Y (cells)')

        # Mark wells
        for wi in range(len(well.x)):
            x_coord = well.x[wi][0] / (reservoir.L/numerical.Nx)
            y_coord = well.y[wi][0] / (reservoir.W/numerical.Ny)

            if wi < 10:  # Producers
                ax.plot(x_coord, y_coord, 'wo', markersize=8, markeredgecolor='red', markeredgewidth=2)
            else:  # Injectors
                ax.plot(x_coord, y_coord, 'w^', markersize=8, markeredgecolor='cyan', markeredgewidth=2)

        plt.colorbar(im, ax=ax, label='Pressure (psi)')

plt.tight_layout()
plot_file = '../../../results/impes_sim/Phase1_pressure_evolution.png'
plt.savefig(plot_file, dpi=150, bbox_inches='tight')
print(f"Pressure evolution plot saved to: {plot_file}")

# Saturation plot
fig2, axes2 = plt.subplots(2, 3, figsize=(18, 12))
fig2.suptitle(f'Phase 1 IMPES Simulation - Water Saturation - {int(t[k])} Days', fontsize=16, fontweight='bold')

for idx, (ax, ti) in enumerate(zip(axes2.flat, plot_indices)):
    if ti <= k:
        # Reshape saturation for plotting
        Sw_2D = Sw_plot[:,ti].reshape((numerical.Ny, numerical.Nx))

        im = ax.contourf(Sw_2D, levels=20, cmap='Blues', vmin=petro.Swr, vmax=1.0-petro.Sor)
        ax.set_title(f'Water Saturation - Time = {t[ti]:.1f} days')
        ax.set_xlabel('X (cells)')
        ax.set_ylabel('Y (cells)')

        # Mark wells
        for wi in range(len(well.x)):
            x_coord = well.x[wi][0] / (reservoir.L/numerical.Nx)
            y_coord = well.y[wi][0] / (reservoir.W/numerical.Ny)

            if wi < 10:  # Producers
                ax.plot(x_coord, y_coord, 'ko', markersize=8, markeredgecolor='red', markeredgewidth=2)
            else:  # Injectors
                ax.plot(x_coord, y_coord, 'k^', markersize=8, markeredgecolor='green', markeredgewidth=2)

        plt.colorbar(im, ax=ax, label='Water Saturation')

plt.tight_layout()
sat_plot_file = '../../../results/impes_sim/Phase1_saturation_evolution.png'
plt.savefig(sat_plot_file, dpi=150, bbox_inches='tight')
print(f"Saturation evolution plot saved to: {sat_plot_file}")

print("\nPhase 1 IMPES simulation completed successfully!")
print("="*70)
