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
import argparse
from datetime import datetime

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

# CLI overrides
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--tfinal', type=float, default=None)
parser.add_argument('--dt', type=float, default=None)
parser.add_argument('--realloc_days', type=float, default=30.0)
parser.add_argument('--no-plots', action='store_true')
args, _ = parser.parse_known_args()

if args.tfinal is not None:
    numerical.tfinal = float(args.tfinal)
if args.dt is not None:
    numerical.dt = float(args.dt)

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

# Per-well phase rates (STB/day) and cumulative volumes (STB)
well.water_rate_stb = np.zeros((len(well.x), nmax+1))
well.oil_rate_stb   = np.zeros((len(well.x), nmax+1))
well.cum_water_prod_stb = np.zeros((len(well.x), nmax+1))  # producers (production only)
well.cum_oil_prod_stb   = np.zeros((len(well.x), nmax+1))  # producers
well.cum_water_inj_stb  = np.zeros((len(well.x), nmax+1))  # injectors (injection only)

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
    else:
        # Lightweight per-step log
        print(f"  Step {k}/{nmax} - t={t[k]:.2f}d")

    P_old = np.copy(P)   #Placeholdering the old array
    Sw_old= np.copy(Sw)   #Placeholdering the old array

    # Injector reallocation every N days (adjust rate targets to meet caps)
    if k > 0 and args.realloc_days > 0 and (abs((t[k] // args.realloc_days) - (t[k-1] // args.realloc_days)) >= 1):
        # Identify injectors (type == 1)
        injectors_mask = np.array([tp[0] == 1 for tp in well.type])
        inj_idx = np.where(injectors_mask)[0]
        if inj_idx.size > 0:
            # Determine period index
            period = int(t[k] // args.realloc_days)
            # Desired targets from schedule if provided
            rates = np.array([well.constraint[i][0] for i in inj_idx], dtype=float)
            if hasattr(numerical, 'schedule_inj_rate') and period < getattr(numerical, 'schedule_inj_rate').shape[0]:
                sched_row = getattr(numerical, 'schedule_inj_rate')[period]
                # If schedule has exact injector count, apply directly
                if sched_row.shape[0] == inj_idx.size:
                    rates = sched_row.astype(float)
                else:
                    # Fallback: proportional map by index min length
                    nmin = min(inj_idx.size, sched_row.shape[0])
                    rates[:nmin] = sched_row[:nmin]
            total = rates.sum()
            # Caps from numerical (from input file)
            total_min = getattr(numerical, 'inj_total_min', 0.0)
            total_max = getattr(numerical, 'inj_total_max', 1e12)
            # Per-well max assumed from config.rate_max (store as attribute on well if present), else large
            per_max = np.full_like(rates, 1e12)

            target = np.clip(total, total_min, total_max)
            if total > 0:
                scaled = rates * (target / total)
            else:
                # If zero, distribute equally
                scaled = np.full_like(rates, target / len(rates))
            # Clamp per-well
            scaled = np.minimum(scaled, per_max)
            # Write back
            for j, i in enumerate(inj_idx):
                well.constraint[i][0] = float(scaled[j])
            print(f"[REALLOC] day {t[k]:.1f}: injector rates set to {scaled.round(2).tolist()} (sum={scaled.sum():.1f})")

        # Producers schedule (BHP or choke)
        producers_mask = np.array([tp[0] == 2 for tp in well.type])
        prod_idx = np.where(producers_mask)[0]
        if prod_idx.size > 0:
            period = int(t[k] // args.realloc_days)
            mode = getattr(numerical, 'producer_control_mode', 'bhp')
            if mode == 'bhp' and hasattr(numerical, 'schedule_prod_bhp') and period < getattr(numerical, 'schedule_prod_bhp').shape[0]:
                row = getattr(numerical, 'schedule_prod_bhp')[period]
                for j, i in enumerate(prod_idx):
                    if j < row.shape[0]:
                        well.pwf[i,0] = float(row[j])
                print(f"[REALLOC] day {t[k]:.1f}: producer BHPs set (first few) to {row[:min(5,row.shape[0])].round(1).tolist()}")
            if mode == 'choke' and hasattr(numerical, 'schedule_prod_choke') and period < getattr(numerical, 'schedule_prod_choke').shape[0]:
                row = getattr(numerical, 'schedule_prod_choke')[period]
                for j, i in enumerate(prod_idx):
                    if j < row.shape[0]:
                        well.choke[i,0] = float(row[j])
                print(f"[REALLOC] day {t[k]:.1f}: producer chokes set (first few) to {row[:min(5,row.shape[0])].round(2).tolist()}")

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

    # Total wellbore rate in rb/day: rate-controlled terms (Qw, Qo) plus BHP-controlled productivity terms
    # For BHP-controlled wells, flow ~ J * (P_block - Pwf)
    P_block = P[well.block[:,0], 0]  # shape (n_wells,)
    # For BHP-controlled producers (type=2), add J*(P - Pwf) with negative sign (production negative). Use per-well Pwf if provided.
    bhp_term = np.zeros(len(well.x))
    producers_mask = np.array([tp[0] == 2 for tp in well.type])
    # Per-well BHP
    pwf_arr = None
    if hasattr(well, 'pwf'):
        pwf_arr = well.pwf[:,0]
    else:
        pwf_arr = np.full(len(well.x), reservoir.Pwf)
    # Choke scaling (0-1) for producers if using choke mode
    choke = np.ones(len(well.x))
    if getattr(numerical, 'producer_control_mode', 'bhp') == 'choke' and hasattr(well, 'choke'):
        choke = well.choke[:,0]
    bhp_term[producers_mask] = - choke[producers_mask] * (well.Jwvec[producers_mask,0] + well.Jovec[producers_mask,0]) * (P_block[producers_mask] - pwf_arr[producers_mask])
    well.Qwf[:,k]  = (Qw[well.block[:,0],0].toarray())[:,0] + (Qo[well.block[:,0],0].toarray())[:,0] + bhp_term

    for i in range(0,len(well.x)):
        kblock  = well.block[i][0]
        krw,kro = rel_perm(petro,Sw[kblock,0])
        M = (kro*fluid.muw[kblock,0])/(krw*fluid.muo[kblock,0])
        well.fw[i,k] = 1/(1+M)
        well.Jwvectime[i,k] = well.Jwvec[i,0]
        well.Jovectime[i,k] = well.Jovec[i,0]
        well.Jvectime[i,k]  = J[kblock,kblock]
        well.Q[i,k]         = Q[kblock,0]

    # Track per-well and field cumulative volumes
    for i in range(0, len(well.x)):
        kblock = well.block[i][0]
        total_rate_rb = well.Qwf[i,k]  # rb/day (positive = injection, negative = production)
        fw_well = well.fw[i,k]
        water_rate_rb = total_rate_rb * fw_well
        oil_rate_rb   = total_rate_rb * (1 - fw_well)

        water_rate_stb = water_rate_rb / fluid.Bw[kblock,0]
        oil_rate_stb   = oil_rate_rb / fluid.Bo[kblock,0]

        # Store instantaneous rates
        well.water_rate_stb[i, k] = water_rate_stb
        well.oil_rate_stb[i, k]   = oil_rate_stb

        # Separate production vs injection contributions
        water_prod = max(0.0, -water_rate_stb)  # STB/day
        oil_prod   = max(0.0, -oil_rate_stb)    # STB/day
        water_inj  = max(0.0,  water_rate_stb)  # STB/day

        # Increment cumulative per-well volumes
        well.cum_water_prod_stb[i, k] = (well.cum_water_prod_stb[i, k-1] if k > 0 else 0.0) + water_prod * numerical.dt
        well.cum_oil_prod_stb[i, k]   = (well.cum_oil_prod_stb[i, k-1]   if k > 0 else 0.0) + oil_prod   * numerical.dt
        well.cum_water_inj_stb[i, k]  = (well.cum_water_inj_stb[i, k-1]  if k > 0 else 0.0) + water_inj  * numerical.dt

        # Field totals
        cumulative_water_injected += water_inj * numerical.dt
        cumulative_water_produced += water_prod * numerical.dt
        cumulative_oil_produced   += oil_prod   * numerical.dt

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

# Save results into a unique run directory
import os
base_results = '../results/impes_sim'
run_id = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
output_dir = os.path.join(base_results, f'run_{run_id}')
os.makedirs(output_dir, exist_ok=True)
output_file = f'{output_dir}/Phase1_n{numerical.N}_t{int(t[k])}_days.npz'
npz_data = {
    'P_plot': P_plot,
    'Sw_plot': Sw_plot,
    'Nx': numerical.Nx,
    'Ny': numerical.Ny,
    'fw': fw,
    't': t,
    'x1': numerical.x1,
    'y1': numerical.y1,
    'well_x': well.x,
    'well_y': well.y,
    'well_fw': well.fw,
    'well_Qwf': well.Qwf,
    'well_water_rate_stb': well.water_rate_stb,
    'well_oil_rate_stb': well.oil_rate_stb,
    'well_cum_water_prod_stb': well.cum_water_prod_stb,
    'well_cum_oil_prod_stb': well.cum_oil_prod_stb,
    'well_cum_water_inj_stb': well.cum_water_inj_stb,
    'cumulative_water_injected': cumulative_water_injected,
    'cumulative_water_produced': cumulative_water_produced,
    'cumulative_oil_produced': cumulative_oil_produced,
}
# Include control series for surrogate training if available
if hasattr(well, 'constrainttime'):
    npz_data['well_constrainttime'] = well.constrainttime
if hasattr(well, 'pwf'):
    npz_data['producer_pwf'] = well.pwf
if hasattr(well, 'choke'):
    npz_data['producer_choke'] = well.choke

np.savez(output_file, **npz_data)
print(f"\nResults saved to: {output_file}")

# Post process for visualization
P_plot[np.argwhere(numerical.D==0),:] = np.nan
Sw_plot[np.argwhere(numerical.D ==0),:] = np.nan

if not args.no_plots:
    print("\nGenerating visualization plots...")

    # Single-time snapshots at 10% of simulation time
    ten_idx = max(0, int(0.10 * k))
    P_2D_10 = P_plot[:, ten_idx].reshape((numerical.Ny, numerical.Nx))
    Sw_2D_10 = Sw_plot[:, ten_idx].reshape((numerical.Ny, numerical.Nx))

    plt.figure(figsize=(7,6))
    plt.title(f'Pressure @ {t[ten_idx]:.1f} days')
    im10p = plt.contourf(P_2D_10, levels=20, cmap='jet')
    plt.colorbar(im10p, label='Pressure (psi)')
    plt.xlabel('X (cells)'); plt.ylabel('Y (cells)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pressure_t10pct.png', dpi=150)

    plt.figure(figsize=(7,6))
    plt.title(f'Water Saturation @ {t[ten_idx]:.1f} days')
    im10s = plt.contourf(Sw_2D_10, levels=20, cmap='Blues', vmin=petro.Swr, vmax=1.0-petro.Sor)
    plt.colorbar(im10s, label='Sw')
    plt.xlabel('X (cells)'); plt.ylabel('Y (cells)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/saturation_t10pct.png', dpi=150)

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
    plot_file = f'{output_dir}/Phase1_pressure_evolution.png'
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
    sat_plot_file = f'{output_dir}/Phase1_saturation_evolution.png'
    plt.savefig(sat_plot_file, dpi=150, bbox_inches='tight')
    print(f"Saturation evolution plot saved to: {sat_plot_file}")

    # Additional charts: cumulative per-well and field totals
    producers_mask = np.array([tp[0] == 2 for tp in well.type])  # 2 = BHP (producers)
    injectors_mask = np.array([tp[0] == 1 for tp in well.type])  # 1 = rate (injectors)

    time_days = t[:k+1]

    # 1) Cumulative oil production per producer well
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    for i in np.where(producers_mask)[0]:
        ax3.plot(time_days, well.cum_oil_prod_stb[i, :k+1], label=f"PROD_{i:03d}")
    ax3.set_xlabel('Time (days)')
    ax3.set_ylabel('Cumulative Oil Production (STB)')
    ax3.set_title('Cumulative Oil Production per Producer Well')
    ax3.grid(True, alpha=0.3)
    if producers_mask.sum() <= 15:
        ax3.legend(ncol=2, fontsize=8)
    out_oil_perwell = f"{output_dir}/cum_oil_per_producer.png"
    plt.tight_layout()
    plt.savefig(out_oil_perwell, dpi=150)

    # 2) Cumulative water production per producer well
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    for i in np.where(producers_mask)[0]:
        ax4.plot(time_days, well.cum_water_prod_stb[i, :k+1], label=f"PROD_{i:03d}")
    ax4.set_xlabel('Time (days)')
    ax4.set_ylabel('Cumulative Water Production (STB)')
    ax4.set_title('Cumulative Water Production per Producer Well')
    ax4.grid(True, alpha=0.3)
    if producers_mask.sum() <= 15:
        ax4.legend(ncol=2, fontsize=8)
    out_water_perwell = f"{output_dir}/cum_water_per_producer.png"
    plt.tight_layout()
    plt.savefig(out_water_perwell, dpi=150)

    # 3) Cumulative water injection per injector well
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    for i in np.where(injectors_mask)[0]:
        ax5.plot(time_days, well.cum_water_inj_stb[i, :k+1], label=f"INJ_{i:03d}")
    ax5.set_xlabel('Time (days)')
    ax5.set_ylabel('Cumulative Water Injection (STB)')
    ax5.set_title('Cumulative Water Injection per Injector Well')
    ax5.grid(True, alpha=0.3)
    if injectors_mask.sum() <= 15:
        ax5.legend(ncol=2, fontsize=8)
    out_inj_perwell = f"{output_dir}/cum_water_per_injector.png"
    plt.tight_layout()
    plt.savefig(out_inj_perwell, dpi=150)

    # 4) Field totals: dual y-axis — oil (left), water (right)
    fig6, ax_left = plt.subplots(figsize=(10, 6))
    total_cum_oil_prod = well.cum_oil_prod_stb[:, :k+1].sum(axis=0)
    total_cum_water_prod = well.cum_water_prod_stb[:, :k+1].sum(axis=0)
    total_cum_water_inj = well.cum_water_inj_stb[:, :k+1].sum(axis=0)

    lns1 = ax_left.plot(time_days, total_cum_oil_prod, color='tab:orange', label='Total Cum Oil Prod (STB)')
    ax_left.set_xlabel('Time (days)')
    ax_left.set_ylabel('Oil (STB)', color='tab:orange')
    ax_left.tick_params(axis='y', labelcolor='tab:orange')
    ax_left.grid(True, alpha=0.3)

    ax_right = ax_left.twinx()
    lns2 = ax_right.plot(time_days, total_cum_water_prod, color='tab:blue', label='Total Cum Water Prod (STB)')
    lns3 = ax_right.plot(time_days, total_cum_water_inj, color='tab:green', label='Total Cum Water Inj (STB)')
    ax_right.set_ylabel('Water (STB)', color='tab:blue')
    ax_right.tick_params(axis='y', labelcolor='tab:blue')

    fig6.suptitle('Field Cumulative Volumes')
    lines = lns1 + lns2 + lns3
    labels = [l.get_label() for l in lines]
    ax_left.legend(lines, labels, loc='upper left')

    out_field_totals = f"{output_dir}/cum_field_totals.png"
    plt.tight_layout()
    plt.savefig(out_field_totals, dpi=150)
    print(f"Additional plots saved to: {out_oil_perwell}, {out_water_perwell}, {out_inj_perwell}, {out_field_totals}")

print("\nPhase 1 IMPES simulation completed successfully!")
print("="*70)
