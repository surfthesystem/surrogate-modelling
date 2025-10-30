#!/usr/bin/env python3
"""
Generate visualization plots for all simulation results.
Creates pressure/saturation maps and performance plots for each scenario.
"""
import argparse
import json
import multiprocessing as mp
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def generate_plots_for_scenario(args_tuple):
    """Generate all plots for a single scenario."""
    scenario_dir, scenario_idx, total_scenarios = args_tuple

    scenario_name = scenario_dir.name
    npz_file = scenario_dir / f'{scenario_name}.npz'

    if not npz_file.exists():
        return {'scenario': scenario_name, 'status': 'npz_not_found'}

    try:
        # Load data
        data = np.load(npz_file)

        # Extract data
        t = data['t']
        Nx = int(data['Nx'])
        Ny = int(data['Ny'])
        n_timesteps = len(t)

        # Reshape flattened arrays to (Nx, Ny, n_timesteps)
        P_plot = data['P_plot'].reshape(Nx, Ny, n_timesteps)
        Sw_plot = data['Sw_plot'].reshape(Nx, Ny, n_timesteps)

        # Reshape coordinates to (Nx, Ny)
        x1 = data['x1'].reshape(Nx, Ny)
        y1 = data['y1'].reshape(Nx, Ny)

        # Flatten well coordinates to 1D
        well_x = data['well_x'].flatten()
        well_y = data['well_y'].flatten()

        # Well performance data
        well_oil_rate = data['well_oil_rate_stb']
        well_water_rate = data['well_water_rate_stb']
        well_cum_oil = data['well_cum_oil_prod_stb']
        well_cum_water_prod = data['well_cum_water_prod_stb']
        well_cum_water_inj = data['well_cum_water_inj_stb']

        n_wells = well_oil_rate.shape[0]

        # Identify producers and injectors (producers have positive oil rates)
        is_producer = well_oil_rate[:, -1] > 0
        producer_indices = np.where(is_producer)[0]
        injector_indices = np.where(~is_producer)[0]

        # Compute field totals from well data (sum across all wells)
        cum_oil_prod = well_cum_oil.sum(axis=0)
        cum_water_prod = well_cum_water_prod[producer_indices, :].sum(axis=0)
        cum_water_inj = well_cum_water_inj[injector_indices, :].sum(axis=0)

        # Create plots directory
        plots_dir = scenario_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)

        # Select timesteps for field plots (beginning, 1/3, 2/3, end)
        plot_timesteps = [0, n_timesteps // 3, 2 * n_timesteps // 3, n_timesteps - 1]

        # ===== 1. PRESSURE EVOLUTION PLOTS =====
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        axes = axes.flatten()

        for idx, tidx in enumerate(plot_timesteps):
            ax = axes[idx]
            P_field = P_plot[:, :, tidx]

            im = ax.contourf(x1, y1, P_field, levels=20, cmap='jet')
            ax.contour(x1, y1, P_field, levels=10, colors='black', linewidths=0.5, alpha=0.3)

            # Plot wells
            ax.plot(well_x[producer_indices], well_y[producer_indices],
                   'wo', markersize=8, markeredgecolor='black', markeredgewidth=2, label='Producers')
            ax.plot(well_x[injector_indices], well_y[injector_indices],
                   'ws', markersize=8, markeredgecolor='black', markeredgewidth=2, label='Injectors')

            ax.set_xlabel('X (m)', fontsize=10)
            ax.set_ylabel('Y (m)', fontsize=10)
            ax.set_title(f'Pressure at t = {t[tidx]:.1f} days', fontsize=12)
            ax.set_aspect('equal')
            ax.legend(loc='upper right', fontsize=8)

            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Pressure (psi)', fontsize=10)

        plt.tight_layout()
        plt.savefig(plots_dir / 'pressure_evolution.png', dpi=150, bbox_inches='tight')
        plt.close()

        # ===== 2. SATURATION EVOLUTION PLOTS =====
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        axes = axes.flatten()

        for idx, tidx in enumerate(plot_timesteps):
            ax = axes[idx]
            Sw_field = Sw_plot[:, :, tidx]

            im = ax.contourf(x1, y1, Sw_field, levels=20, cmap='Blues', vmin=0, vmax=1)
            ax.contour(x1, y1, Sw_field, levels=10, colors='black', linewidths=0.5, alpha=0.3)

            # Plot wells
            ax.plot(well_x[producer_indices], well_y[producer_indices],
                   'wo', markersize=8, markeredgecolor='black', markeredgewidth=2, label='Producers')
            ax.plot(well_x[injector_indices], well_y[injector_indices],
                   'ws', markersize=8, markeredgecolor='black', markeredgewidth=2, label='Injectors')

            ax.set_xlabel('X (m)', fontsize=10)
            ax.set_ylabel('Y (m)', fontsize=10)
            ax.set_title(f'Water Saturation at t = {t[tidx]:.1f} days', fontsize=12)
            ax.set_aspect('equal')
            ax.legend(loc='upper right', fontsize=8)

            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Water Saturation', fontsize=10)

        plt.tight_layout()
        plt.savefig(plots_dir / 'saturation_evolution.png', dpi=150, bbox_inches='tight')
        plt.close()

        # ===== 3. WELL RATES (Individual Wells) =====
        # Producers
        if len(producer_indices) > 0:
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))

            # Oil rates
            for i in producer_indices:
                axes[0].plot(t, well_oil_rate[i, :], label=f'Prod {i}', linewidth=2)
            axes[0].set_xlabel('Time (days)', fontsize=12)
            axes[0].set_ylabel('Oil Rate (STB/day)', fontsize=12)
            axes[0].set_title('Producer Oil Rates', fontsize=14, fontweight='bold')
            axes[0].grid(True, alpha=0.3)
            axes[0].legend(ncol=2, fontsize=8)

            # Water rates
            for i in producer_indices:
                axes[1].plot(t, well_water_rate[i, :], label=f'Prod {i}', linewidth=2)
            axes[1].set_xlabel('Time (days)', fontsize=12)
            axes[1].set_ylabel('Water Rate (STB/day)', fontsize=12)
            axes[1].set_title('Producer Water Rates', fontsize=14, fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            axes[1].legend(ncol=2, fontsize=8)

            plt.tight_layout()
            plt.savefig(plots_dir / 'producer_rates.png', dpi=150, bbox_inches='tight')
            plt.close()

        # Injectors
        if len(injector_indices) > 0:
            fig, ax = plt.subplots(figsize=(12, 6))

            for i in injector_indices:
                ax.plot(t, -well_water_rate[i, :], label=f'Inj {i}', linewidth=2)
            ax.set_xlabel('Time (days)', fontsize=12)
            ax.set_ylabel('Injection Rate (STB/day)', fontsize=12)
            ax.set_title('Injector Water Rates', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(ncol=2, fontsize=8)

            plt.tight_layout()
            plt.savefig(plots_dir / 'injector_rates.png', dpi=150, bbox_inches='tight')
            plt.close()

        # ===== 4. CUMULATIVE PRODUCTION (Individual Wells) =====
        if len(producer_indices) > 0:
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))

            # Cumulative oil
            for i in producer_indices:
                axes[0].plot(t, well_cum_oil[i, :], label=f'Prod {i}', linewidth=2)
            axes[0].set_xlabel('Time (days)', fontsize=12)
            axes[0].set_ylabel('Cumulative Oil (STB)', fontsize=12)
            axes[0].set_title('Cumulative Oil Production by Well', fontsize=14, fontweight='bold')
            axes[0].grid(True, alpha=0.3)
            axes[0].legend(ncol=2, fontsize=8)

            # Cumulative water
            for i in producer_indices:
                axes[1].plot(t, well_cum_water_prod[i, :], label=f'Prod {i}', linewidth=2)
            axes[1].set_xlabel('Time (days)', fontsize=12)
            axes[1].set_ylabel('Cumulative Water (STB)', fontsize=12)
            axes[1].set_title('Cumulative Water Production by Well', fontsize=14, fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            axes[1].legend(ncol=2, fontsize=8)

            plt.tight_layout()
            plt.savefig(plots_dir / 'cumulative_production_by_well.png', dpi=150, bbox_inches='tight')
            plt.close()

        # ===== 5. FIELD TOTALS =====
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Cumulative oil
        axes[0, 0].plot(t, cum_oil_prod / 1e6, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Time (days)', fontsize=12)
        axes[0, 0].set_ylabel('Cumulative Oil (Million STB)', fontsize=12)
        axes[0, 0].set_title('Field Cumulative Oil Production', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)

        # Cumulative water production
        axes[0, 1].plot(t, cum_water_prod / 1e3, 'c-', linewidth=2)
        axes[0, 1].set_xlabel('Time (days)', fontsize=12)
        axes[0, 1].set_ylabel('Cumulative Water Prod (Thousand STB)', fontsize=12)
        axes[0, 1].set_title('Field Cumulative Water Production', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)

        # Cumulative water injection
        axes[1, 0].plot(t, cum_water_inj / 1e6, 'r-', linewidth=2)
        axes[1, 0].set_xlabel('Time (days)', fontsize=12)
        axes[1, 0].set_ylabel('Cumulative Water Inj (Million STB)', fontsize=12)
        axes[1, 0].set_title('Field Cumulative Water Injection', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)

        # Oil rate (derivative of cumulative)
        oil_rate_field = np.diff(cum_oil_prod) / np.diff(t)
        axes[1, 1].plot(t[1:], oil_rate_field, 'b-', linewidth=2)
        axes[1, 1].set_xlabel('Time (days)', fontsize=12)
        axes[1, 1].set_ylabel('Field Oil Rate (STB/day)', fontsize=12)
        axes[1, 1].set_title('Field Oil Production Rate', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plots_dir / 'field_totals.png', dpi=150, bbox_inches='tight')
        plt.close()

        # ===== 6. SUMMARY PLOT (Key Metrics) =====
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # Final pressure field
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.contourf(x1, y1, P_plot[:, :, -1], levels=20, cmap='jet')
        ax1.plot(well_x[producer_indices], well_y[producer_indices],
                'wo', markersize=6, markeredgecolor='black', markeredgewidth=1.5)
        ax1.plot(well_x[injector_indices], well_y[injector_indices],
                'ws', markersize=6, markeredgecolor='black', markeredgewidth=1.5)
        ax1.set_title(f'Final Pressure (t={t[-1]:.1f} days)', fontsize=11, fontweight='bold')
        ax1.set_aspect('equal')
        plt.colorbar(im1, ax=ax1, label='Pressure (psi)')

        # Final saturation field
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.contourf(x1, y1, Sw_plot[:, :, -1], levels=20, cmap='Blues', vmin=0, vmax=1)
        ax2.plot(well_x[producer_indices], well_y[producer_indices],
                'wo', markersize=6, markeredgecolor='black', markeredgewidth=1.5)
        ax2.plot(well_x[injector_indices], well_y[injector_indices],
                'ws', markersize=6, markeredgecolor='black', markeredgewidth=1.5)
        ax2.set_title(f'Final Water Saturation (t={t[-1]:.1f} days)', fontsize=11, fontweight='bold')
        ax2.set_aspect('equal')
        plt.colorbar(im2, ax=ax2, label='Water Saturation')

        # Field production rates
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(t[1:], oil_rate_field, 'b-', linewidth=2, label='Oil Rate')
        ax3.set_xlabel('Time (days)')
        ax3.set_ylabel('Oil Rate (STB/day)', color='b')
        ax3.tick_params(axis='y', labelcolor='b')
        ax3.set_title('Field Production Rates', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        ax3b = ax3.twinx()
        water_rate_field = np.diff(cum_water_prod) / np.diff(t)
        ax3b.plot(t[1:], water_rate_field, 'c-', linewidth=2, label='Water Rate')
        ax3b.set_ylabel('Water Rate (STB/day)', color='c')
        ax3b.tick_params(axis='y', labelcolor='c')

        # Cumulative production
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(t, cum_oil_prod / 1e6, 'b-', linewidth=3, label='Oil')
        ax4.set_xlabel('Time (days)')
        ax4.set_ylabel('Cumulative Oil (Million STB)')
        ax4.set_title('Cumulative Oil Production', fontsize=11, fontweight='bold')
        ax4.grid(True, alpha=0.3)

        # Water cut
        ax5 = fig.add_subplot(gs[1, 1])
        water_cut = cum_water_prod / (cum_oil_prod + cum_water_prod + 1e-10)
        ax5.plot(t, water_cut * 100, 'c-', linewidth=3)
        ax5.set_xlabel('Time (days)')
        ax5.set_ylabel('Water Cut (%)')
        ax5.set_title('Field Water Cut', fontsize=11, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim([0, 100])

        # Recovery factor (assuming OOIP)
        ax6 = fig.add_subplot(gs[1, 2])
        # Simple OOIP estimate (this is approximate)
        pore_volume = Nx * Ny * 50 * 50 * 50  # ft^3 (assuming 50 ft thickness)
        porosity_avg = 0.2  # average
        Swi = 0.2  # initial water saturation
        Bo = 1.2  # oil formation volume factor
        OOIP = pore_volume * porosity_avg * (1 - Swi) / Bo / 5.615  # STB
        recovery_factor = cum_oil_prod / OOIP * 100
        ax6.plot(t, recovery_factor, 'g-', linewidth=3)
        ax6.set_xlabel('Time (days)')
        ax6.set_ylabel('Recovery Factor (%)')
        ax6.set_title(f'Oil Recovery Factor (OOIP~{OOIP/1e6:.1f}M STB)', fontsize=11, fontweight='bold')
        ax6.grid(True, alpha=0.3)

        plt.suptitle(f'{scenario_name.upper()} - Simulation Summary', fontsize=16, fontweight='bold')
        plt.savefig(plots_dir / 'summary.png', dpi=150, bbox_inches='tight')
        plt.close()

        return {
            'scenario': scenario_name,
            'status': 'success',
            'plots_generated': 6
        }

    except Exception as e:
        return {
            'scenario': scenario_name,
            'status': 'failed',
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description='Generate visualization plots for all scenarios')
    parser.add_argument('--data_dir', type=str, default='results/training_data',
                       help='Directory containing scenario results')
    parser.add_argument('--workers', type=int, default=8,
                       help='Number of parallel workers')
    parser.add_argument('--max_scenarios', type=int, default=None,
                       help='Maximum number of scenarios to process (for testing)')
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / args.data_dir

    # Find all scenario directories
    scenario_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('doe_')])

    if args.max_scenarios:
        scenario_dirs = scenario_dirs[:args.max_scenarios]

    total_scenarios = len(scenario_dirs)

    if total_scenarios == 0:
        print(f"No scenario directories found in {data_dir}")
        return 1

    print(f"Found {total_scenarios} scenarios to process")
    print(f"Using {args.workers} parallel workers")
    print()

    # Prepare tasks
    tasks = [(scen_dir, i, total_scenarios) for i, scen_dir in enumerate(scenario_dirs)]

    # Process in parallel
    results = []
    success_count = 0
    fail_count = 0

    print("Generating visualization plots...")
    print()

    with mp.Pool(processes=args.workers) as pool:
        for i, result in enumerate(pool.imap_unordered(generate_plots_for_scenario, tasks)):
            results.append(result)

            if result['status'] == 'success':
                success_count += 1
                print(f"✓ [{i+1}/{total_scenarios}] {result['scenario']}: {result['plots_generated']} plot sets generated")
            else:
                fail_count += 1
                print(f"✗ [{i+1}/{total_scenarios}] {result['scenario']}: {result['status']} - {result.get('error', 'Unknown')[:80]}")

    print()
    print("=" * 80)
    print("VISUALIZATION GENERATION COMPLETE")
    print("=" * 80)
    print(f"Total scenarios: {total_scenarios}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    print()

    # Save summary
    summary_file = data_dir / 'visualization_summary.json'
    summary = {
        'total_scenarios': total_scenarios,
        'successful': success_count,
        'failed': fail_count,
        'results': results
    }

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to: {summary_file.relative_to(repo_root)}")

    return 0 if fail_count == 0 else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
