"""
Post-processing and visualization for LHS Optimal Injection Study

Creates:
1. Pressure/saturation evolution plots for each simulation
2. Cumulative production plots per well (interactive)
3. Overall comparison plot
4. Sensitivity analysis

Author: Generated for Reservoir Surrogate Modeling Project
Date: 2025-10-29
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import seaborn as sns
import os
import glob
import json
from scipy.stats import spearmanr
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Configuration
RESULTS_DIR = 'lhs_results'
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
N_TIMESTEPS = 10  # Number of snapshots to show

# Create plots directory
os.makedirs(PLOTS_DIR, exist_ok=True)

# =============================================================================
# DATA EXTRACTION
# =============================================================================

def extract_simulation_results(run_dir):
    """
    Extract results from a completed simulation.

    Parameters:
    -----------
    run_dir : str
        Path to run directory

    Returns:
    --------
    results : dict
        Extracted metrics and data
    """

    results = {}

    # Load main results file
    result_files = glob.glob(os.path.join(run_dir, 'Phase1_*.npz'))

    if not result_files:
        print(f"Warning: No results file found in {run_dir}")
        return None

    data = np.load(result_files[0], allow_pickle=True)

    # Extract time series data
    results['time'] = data.get('time', np.arange(data['P'].shape[0]))
    results['pressure'] = data['P']  # (timesteps, cells)
    results['saturation'] = data['Sw']  # (timesteps, cells)

    # Extract well data if available
    if 'qwf' in data:
        results['well_rates'] = data['qwf']  # (timesteps, wells)
    if 'Pwf' in data:
        results['well_pressures'] = data['Pwf']

    # Calculate cumulative metrics
    if 'cum_oil_prod' in data:
        results['cumulative_oil'] = data['cum_oil_prod'][-1]
    else:
        # Estimate from production rates if available
        results['cumulative_oil'] = 0

    if 'cum_water_prod' in data:
        results['cumulative_water'] = data['cum_water_prod'][-1]
    else:
        results['cumulative_water'] = 0

    # Calculate recovery factor if OOIP available
    if 'OOIP' in data:
        results['OOIP'] = data['OOIP']
        results['recovery_factor'] = results['cumulative_oil'] / results['OOIP'] * 100
    else:
        results['recovery_factor'] = 0

    # Final metrics
    results['final_pressure_mean'] = np.mean(results['pressure'][-1, :])
    results['final_saturation_mean'] = np.mean(results['saturation'][-1, :])

    # Water cut
    total_fluid = results['cumulative_oil'] + results['cumulative_water']
    if total_fluid > 0:
        results['water_cut'] = results['cumulative_water'] / total_fluid * 100
    else:
        results['water_cut'] = 0

    return results


def load_all_results(n_samples):
    """Load results from all simulations."""

    all_results = []

    for i in range(n_samples):
        # Load parameters
        param_file = os.path.join(RESULTS_DIR, f'params_sample_{i:03d}.json')
        if not os.path.exists(param_file):
            continue

        with open(param_file, 'r') as f:
            params = json.load(f)

        # Load simulation results
        run_dir = os.path.join(RESULTS_DIR, f'run_{i:03d}')
        results = extract_simulation_results(run_dir)

        if results is not None:
            # Combine parameters and results
            combined = {**params, **results}
            # Don't include large arrays in summary
            combined_summary = {k: v for k, v in combined.items()
                              if not isinstance(v, np.ndarray)}
            all_results.append(combined_summary)

    return pd.DataFrame(all_results)


# =============================================================================
# EVOLUTION PLOTS (Pressure & Saturation)
# =============================================================================

def plot_evolution(sample_id):
    """
    Create pressure and saturation evolution plots for one simulation.

    Parameters:
    -----------
    sample_id : int
        Sample ID
    """

    print(f"Creating evolution plots for sample {sample_id:03d}...")

    run_dir = os.path.join(RESULTS_DIR, f'run_{sample_id:03d}')
    results = extract_simulation_results(run_dir)

    if results is None:
        print(f"  Skipping sample {sample_id:03d} (no results)")
        return

    # Get data
    P = results['pressure']  # (timesteps, cells)
    Sw = results['saturation']
    time = results['time']

    n_timesteps = P.shape[0]
    n_cells = P.shape[1]
    grid_size = int(np.sqrt(n_cells))  # Assuming square grid

    # Reshape to 2D grid
    P_grid = P.reshape(n_timesteps, grid_size, grid_size)
    Sw_grid = Sw.reshape(n_timesteps, grid_size, grid_size)

    # Select timesteps to plot
    plot_indices = np.linspace(0, n_timesteps-1, N_TIMESTEPS, dtype=int)

    # Create figure
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(f'Sample {sample_id:03d} - Pressure and Saturation Evolution',
                 fontsize=16, fontweight='bold')

    # Pressure plots
    for idx, t_idx in enumerate(plot_indices):
        ax = plt.subplot(2, N_TIMESTEPS, idx + 1)
        im = ax.imshow(P_grid[t_idx], cmap='jet', aspect='auto',
                      vmin=P_grid.min(), vmax=P_grid.max())
        ax.set_title(f't = {time[t_idx]:.1f} days', fontsize=10)
        ax.axis('off')
        if idx == 0:
            ax.set_ylabel('Pressure (psi)', fontsize=12, fontweight='bold')

    # Add colorbar for pressure
    cbar_ax = fig.add_axes([0.92, 0.55, 0.01, 0.35])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Pressure (psi)', rotation=270, labelpad=20)

    # Saturation plots
    for idx, t_idx in enumerate(plot_indices):
        ax = plt.subplot(2, N_TIMESTEPS, N_TIMESTEPS + idx + 1)
        im = ax.imshow(Sw_grid[t_idx], cmap='Blues', aspect='auto',
                      vmin=0, vmax=1)
        ax.set_title(f't = {time[t_idx]:.1f} days', fontsize=10)
        ax.axis('off')
        if idx == 0:
            ax.set_ylabel('Water Saturation', fontsize=12, fontweight='bold')

    # Add colorbar for saturation
    cbar_ax = fig.add_axes([0.92, 0.1, 0.01, 0.35])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Water Saturation', rotation=270, labelpad=20)

    plt.tight_layout(rect=[0, 0, 0.9, 0.96])

    # Save figure
    output_file = os.path.join(PLOTS_DIR, f'evolution_sample_{sample_id:03d}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_file}")


# =============================================================================
# CUMULATIVE PRODUCTION PLOTS (Interactive)
# =============================================================================

def plot_well_production_interactive(sample_id):
    """
    Create interactive cumulative production plot for all wells.

    Parameters:
    -----------
    sample_id : int
        Sample ID
    """

    print(f"Creating interactive production plots for sample {sample_id:03d}...")

    run_dir = os.path.join(RESULTS_DIR, f'run_{sample_id:03d}')
    results = extract_simulation_results(run_dir)

    if results is None or 'well_rates' not in results:
        print(f"  Skipping sample {sample_id:03d} (no well data)")
        return

    time = results['time']
    rates = results['well_rates']  # (timesteps, n_wells)
    n_wells = rates.shape[1]

    # Calculate cumulative production (assuming rates are in STB/day)
    dt = np.diff(time, prepend=0)
    cumulative = np.cumsum(rates * dt[:, np.newaxis], axis=0)

    # Separate producers and injectors (first 40 are producers, last 18 are injectors)
    n_producers = 40
    n_injectors = 18

    # Create interactive plot
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Producers', 'Injectors'),
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}]]
    )

    # Plot producers
    for i in range(n_producers):
        fig.add_trace(
            go.Scatter(
                x=time,
                y=cumulative[:, i],
                mode='lines',
                name=f'PROD_{i:03d}',
                legendgroup='producers',
                hovertemplate=f'PROD_{i:03d}<br>Time: %{{x:.1f}} days<br>Cumulative: %{{y:.0f}} STB'
            ),
            row=1, col=1
        )

    # Plot injectors
    for i in range(n_injectors):
        fig.add_trace(
            go.Scatter(
                x=time,
                y=cumulative[:, n_producers + i],
                mode='lines',
                name=f'INJ_{i:03d}',
                legendgroup='injectors',
                hovertemplate=f'INJ_{i:03d}<br>Time: %{{x:.1f}} days<br>Cumulative: %{{y:.0f}} STB'
            ),
            row=1, col=2
        )

    # Update layout
    fig.update_xaxes(title_text="Time (days)", row=1, col=1)
    fig.update_xaxes(title_text="Time (days)", row=1, col=2)
    fig.update_yaxes(title_text="Cumulative Production (STB)", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Injection (STB)", row=1, col=2)

    fig.update_layout(
        title=f'Sample {sample_id:03d} - Cumulative Well Performance',
        height=600,
        showlegend=True,
        hovermode='closest'
    )

    # Save as HTML
    output_file = os.path.join(PLOTS_DIR, f'well_production_sample_{sample_id:03d}.html')
    fig.write_html(output_file)

    print(f"  ✓ Saved: {output_file}")


# =============================================================================
# OVERALL COMPARISON PLOT
# =============================================================================

def plot_overall_comparison(df_results):
    """
    Create comparison plot showing all simulations.

    Parameters:
    -----------
    df_results : DataFrame
        Results from all simulations
    """

    print("\nCreating overall comparison plots...")

    # Sort by cumulative oil production
    df_sorted = df_results.sort_values('cumulative_oil', ascending=False)

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig)

    # 1. Cumulative oil production comparison
    ax1 = fig.add_subplot(gs[0, :])
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(df_sorted)))
    bars = ax1.bar(range(len(df_sorted)), df_sorted['cumulative_oil'],
                   color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Sample ID (sorted by recovery)', fontsize=12)
    ax1.set_ylabel('Cumulative Oil Production (STB)', fontsize=12)
    ax1.set_title('Comparison of All Simulations - Cumulative Oil Recovery',
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Annotate best and worst
    best_idx = df_sorted.index[0]
    worst_idx = df_sorted.index[-1]
    ax1.text(0, df_sorted['cumulative_oil'].iloc[0] * 1.05,
            f'Best: Sample {best_idx}', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))

    # 2. Recovery factor vs Total injection
    ax2 = fig.add_subplot(gs[1, 0])
    scatter = ax2.scatter(df_results['total_injection'],
                         df_results['recovery_factor'],
                         c=df_results['producer_bhp'],
                         s=100, alpha=0.6, cmap='viridis',
                         edgecolors='black', linewidth=0.5)
    ax2.set_xlabel('Total Injection Rate (STB/day)', fontsize=11)
    ax2.set_ylabel('Recovery Factor (%)', fontsize=11)
    ax2.set_title('Recovery vs. Injection Rate', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Producer BHP (psi)', rotation=270, labelpad=20)

    # 3. Recovery factor vs Producer BHP
    ax3 = fig.add_subplot(gs[1, 1])
    scatter = ax3.scatter(df_results['producer_bhp'],
                         df_results['recovery_factor'],
                         c=df_results['total_injection'],
                         s=100, alpha=0.6, cmap='plasma',
                         edgecolors='black', linewidth=0.5)
    ax3.set_xlabel('Producer BHP (psi)', fontsize=11)
    ax3.set_ylabel('Recovery Factor (%)', fontsize=11)
    ax3.set_title('Recovery vs. Producer BHP', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Total Injection (STB/day)', rotation=270, labelpad=20)

    # 4. Water cut vs Recovery factor
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.scatter(df_results['recovery_factor'], df_results['water_cut'],
               c=df_results['total_injection'], s=100, alpha=0.6,
               cmap='coolwarm', edgecolors='black', linewidth=0.5)
    ax4.set_xlabel('Recovery Factor (%)', fontsize=11)
    ax4.set_ylabel('Water Cut (%)', fontsize=11)
    ax4.set_title('Water Cut vs. Recovery Factor', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # 5. Top 5 strategies table
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')

    top5 = df_sorted.head(5)[['sample_id', 'total_injection', 'producer_bhp',
                              'cumulative_oil', 'recovery_factor']]
    top5_display = top5.copy()
    top5_display.columns = ['Sample', 'Inj (STB/d)', 'BHP (psi)',
                           'Cum Oil (STB)', 'RF (%)']

    table = ax5.table(cellText=top5_display.values,
                     colLabels=top5_display.columns,
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(len(top5_display.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style best row
    for i in range(len(top5_display.columns)):
        table[(1, i)].set_facecolor('#90EE90')

    ax5.set_title('Top 5 Injection Strategies', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()

    # Save figure
    output_file = os.path.join(PLOTS_DIR, 'overall_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")

    plt.close()


# =============================================================================
# SENSITIVITY ANALYSIS
# =============================================================================

def plot_sensitivity_analysis(df_results):
    """
    Create sensitivity analysis (tornado chart).

    Parameters:
    -----------
    df_results : DataFrame
        Results from all simulations
    """

    print("Creating sensitivity analysis...")

    # Calculate Spearman correlation with recovery factor
    parameters = ['base_injection_rate', 'perturbation_std', 'producer_bhp', 'total_injection']
    correlations = {}

    for param in parameters:
        if param in df_results.columns:
            corr, pval = spearmanr(df_results[param], df_results['recovery_factor'])
            correlations[param] = {'correlation': corr, 'p_value': pval}

    # Sort by absolute correlation
    sorted_params = sorted(correlations.items(), key=lambda x: abs(x[1]['correlation']), reverse=True)

    # Create tornado chart
    fig, ax = plt.subplots(figsize=(10, 6))

    params = [p[0] for p in sorted_params]
    corrs = [p[1]['correlation'] for p in sorted_params]
    colors = ['green' if c > 0 else 'red' for c in corrs]

    y_pos = np.arange(len(params))
    ax.barh(y_pos, corrs, color=colors, edgecolor='black', linewidth=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([p.replace('_', ' ').title() for p in params])
    ax.set_xlabel('Spearman Correlation with Recovery Factor', fontsize=12)
    ax.set_title('Sensitivity Analysis - Impact on Oil Recovery', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.grid(True, alpha=0.3, axis='x')

    # Add correlation values
    for i, (param, corr) in enumerate(zip(params, corrs)):
        ax.text(corr, i, f'  {corr:.3f}', va='center',
               ha='left' if corr > 0 else 'right', fontsize=10)

    plt.tight_layout()

    # Save
    output_file = os.path.join(PLOTS_DIR, 'sensitivity_analysis.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")

    plt.close()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main post-processing function."""

    print("="*70)
    print("POST-PROCESSING LHS RESULTS")
    print("="*70)

    # Load all results
    print("\nLoading results from all simulations...")
    df_results = load_all_results(n_samples=100)  # Adjust as needed

    if len(df_results) == 0:
        print("Error: No results found!")
        return

    print(f"✓ Loaded {len(df_results)} successful simulations")

    # Save summary
    summary_file = os.path.join(RESULTS_DIR, 'results_summary.csv')
    df_results.to_csv(summary_file, index=False)
    print(f"✓ Summary saved to: {summary_file}")

    # Create evolution plots for each simulation
    print("\n" + "-"*70)
    print("Creating evolution plots...")
    print("-"*70)

    for sample_id in df_results['sample_id'].astype(int):
        plot_evolution(sample_id)

    # Create interactive well production plots
    print("\n" + "-"*70)
    print("Creating interactive well production plots...")
    print("-"*70)

    for sample_id in df_results['sample_id'].astype(int):
        plot_well_production_interactive(sample_id)

    # Create overall comparison
    print("\n" + "-"*70)
    print("Creating overall comparison plots...")
    print("-"*70)

    plot_overall_comparison(df_results)

    # Sensitivity analysis
    print("\n" + "-"*70)
    print("Creating sensitivity analysis...")
    print("-"*70)

    plot_sensitivity_analysis(df_results)

    # Print best result
    best_sample = df_results.loc[df_results['cumulative_oil'].idxmax()]

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nBest performing strategy:")
    print(f"  Sample ID: {int(best_sample['sample_id'])}")
    print(f"  Total injection: {best_sample['total_injection']:.1f} STB/day")
    print(f"  Producer BHP: {best_sample['producer_bhp']:.1f} psi")
    print(f"  Cumulative oil: {best_sample['cumulative_oil']:.0f} STB")
    print(f"  Recovery factor: {best_sample['recovery_factor']:.2f}%")
    print(f"  Water cut: {best_sample['water_cut']:.2f}%")

    print(f"\nAll plots saved in: {PLOTS_DIR}/")
    print("="*70)


if __name__ == '__main__':
    main()
