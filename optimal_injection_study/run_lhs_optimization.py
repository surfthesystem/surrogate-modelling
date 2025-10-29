"""
Optimal Water Injection Strategy Study using Latin Hypercube Sampling

This script runs an LHS study to find the optimal injection strategy
that maximizes oil recovery over 90 days with a total injection constraint.

Author: Generated for Reservoir Surrogate Modeling Project
Date: 2025-10-29
"""

import numpy as np
import pandas as pd
from scipy.stats import qmc
import os
import sys
import json
import time
from datetime import datetime
import subprocess
import shutil

# Add simulator path
sys.path.append('../simulator')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Study parameters
N_SAMPLES = 30  # Number of LHS samples (increase for more thorough study)
MAX_TOTAL_INJECTION = 27000  # STB/day - total injection constraint
SIMULATION_DAYS = 90  # 3 months
OUTPUT_DIR = 'lhs_results'
RANDOM_SEED = 42

# Injector and producer counts
N_INJECTORS = 18
N_PRODUCERS = 40
N_WELLS_TOTAL = 58

# Individual injector bounds
MIN_INJ_RATE = 200   # STB/day minimum per injector
MAX_INJ_RATE = 3000  # STB/day maximum per injector

# Producer BHP range
MIN_PROD_BHP = 900   # psi
MAX_PROD_BHP = 1100  # psi

# LHS Parameter definition (Option 3: Simple approach)
PARAMETERS = {
    'base_injection_rate': [1000, 2000],  # Base rate for all injectors (STB/day)
    'perturbation_std': [0.1, 0.4],       # Relative std dev for perturbations
    'producer_bhp': [900, 1100],          # Producer BHP (psi)
}

# =============================================================================
# LATIN HYPERCUBE SAMPLING
# =============================================================================

def generate_lhs_samples(parameters, n_samples, seed=RANDOM_SEED):
    """
    Generate Latin Hypercube samples.

    Parameters:
    -----------
    parameters : dict
        Parameter ranges {name: [min, max]}
    n_samples : int
        Number of samples
    seed : int
        Random seed

    Returns:
    --------
    samples : np.ndarray
        Sample array (n_samples × n_parameters)
    param_names : list
        Parameter names
    """
    param_names = list(parameters.keys())
    n_params = len(param_names)

    print(f"\nGenerating {n_samples} LHS samples for {n_params} parameters...")

    # Create Latin Hypercube sampler
    sampler = qmc.LatinHypercube(d=n_params, seed=seed)

    # Generate samples in [0, 1]^d
    samples_unit = sampler.random(n=n_samples)

    # Scale to parameter ranges
    samples = np.zeros_like(samples_unit)
    for i, param in enumerate(param_names):
        min_val, max_val = parameters[param]
        samples[:, i] = min_val + samples_unit[:, i] * (max_val - min_val)

    print(f"✓ LHS samples generated")

    return samples, param_names


def create_injection_rates(base_rate, perturbation_std, n_injectors, seed):
    """
    Create injection rates for all injectors with perturbations.

    Parameters:
    -----------
    base_rate : float
        Base injection rate (STB/day)
    perturbation_std : float
        Relative standard deviation for perturbations
    n_injectors : int
        Number of injectors
    seed : int
        Random seed for perturbations

    Returns:
    --------
    rates : np.ndarray
        Injection rates for all injectors (STB/day)
    """
    np.random.seed(seed)

    # Generate perturbations
    perturbations = np.random.normal(1.0, perturbation_std, n_injectors)

    # Apply perturbations to base rate
    rates = base_rate * perturbations

    # Clip to individual bounds
    rates = np.clip(rates, MIN_INJ_RATE, MAX_INJ_RATE)

    # Normalize to satisfy total injection constraint
    current_total = np.sum(rates)
    if current_total > MAX_TOTAL_INJECTION:
        # Scale down proportionally
        rates = rates * (MAX_TOTAL_INJECTION / current_total)

    return rates


# =============================================================================
# INPUT FILE GENERATION
# =============================================================================

def read_well_locations():
    """Read all 58 well locations from Phase 1 data."""

    wells_file = '../data/well_locations.csv'
    df = pd.read_csv(wells_file)

    # Convert to feet
    df['x_ft'] = df['x_m'] * 3.28084
    df['y_ft'] = df['y_m'] * 3.28084

    # Separate producers and injectors
    producers = df[df['well_type'] == 'producer'].reset_index(drop=True)
    injectors = df[df['well_type'] == 'injector'].reset_index(drop=True)

    return producers, injectors


def create_input_file(sample_id, inj_rates, prod_bhp, output_dir):
    """
    Create a customized input file for this LHS sample.

    Parameters:
    -----------
    sample_id : int
        Sample ID
    inj_rates : array
        Injection rates for 18 injectors (STB/day)
    prod_bhp : float
        Producer BHP constraint (psi)
    output_dir : str
        Output directory

    Returns:
    --------
    input_file : str
        Path to created input file
    """

    # Read template
    template_file = '../simulator/input_file_phase1.py'
    with open(template_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Read well locations
    producers, injectors = read_well_locations()

    # Create modified input file
    output_file = os.path.join(output_dir, f'input_sample_{sample_id:03d}.py')

    with open(output_file, 'w', encoding='utf-8') as f:
        # Write header
        f.write(f'"""\n')
        f.write(f'LHS Sample {sample_id:03d} - Optimal Injection Study\n')
        f.write(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'Total injection: {np.sum(inj_rates):.1f} STB/day\n')
        f.write(f'Producer BHP: {prod_bhp:.1f} psi\n')
        f.write(f'"""\n\n')

        # Copy imports and class definitions (lines 1-49)
        for i in range(49):
            f.write(lines[i])

        # Start inputfile function
        f.write('\ndef inputfile(fluid,reservoir,petro,numerical,BC,IC,well):\n')

        # Write numerical parameters (modified for 90 days)
        f.write(f'    numerical.dt     = 1.0  # time step (days)\n')
        f.write(f'    numerical.tfinal = {SIMULATION_DAYS}  # 90 days (3 months)\n')
        f.write(f'    numerical.PV_final = 1\n')
        f.write(f'    numerical.Nx     = 100\n')
        f.write(f'    numerical.Ny     = 100\n')
        f.write(f'    numerical.N  = numerical.Nx * numerical.Ny\n')
        f.write(f'    numerical.method = "IMPES"\n')
        f.write(f'    numerical.tswitch= 500\n\n')

        # Copy fluid and petro parameters (lines 61-86)
        for i in range(61, 86):
            f.write('    ' + lines[i].strip() + '\n')

        # Write reservoir parameters (with all 58 wells)
        f.write('\n    # Reading Phase 1 generated files\n')
        f.write('    base_path = "../../data/impes_input/"\n')
        f.write('    depth    =-np.loadtxt(base_path + "depth.txt")\n')
        f.write('    porosity = np.loadtxt(base_path + "porosity.txt")\n')
        f.write('    permx    = np.loadtxt(base_path + "permeability.txt")\n\n')

        # Reservoir properties (lines 97-111)
        for i in range(96, 111):
            f.write('    ' + lines[i].strip() + '\n')

        # Write well configuration for ALL 58 wells
        f.write('\n    # Well configuration - ALL 58 WELLS\n')
        f.write(f'    # {N_PRODUCERS} producers (BHP control) + {N_INJECTORS} injectors (rate control)\n\n')

        # Well x-coordinates
        f.write('    well.x = [\n')
        for idx, row in producers.iterrows():
            f.write(f'        [{row["x_ft"]:.2f}],  # PROD_{idx:03d}\n')
        for idx, row in injectors.iterrows():
            f.write(f'        [{row["x_ft"]:.2f}],  # INJ_{idx:03d}\n')
        f.write('    ]\n\n')

        # Well y-coordinates
        f.write('    well.y = [\n')
        for idx, row in producers.iterrows():
            f.write(f'        [{row["y_ft"]:.2f}],  # PROD_{idx:03d}\n')
        for idx, row in injectors.iterrows():
            f.write(f'        [{row["y_ft"]:.2f}],  # INJ_{idx:03d}\n')
        f.write('    ]\n\n')

        # Well types
        f.write('    well.type = [\n')
        f.write(f'        {[[2]]*N_PRODUCERS},  # Producers (Type 2 = BHP)\n')
        f.write(f'        {[[1]]*N_INJECTORS}   # Injectors (Type 1 = Rate)\n')
        f.write('    ]\n')
        # Flatten the list
        f.write('    well.type = [item for sublist in well.type for item in sublist]\n\n')

        # Well constraints
        f.write('    well.constraint = [\n')
        for i in range(N_PRODUCERS):
            f.write(f'        [{prod_bhp:.1f}],  # PROD_{i:03d} BHP (psi)\n')
        for i in range(N_INJECTORS):
            f.write(f'        [{inj_rates[i]:.2f}],  # INJ_{i:03d} rate (STB/day)\n')
        f.write('    ]\n\n')

        # Other well properties
        f.write(f'    well.rw = [[0.25]]*{N_WELLS_TOTAL}\n')
        f.write(f'    well.skin = [[0]]*{N_WELLS_TOTAL}\n')
        f.write(f'    well.direction = [["v"]]*{N_WELLS_TOTAL}\n\n')

        # Copy rest of input file (grid setup, BC, IC)
        # Lines 181-261 contain the grid discretization, BC, and IC setup
        for i in range(181, 262):
            line = lines[i]
            # Preserve indentation by checking original indentation
            if line.strip():  # Non-empty line
                # Count original indentation
                orig_indent = len(line) - len(line.lstrip())
                # We're inside inputfile function (4 spaces base)
                # Original function also has 4 spaces base
                # So we keep the same relative indentation
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

    print(f"  ✓ Created input file: {output_file}")
    return output_file


# =============================================================================
# SIMULATION EXECUTION
# =============================================================================

def run_simulation(sample_id, input_file, output_dir):
    """
    Run the IMPES simulator for this sample.

    Parameters:
    -----------
    sample_id : int
        Sample ID
    input_file : str
        Path to input file
    output_dir : str
        Output directory for results

    Returns:
    --------
    success : bool
        True if simulation completed successfully
    """

    print(f"\n{'='*70}")
    print(f"Running Simulation {sample_id:03d}/{N_SAMPLES}")
    print(f"{'='*70}")

    # Create run directory
    run_dir = os.path.join(output_dir, f'run_{sample_id:03d}')
    os.makedirs(run_dir, exist_ok=True)

    # Copy input file to run directory
    shutil.copy(input_file, os.path.join(run_dir, 'input_file_phase1.py'))

    # Copy simulator files to run directory
    simulator_files = [
        'IMPES_phase1.py', 'IMPES.py', 'myarrays.py', 'updatewells.py',
        'cap_press.py', 'rel_perm.py', 'fluid_properties.py',
        'petrophysics.py', 'prodindex.py', 'Thalf.py', 'spdiaginv.py',
        'mobilityfun.py', 'postprocess.py', 'init_plot.py',
        'assignment1_reservior_init.py', 'petroplots.py'
    ]

    for sim_file in simulator_files:
        src = os.path.join('../simulator', sim_file)
        if os.path.exists(src):
            shutil.copy(src, run_dir)

    # Run simulation
    start_time = time.time()

    try:
        # Change to run directory
        original_dir = os.getcwd()
        os.chdir(run_dir)

        # Run the simulator
        result = subprocess.run(
            [sys.executable, 'IMPES_phase1.py'],
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout
        )

        # Change back to original directory
        os.chdir(original_dir)

        elapsed_time = time.time() - start_time

        if result.returncode == 0:
            print(f"✓ Simulation {sample_id:03d} completed successfully")
            print(f"  Time: {elapsed_time/60:.1f} minutes")
            return True
        else:
            print(f"✗ Simulation {sample_id:03d} failed:")
            print(f"  Error: {result.stderr[:500]}")
            return False

    except subprocess.TimeoutExpired:
        os.chdir(original_dir)
        print(f"✗ Simulation {sample_id:03d} timed out")
        return False
    except Exception as e:
        os.chdir(original_dir)
        print(f"✗ Error running simulation {sample_id:03d}: {e}")
        return False


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""

    print("="*70)
    print("OPTIMAL WATER INJECTION STRATEGY - LHS STUDY")
    print("="*70)
    print(f"\nObjective: Maximize oil recovery over {SIMULATION_DAYS} days")
    print(f"Constraint: Total injection ≤ {MAX_TOTAL_INJECTION} STB/day")
    print(f"Wells: {N_WELLS_TOTAL} ({N_PRODUCERS} producers + {N_INJECTORS} injectors)")
    print(f"LHS Samples: {N_SAMPLES}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate LHS samples
    print("\n" + "-"*70)
    print("STEP 1: Generating LHS Samples")
    print("-"*70)

    samples, param_names = generate_lhs_samples(PARAMETERS, N_SAMPLES)

    # Save LHS design
    df_design = pd.DataFrame(samples, columns=param_names)
    df_design.index.name = 'sample_id'
    design_file = os.path.join(OUTPUT_DIR, 'lhs_design.csv')
    df_design.to_csv(design_file)
    print(f"\n✓ LHS design saved to: {design_file}")
    print(f"\nDesign summary:")
    print(df_design.describe())

    # Generate injection rates and parameters for each sample
    print("\n" + "-"*70)
    print("STEP 2: Generating Injection Strategies")
    print("-"*70)

    all_parameters = []

    for i in range(N_SAMPLES):
        base_rate = samples[i, 0]
        pert_std = samples[i, 1]
        prod_bhp = samples[i, 2]

        # Create injection rates
        inj_rates = create_injection_rates(base_rate, pert_std, N_INJECTORS, seed=RANDOM_SEED + i)

        # Store parameters
        params = {
            'sample_id': i,
            'base_injection_rate': base_rate,
            'perturbation_std': pert_std,
            'producer_bhp': prod_bhp,
            'total_injection': np.sum(inj_rates),
        }

        # Add individual injector rates
        for j in range(N_INJECTORS):
            params[f'inj_{j:03d}_rate'] = inj_rates[j]

        all_parameters.append(params)

        # Save to JSON
        param_file = os.path.join(OUTPUT_DIR, f'params_sample_{i:03d}.json')
        with open(param_file, 'w') as f:
            json.dump(params, f, indent=2)

        print(f"Sample {i:03d}: Total injection = {np.sum(inj_rates):.1f} STB/day, Prod BHP = {prod_bhp:.1f} psi")

    # Save all parameters
    df_params = pd.DataFrame(all_parameters)
    params_file = os.path.join(OUTPUT_DIR, 'all_parameters.csv')
    df_params.to_csv(params_file, index=False)
    print(f"\n✓ All parameters saved to: {params_file}")

    # Create input files and run simulations
    print("\n" + "-"*70)
    print("STEP 3: Creating Input Files and Running Simulations")
    print("-"*70)

    successful_runs = 0

    for i in range(N_SAMPLES):
        # Load parameters
        with open(os.path.join(OUTPUT_DIR, f'params_sample_{i:03d}.json'), 'r') as f:
            params = json.load(f)

        # Extract injection rates
        inj_rates = [params[f'inj_{j:03d}_rate'] for j in range(N_INJECTORS)]
        prod_bhp = params['producer_bhp']

        # Create input file
        input_file = create_input_file(i, inj_rates, prod_bhp, OUTPUT_DIR)

        # Run simulation
        success = run_simulation(i, input_file, OUTPUT_DIR)

        if success:
            successful_runs += 1

    # Summary
    print("\n" + "="*70)
    print("LHS STUDY COMPLETE")
    print("="*70)
    print(f"Total samples: {N_SAMPLES}")
    print(f"Successful runs: {successful_runs}")
    print(f"Failed runs: {N_SAMPLES - successful_runs}")
    print(f"\nResults saved in: {OUTPUT_DIR}/")
    print(f"\nNext steps:")
    print(f"1. Run post-processing: python plot_results.py")
    print(f"2. Analyze optimal strategy from results")
    print("="*70)


if __name__ == '__main__':
    main()
