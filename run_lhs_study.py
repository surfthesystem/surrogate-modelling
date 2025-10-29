"""
Latin Hypercube Sampling for Injection Scenario Study

This script generates a design of experiments using LHS to test
different injection scenarios with the IMPES reservoir simulator.
"""

import numpy as np
from scipy.stats import qmc
import json
import os
import subprocess
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

# Define parameter ranges for LHS
PARAMETERS = {
    # Well parameters (most important)
    'injector1_rate': [100, 600],      # bbl/day (water injection)
    'injector2_rate': [100, 600],      # bbl/day (water injection)
    'producer_bhp': [250, 400],        # psi (bottom-hole pressure)

    # Optional: Reservoir properties
    # 'initial_pressure': [400, 600],  # psi
    # 'permeability_multiplier': [0.5, 2.0],  # Multiply existing perm field

    # Simulation parameters
    'simulation_time': [365, 1825],    # days (1 to 5 years)
}

# Number of LHS samples
N_SAMPLES = 20  # Increase for more comprehensive study

# Output directory
OUTPUT_DIR = 'lhs_results'

# ============================================================================
# LATIN HYPERCUBE SAMPLING
# ============================================================================

def generate_lhs_samples(parameters, n_samples, seed=42):
    """
    Generate Latin Hypercube samples for given parameters.

    Parameters:
    -----------
    parameters : dict
        Dictionary with parameter names as keys and [min, max] ranges as values
    n_samples : int
        Number of samples to generate
    seed : int
        Random seed for reproducibility

    Returns:
    --------
    samples : np.ndarray
        Array of shape (n_samples, n_parameters) with sampled values
    param_names : list
        List of parameter names in the same order as columns in samples
    """
    param_names = list(parameters.keys())
    n_params = len(param_names)

    # Create Latin Hypercube sampler
    sampler = qmc.LatinHypercube(d=n_params, seed=seed)

    # Generate samples in [0, 1]^d
    samples_unit = sampler.random(n=n_samples)

    # Scale to parameter ranges
    samples = np.zeros_like(samples_unit)
    for i, param in enumerate(param_names):
        min_val, max_val = parameters[param]
        samples[:, i] = min_val + samples_unit[:, i] * (max_val - min_val)

    return samples, param_names


def save_lhs_design(samples, param_names, filename='lhs_design.csv'):
    """Save LHS design to CSV file."""
    import pandas as pd

    df = pd.DataFrame(samples, columns=param_names)
    df.index.name = 'sample_id'
    df.to_csv(filename)

    print(f"LHS design saved to {filename}")
    print(f"\nDesign summary:")
    print(df.describe())

    return df


# ============================================================================
# SIMULATOR CONFIGURATION
# ============================================================================

def create_input_file(sample_params, sample_id, output_dir):
    """
    Create a modified input file for a specific LHS sample.

    This function modifies the simulator input file with the
    parameter values from the LHS sample.

    Parameters:
    -----------
    sample_params : dict
        Dictionary with parameter names and values for this sample
    sample_id : int
        ID number for this sample
    output_dir : str
        Directory to save results
    """

    # Read template input file
    template_file = 'simulator/input_file_phase1.py'

    with open(template_file, 'r') as f:
        template_content = f.read()

    # Create modified input file
    modified_content = template_content

    # Modify well constraints
    # IMPORTANT: This is a simplified example. You'll need to adapt
    # based on your actual input file structure.

    # Example: Replace injection rates
    if 'injector1_rate' in sample_params:
        # Convert bbl/day to scf/day (multiply by 5.61 for water)
        inj1_scf = sample_params['injector1_rate'] * 5.61
        # Find and replace in the well.constraint array
        # This is highly specific to your input file format
        modified_content = modified_content.replace(
            'well.constraint = [[',
            f'well.constraint = [[{inj1_scf}],'
        )

    # Save modified input file
    output_file = os.path.join(output_dir, f'input_sample_{sample_id}.py')
    with open(output_file, 'w') as f:
        f.write(modified_content)

    # Also save parameters as JSON for easy reference
    param_file = os.path.join(output_dir, f'params_sample_{sample_id}.json')
    with open(param_file, 'w') as f:
        json.dump(sample_params, f, indent=2)

    return output_file


def run_simulation(input_file, sample_id, output_dir):
    """
    Run the simulator with the given input file.

    Parameters:
    -----------
    input_file : str
        Path to the input file
    sample_id : int
        Sample ID number
    output_dir : str
        Directory for results
    """

    print(f"\nRunning simulation {sample_id}...")

    # Create subdirectory for this run
    run_dir = os.path.join(output_dir, f'run_{sample_id}')
    os.makedirs(run_dir, exist_ok=True)

    # Run the simulator
    # NOTE: You'll need to adapt this based on how your simulator is run
    cmd = [
        'python',
        'simulator/IMPES_phase1.py',
        '--input', input_file,
        '--output-dir', run_dir
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        if result.returncode == 0:
            print(f"  ✓ Simulation {sample_id} completed successfully")
        else:
            print(f"  ✗ Simulation {sample_id} failed:")
            print(f"    {result.stderr}")

    except subprocess.TimeoutExpired:
        print(f"  ✗ Simulation {sample_id} timed out")
    except Exception as e:
        print(f"  ✗ Error running simulation {sample_id}: {e}")


# ============================================================================
# POST-PROCESSING
# ============================================================================

def extract_results(run_dir):
    """
    Extract key results from a simulation run.

    Returns:
    --------
    results : dict
        Dictionary with key performance indicators
    """

    results = {}

    # Load simulation results (adapt to your output format)
    result_file = os.path.join(run_dir, 'Phase1_n10000_t10_days.npz')

    if os.path.exists(result_file):
        data = np.load(result_file, allow_pickle=True)

        # Extract key metrics
        results['final_pressure_mean'] = np.mean(data['P'][-1, :])
        results['final_saturation_mean'] = np.mean(data['Sw'][-1, :])
        results['cumulative_water_injected'] = data.get('cum_water_inj', 0)
        results['cumulative_oil_produced'] = data.get('cum_oil_prod', 0)
        results['cumulative_water_produced'] = data.get('cum_water_prod', 0)

        # Calculate recovery factor
        if 'OOIP' in data:
            results['recovery_factor'] = results['cumulative_oil_produced'] / data['OOIP']

        # Water cut
        total_fluid = results['cumulative_oil_produced'] + results['cumulative_water_produced']
        if total_fluid > 0:
            results['water_cut'] = results['cumulative_water_produced'] / total_fluid

    return results


def summarize_all_results(output_dir, n_samples):
    """
    Summarize results from all LHS samples.

    Creates a CSV file with all parameters and results for analysis.
    """
    import pandas as pd

    all_results = []

    for i in range(n_samples):
        # Load parameters
        param_file = os.path.join(output_dir, f'params_sample_{i}.json')
        if os.path.exists(param_file):
            with open(param_file, 'r') as f:
                params = json.load(f)
        else:
            params = {}

        # Load results
        run_dir = os.path.join(output_dir, f'run_{i}')
        results = extract_results(run_dir)

        # Combine
        combined = {**params, **results, 'sample_id': i}
        all_results.append(combined)

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Save to CSV
    summary_file = os.path.join(output_dir, 'lhs_study_summary.csv')
    df.to_csv(summary_file, index=False)

    print(f"\n" + "="*60)
    print(f"LHS STUDY COMPLETE")
    print(f"="*60)
    print(f"Summary saved to: {summary_file}")
    print(f"\nResults summary:")
    print(df.describe())

    return df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""

    print("="*60)
    print("LATIN HYPERCUBE SAMPLING - INJECTION SCENARIO STUDY")
    print("="*60)
    print(f"\nParameters to vary: {list(PARAMETERS.keys())}")
    print(f"Number of samples: {N_SAMPLES}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate LHS samples
    print("\n" + "-"*60)
    print("Step 1: Generating LHS samples...")
    print("-"*60)

    samples, param_names = generate_lhs_samples(PARAMETERS, N_SAMPLES)

    # Save design
    design_file = os.path.join(OUTPUT_DIR, 'lhs_design.csv')
    df_design = save_lhs_design(samples, param_names, design_file)

    # Run simulations
    print("\n" + "-"*60)
    print("Step 2: Running simulations...")
    print("-"*60)

    for i in range(N_SAMPLES):
        # Get parameters for this sample
        sample_params = {name: samples[i, j] for j, name in enumerate(param_names)}

        print(f"\nSample {i}/{N_SAMPLES}:")
        for name, value in sample_params.items():
            print(f"  {name}: {value:.2f}")

        # Create input file (you'll need to implement this based on your simulator)
        # input_file = create_input_file(sample_params, i, OUTPUT_DIR)

        # Run simulation (you'll need to adapt this)
        # run_simulation(input_file, i, OUTPUT_DIR)

    print("\n⚠️  NOTE: Input file creation and simulation execution are")
    print("    commented out. You need to adapt these functions to your")
    print("    specific simulator input file format.")

    # Post-process results
    print("\n" + "-"*60)
    print("Step 3: Post-processing results...")
    print("-"*60)

    # df_results = summarize_all_results(OUTPUT_DIR, N_SAMPLES)

    print("\n✓ LHS study setup complete!")
    print(f"\nNext steps:")
    print(f"1. Adapt create_input_file() to modify your simulator input")
    print(f"2. Adapt run_simulation() to run your simulator")
    print(f"3. Adapt extract_results() to parse your output files")
    print(f"4. Run this script: python run_lhs_study.py")
    print(f"5. Analyze results in {OUTPUT_DIR}/lhs_study_summary.csv")


if __name__ == '__main__':
    main()
