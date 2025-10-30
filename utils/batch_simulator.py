#!/usr/bin/env python3
"""
Batch simulation runner with progress tracking and parallel execution.
"""
import argparse
import json
import multiprocessing as mp
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


def run_single_scenario(args_tuple):
    """Run a single scenario and return status."""
    scenario_path, scenario_idx, total_scenarios, tfinal, dt, realloc_days, repo_root = args_tuple

    scenario_name = scenario_path.stem
    start_time = time.time()

    # Load scenario
    try:
        with open(scenario_path, 'r') as f:
            scn = json.load(f)
    except Exception as e:
        return {
            'scenario': scenario_name,
            'index': scenario_idx,
            'status': 'load_failed',
            'error': str(e),
            'duration': 0
        }

    # Write schedules to data/impes_input/
    base_path = repo_root / 'data' / 'impes_input'
    try:
        if 'producer_bhp' in scn:
            np.savetxt(base_path / 'schedule_producer_bhp.csv',
                      np.array(scn['producer_bhp']), delimiter=',', fmt='%.3f')
        if 'injector_rates' in scn:
            np.savetxt(base_path / 'schedule_injector_rates.csv',
                      np.array(scn['injector_rates']), delimiter=',', fmt='%.3f')
    except Exception as e:
        return {
            'scenario': scenario_name,
            'index': scenario_idx,
            'status': 'schedule_write_failed',
            'error': str(e),
            'duration': time.time() - start_time
        }

    # Run IMPES simulation
    sim_dir = repo_root / 'simulator'
    cmd = [
        sys.executable,
        'IMPES_phase1.py',
        '--tfinal', str(tfinal),
        '--dt', str(dt),
        '--realloc_days', str(realloc_days),
        '--no-plots'  # Skip plotting to save time
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(sim_dir),
            capture_output=True,
            text=True,
            timeout=3600  # 60 minute timeout per simulation
        )

        if result.returncode != 0:
            return {
                'scenario': scenario_name,
                'index': scenario_idx,
                'status': 'sim_failed',
                'error': result.stderr[:500] if result.stderr else 'Unknown error',
                'duration': time.time() - start_time
            }

    except subprocess.TimeoutExpired:
        return {
            'scenario': scenario_name,
            'index': scenario_idx,
            'status': 'timeout',
            'error': 'Simulation exceeded 10 minute timeout',
            'duration': time.time() - start_time
        }
    except Exception as e:
        return {
            'scenario': scenario_name,
            'index': scenario_idx,
            'status': 'error',
            'error': str(e),
            'duration': time.time() - start_time
        }

    # Find the latest NPZ file (search recursively in subdirectories)
    results_dir = repo_root / 'results' / 'impes_sim'
    npz_files = sorted(results_dir.glob('*/Phase1_n*_t*_days.npz'),
                      key=lambda p: p.stat().st_mtime, reverse=True)

    if not npz_files:
        return {
            'scenario': scenario_name,
            'index': scenario_idx,
            'status': 'no_output',
            'error': 'No NPZ file generated',
            'duration': time.time() - start_time
        }

    # Extract KPIs
    try:
        data = np.load(npz_files[0])
        kpis = {
            'cum_oil_prod_stb': float(data['well_cum_oil_prod_stb'][:,-1].sum()),
            'cum_water_prod_stb': float(data['well_cum_water_prod_stb'][:,-1].sum()),
            'cum_water_inj_stb': float(data['well_cum_water_inj_stb'][:,-1].sum()),
            'final_days': float(data['t'][-1]),
            'timesteps': int(len(data['t']) - 1),
        }
    except Exception as e:
        return {
            'scenario': scenario_name,
            'index': scenario_idx,
            'status': 'kpi_extraction_failed',
            'error': str(e),
            'duration': time.time() - start_time
        }

    # Move NPZ to scenario-specific location
    scenario_results_dir = repo_root / 'results' / 'training_data' / scenario_name
    scenario_results_dir.mkdir(parents=True, exist_ok=True)

    target_npz = scenario_results_dir / f'{scenario_name}.npz'
    npz_files[0].rename(target_npz)

    # Save KPIs
    with open(scenario_results_dir / 'kpis.json', 'w') as f:
        json.dump(kpis, f, indent=2)

    duration = time.time() - start_time

    return {
        'scenario': scenario_name,
        'index': scenario_idx,
        'status': 'success',
        'duration': duration,
        'npz_path': str(target_npz.relative_to(repo_root)),
        **kpis
    }


def print_progress(completed, total, start_time, success_count, fail_count):
    """Print progress bar and statistics."""
    elapsed = time.time() - start_time
    pct = 100 * completed / total
    bar_len = 40
    filled = int(bar_len * completed / total)
    bar = '█' * filled + '░' * (bar_len - filled)

    rate = completed / elapsed if elapsed > 0 else 0
    eta_seconds = (total - completed) / rate if rate > 0 else 0
    eta_str = f"{int(eta_seconds//60)}m {int(eta_seconds%60)}s"

    print(f"\r[{bar}] {completed}/{total} ({pct:.1f}%) | "
          f"✓ {success_count} ✗ {fail_count} | "
          f"Rate: {rate:.2f}/s | ETA: {eta_str}",
          end='', flush=True)


def main():
    parser = argparse.ArgumentParser(description='Batch simulation runner with progress tracking')
    parser.add_argument('--scenarios', type=str, default='scenarios',
                       help='Directory containing scenario JSON files')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers')
    parser.add_argument('--tfinal', type=float, default=180,
                       help='Simulation duration in days (default: 180 = 6 months)')
    parser.add_argument('--dt', type=float, default=1.0,
                       help='Time step in days')
    parser.add_argument('--realloc_days', type=float, default=30.0,
                       help='Control reallocation period in days')
    parser.add_argument('--max_scenarios', type=int, default=None,
                       help='Maximum number of scenarios to run (for testing)')
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    scenarios_dir = repo_root / args.scenarios

    # Find all scenario files
    scenario_files = sorted(scenarios_dir.glob('*.json'))
    if args.max_scenarios:
        scenario_files = scenario_files[:args.max_scenarios]

    total_scenarios = len(scenario_files)

    if total_scenarios == 0:
        print(f"No scenario files found in {scenarios_dir}")
        return 1

    print(f"Found {total_scenarios} scenarios to run")
    print(f"Using {args.workers} parallel workers")
    print(f"Simulation parameters: tfinal={args.tfinal} days, dt={args.dt} days, realloc_days={args.realloc_days} days")
    print(f"Results will be saved to: results/training_data/")
    print()

    # Prepare arguments for parallel execution
    tasks = [
        (scen_file, i, total_scenarios, args.tfinal, args.dt, args.realloc_days, repo_root)
        for i, scen_file in enumerate(scenario_files)
    ]

    # Run simulations in parallel with progress tracking
    results = []
    success_count = 0
    fail_count = 0
    start_time = time.time()

    print("Starting batch simulation...")
    print()

    with mp.Pool(processes=args.workers) as pool:
        for i, result in enumerate(pool.imap_unordered(run_single_scenario, tasks)):
            results.append(result)

            if result['status'] == 'success':
                success_count += 1
            else:
                fail_count += 1

            # Update progress bar
            print_progress(i + 1, total_scenarios, start_time, success_count, fail_count)

            # Print per-scenario completion message
            print()  # New line after progress bar
            if result['status'] == 'success':
                print(f"✓ Completed: {result['scenario']} in {result['duration']:.1f}s | "
                      f"Oil: {result['cum_oil_prod_stb']:.0f} STB, "
                      f"Water Prod: {result['cum_water_prod_stb']:.0f} STB, "
                      f"Water Inj: {result['cum_water_inj_stb']:.0f} STB")
            else:
                print(f"✗ Failed: {result['scenario']} ({result['status']}) - {result.get('error', 'Unknown')[:80]}")

    print()  # New line after progress bar
    print()

    total_duration = time.time() - start_time

    # Print summary
    print("=" * 80)
    print("BATCH SIMULATION COMPLETE")
    print("=" * 80)
    print(f"Total scenarios: {total_scenarios}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Total time: {int(total_duration//60)}m {int(total_duration%60)}s")
    print(f"Average time per simulation: {total_duration/total_scenarios:.1f}s")
    print()

    # Save summary
    summary_file = repo_root / 'results' / 'training_data' / 'batch_summary.json'
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        'total_scenarios': total_scenarios,
        'successful': success_count,
        'failed': fail_count,
        'total_duration_seconds': total_duration,
        'avg_duration_seconds': total_duration / total_scenarios if total_scenarios > 0 else 0,
        'timestamp': datetime.utcnow().isoformat(),
        'parameters': {
            'tfinal': args.tfinal,
            'dt': args.dt,
            'realloc_days': args.realloc_days,
            'workers': args.workers
        },
        'results': results
    }

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to: {summary_file.relative_to(repo_root)}")

    # Print failed scenarios if any
    if fail_count > 0:
        print()
        print("Failed scenarios:")
        for r in results:
            if r['status'] != 'success':
                print(f"  - {r['scenario']}: {r['status']} - {r.get('error', 'Unknown error')[:100]}")

    # Print KPI statistics for successful runs
    if success_count > 0:
        successful_results = [r for r in results if r['status'] == 'success']
        oil_prod = [r['cum_oil_prod_stb'] for r in successful_results]
        water_prod = [r['cum_water_prod_stb'] for r in successful_results]
        water_inj = [r['cum_water_inj_stb'] for r in successful_results]

        print()
        print("KPI Statistics (successful runs):")
        print(f"  Cumulative oil production:")
        print(f"    Min: {min(oil_prod):.1f} STB")
        print(f"    Max: {max(oil_prod):.1f} STB")
        print(f"    Mean: {np.mean(oil_prod):.1f} STB")
        print(f"  Cumulative water production:")
        print(f"    Min: {min(water_prod):.1f} STB")
        print(f"    Max: {max(water_prod):.1f} STB")
        print(f"    Mean: {np.mean(water_prod):.1f} STB")
        print(f"  Cumulative water injection:")
        print(f"    Min: {min(water_inj):.1f} STB")
        print(f"    Max: {max(water_inj):.1f} STB")
        print(f"    Mean: {np.mean(water_inj):.1f} STB")

    return 0 if fail_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
