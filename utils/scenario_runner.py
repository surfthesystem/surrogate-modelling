import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def detect_well_counts(base_path: Path, fallback_prod: int = 10, fallback_inj: int = 5) -> tuple[int, int]:
    """Try to infer number of producers/injectors from selected_wells.csv; fallback to defaults."""
    sel = base_path / 'selected_wells.csv'
    if not sel.exists():
        return fallback_prod, fallback_inj
    try:
        with sel.open('r') as f:
            rdr = csv.reader(f)
            rows = list(rdr)
        # Heuristic: count IDs starting with PROD_ and INJ_
        prod = 0
        inj = 0
        for row in rows[1:]:
            for cell in row:
                if isinstance(cell, str):
                    if cell.strip().startswith('PROD_'):
                        prod += 1
                    if cell.strip().startswith('INJ_'):
                        inj += 1
        # If file lists them once each, use detected; else fall back
        if prod > 0 and inj > 0:
            return prod, inj
    except Exception:
        pass
    return fallback_prod, fallback_inj


def write_schedule_csvs(
    base_path: Path,
    n_months: int,
    prod_schedule_bhp: np.ndarray | None,
    prod_schedule_choke: np.ndarray | None,
    inj_schedule_rates: np.ndarray | None,
) -> None:
    base_path.mkdir(parents=True, exist_ok=True)
    if prod_schedule_bhp is not None:
        np.savetxt(base_path / 'schedule_producer_bhp.csv', prod_schedule_bhp, delimiter=',', fmt='%.3f')
    if prod_schedule_choke is not None:
        np.savetxt(base_path / 'schedule_producer_choke.csv', prod_schedule_choke, delimiter=',', fmt='%.3f')
    if inj_schedule_rates is not None:
        np.savetxt(base_path / 'schedule_injector_rates.csv', inj_schedule_rates, delimiter=',', fmt='%.3f')


def run_impes(sim_dir: Path, tfinal: float, dt: float, realloc_days: float, env: dict | None = None, live: bool = False):
    cmd = [
        sys.executable,
        'IMPES_phase1.py',
        '--tfinal', str(tfinal),
        '--dt', str(dt),
        '--realloc_days', str(realloc_days),
    ]
    if live:
        # Stream output to this process' stdout/stderr
        return subprocess.run(cmd, cwd=str(sim_dir), env=env)
    else:
        return subprocess.run(cmd, cwd=str(sim_dir), capture_output=True, text=True, env=env)


def extract_kpis(npz_path: Path) -> dict:
    d = np.load(npz_path)
    kpis = {
        'cum_oil_prod_stb': float(d['well_cum_oil_prod_stb'][:,-1].sum()),
        'cum_water_prod_stb': float(d['well_cum_water_prod_stb'][:,-1].sum()),
        'cum_water_inj_stb': float(d['well_cum_water_inj_stb'][:,-1].sum()),
        'timesteps': int(len(d['t']) - 1),
        'days': float(d['t'][-1]),
    }
    return kpis


def load_scenario_json(path: Path) -> dict:
    with path.open('r') as f:
        return json.load(f)


def build_schedules_from_scenario(scn: dict, n_months: int, n_prod: int, n_inj: int) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Return (prod_bhp[n_months,n_prod], prod_choke[n_months,n_prod], inj_rates[n_months,n_inj]). Missing → None."""
    prod_bhp = None
    prod_choke = None
    inj_rates = None

    if 'producer_bhp' in scn:
        arr = np.array(scn['producer_bhp'], dtype=float)
        prod_bhp = np.broadcast_to(arr, (n_months, n_prod)) if arr.ndim == 1 else arr
    if 'producer_choke' in scn:
        arr = np.array(scn['producer_choke'], dtype=float)
        prod_choke = np.broadcast_to(arr, (n_months, n_prod)) if arr.ndim == 1 else arr
    if 'injector_rates' in scn:
        arr = np.array(scn['injector_rates'], dtype=float)
        inj_rates = np.broadcast_to(arr, (n_months, n_inj)) if arr.ndim == 1 else arr

    # Trim/pad to correct shapes
    def _fit(arr: np.ndarray | None, cols: int) -> np.ndarray | None:
        if arr is None:
            return None
        a = np.array(arr, dtype=float)
        # rows
        if a.shape[0] != n_months:
            if a.shape[0] > n_months:
                a = a[:n_months]
            else:
                pad = np.tile(a[-1], (n_months - a.shape[0], 1))
                a = np.vstack([a, pad])
        # cols
        if a.shape[1] != cols:
            if a.shape[1] > cols:
                a = a[:, :cols]
            else:
                pad = np.tile(a[:, [-1]], (1, cols - a.shape[1]))
                a = np.hstack([a, pad])
        return a

    prod_bhp = _fit(prod_bhp, n_prod)
    prod_choke = _fit(prod_choke, n_prod)
    inj_rates = _fit(inj_rates, n_inj)
    return prod_bhp, prod_choke, inj_rates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, help='Path to scenario JSON; if directory, run all *.json inside')
    parser.add_argument('--outdir', type=str, default='../results/scenarios')
    parser.add_argument('--n_months', type=int, default=12)
    parser.add_argument('--n_prod', type=int, default=None)
    parser.add_argument('--n_inj', type=int, default=None)
    parser.add_argument('--tfinal', type=float, default=365)
    parser.add_argument('--dt', type=float, default=1.0)
    parser.add_argument('--realloc_days', type=float, default=30.0)
    parser.add_argument('--live', action='store_true', help='Stream simulator output instead of capturing')
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    sim_dir = repo_root / 'simulator'
    base_path = repo_root / 'data' / 'impes_input'
    _ensure_dir(base_path)

    # Detect well counts if not provided
    d_prod, d_inj = detect_well_counts(base_path)
    n_prod = args.n_prod or d_prod
    n_inj = args.n_inj or d_inj

    # Scenarios to run
    scen_paths = []
    if args.scenario:
        p = Path(args.scenario)
        if p.is_dir():
            scen_paths = sorted(p.glob('*.json'))
        else:
            scen_paths = [p]
    else:
        print('No scenario provided; exiting.')
        return 0

    outdir = Path(args.outdir)
    _ensure_dir(outdir)
    summary_rows = []

    for scen_file in scen_paths:
        scn = load_scenario_json(scen_file)
        prod_bhp, prod_choke, inj_rates = build_schedules_from_scenario(scn, args.n_months, n_prod, n_inj)

        # Write schedules
        write_schedule_csvs(base_path, args.n_months, prod_bhp, prod_choke, inj_rates)

        # Run IMPES
        run_id = scen_file.stem + '_' + datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
        print(f'Running scenario {scen_file.name} as {run_id} ...')
        res = run_impes(sim_dir, args.tfinal, args.dt, args.realloc_days, live=args.live)
        if res.returncode != 0:
            if not args.live:
                print(f'  FAILED: {res.stderr[:500]}')
            else:
                print('  FAILED (see console output)')
            summary_rows.append({'run_id': run_id, 'scenario': scen_file.name, 'status': 'failed'})
            continue

        # Find latest NPZ
        impes_out = repo_root / 'results' / 'impes_sim'
        npzs = sorted(impes_out.glob('Phase1_n*_t*_days.npz'), key=lambda p: p.stat().st_mtime, reverse=True)
        if not npzs:
            print('  WARNING: No NPZ found; skipping KPI extraction.')
            summary_rows.append({'run_id': run_id, 'scenario': scen_file.name, 'status': 'no_npz'})
            continue
        kpis = extract_kpis(npzs[0])

        # Save run artifacts
        run_dir = outdir / run_id
        _ensure_dir(run_dir)
        with (run_dir / 'kpis.json').open('w') as f:
            json.dump(kpis, f, indent=2)
        if not args.live:
            with (run_dir / 'stdout.txt').open('w') as f:
                f.write(res.stdout)
            with (run_dir / 'stderr.txt').open('w') as f:
                f.write(res.stderr)

        row = {'run_id': run_id, 'scenario': scen_file.name, 'status': 'ok'}
        row.update(kpis)
        summary_rows.append(row)
        print(f"  OK: oil={kpis['cum_oil_prod_stb']:.1f} stb, water_prod={kpis['cum_water_prod_stb']:.1f} stb, water_inj={kpis['cum_water_inj_stb']:.1f} stb")

    # Write summary CSV
    if summary_rows:
        out_csv = outdir / 'summary.csv'
        keys = sorted({k for r in summary_rows for k in r.keys()})
        with out_csv.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(summary_rows)
        print(f'Summary written to {out_csv}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())


