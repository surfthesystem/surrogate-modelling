import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import qmc


def load_well_coords(base_path: Path):
    """Return producer and injector coordinate arrays (ft) as (P,Np,2), (I,Ni,2).
    Prioritizes selected_wells.csv if available, otherwise uses well_locations_ft.csv.
    """
    # Try selected wells first (these are the ones actually used by simulator)
    loc = base_path / 'selected_wells.csv'
    if not loc.exists():
        # Fallback to all wells
        loc = base_path / 'well_locations_ft.csv'

    if not loc.exists():
        raise FileNotFoundError(f'{loc} not found. Run convert_phase1_data.py first.')

    prods, injs = [], []
    with loc.open('r') as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            t = r.get('well_type', r.get('type','')).lower()
            x = float(r['x_ft']); y = float(r['y_ft'])
            if t.startswith('prod'):
                prods.append([x,y])
            elif t.startswith('inj'):
                injs.append([x,y])
    return np.array(prods, dtype=float), np.array(injs, dtype=float)


def kmeans_2d(points: np.ndarray, k: int, iters: int = 20, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.choice(points.shape[0], size=k, replace=False)
    centers = points[idx]
    for _ in range(iters):
        d2 = ((points[:,None,:] - centers[None,:,:])**2).sum(-1)
        labels = d2.argmin(1)
        for j in range(k):
            m = labels == j
            if m.any():
                centers[j] = points[m].mean(0)
    return labels


def spline_knots_to_months(values_knots: np.ndarray, months: int, knot_months: list[int]) -> np.ndarray:
    """Linear interpolation from knot values per group to monthly values.
    values_knots: shape (G, K)
    returns: (months, G)
    """
    G, K = values_knots.shape
    mgrid = np.arange(months)
    out = np.zeros((months, G))
    for g in range(G):
        out[:, g] = np.interp(mgrid, knot_months, values_knots[g])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', type=str, default='scenarios')
    ap.add_argument('--n', type=int, default=100, help='number of scenarios')
    ap.add_argument('--months', type=int, default=12)
    ap.add_argument('--prod_groups', type=int, default=3)
    ap.add_argument('--inj_groups', type=int, default=3)
    ap.add_argument('--knots', type=str, default='0,3,6,9,12', help='comma-separated months for knots')
    ap.add_argument('--prod_mode', type=str, default='bhp', choices=['bhp','choke'])
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    base_path = repo / 'data' / 'impes_input'
    outdir = repo / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    prod_xy, inj_xy = load_well_coords(base_path)
    Np, Ni = len(prod_xy), len(inj_xy)
    kprod = min(args.prod_groups, Np) or 1
    kinj = min(args.inj_groups, Ni) or 1

    # Group wells by k-means (spatial)
    prod_labels = kmeans_2d(prod_xy, kprod, seed=args.seed) if Np>0 else np.array([])
    inj_labels = kmeans_2d(inj_xy, kinj, seed=args.seed) if Ni>0 else np.array([])

    # Bounds (simple defaults; adjust if config.yaml is different)
    bhp_min, bhp_max = 900.0, 1400.0
    choke_min, choke_max = 0.2, 1.0
    inj_min, inj_max = 300.0, 1200.0

    knot_months = [int(x) for x in args.knots.split(',')]
    # Remove knots beyond months-1
    knot_months = [m for m in knot_months if m <= args.months-1]
    if 0 not in knot_months:
        knot_months = [0] + knot_months
    if args.months-1 not in knot_months:
        knot_months.append(args.months-1)
    knot_months = sorted(set(knot_months))
    K = len(knot_months)

    D = kprod*K + kinj*K
    sobol = qmc.Sobol(d=D, scramble=True, seed=args.seed)
    U = sobol.random(args.n)  # (n, D) in [0,1)

    for i in range(args.n):
        u = U[i]
        # Split into producer and injector parts
        up = u[:kprod*K].reshape(kprod, K)
        ui = u[kprod*K:].reshape(kinj, K)

        if args.prod_mode == 'bhp':
            prod_knots = bhp_min + up * (bhp_max - bhp_min)
        else:
            prod_knots = choke_min + up * (choke_max - choke_min)
        inj_knots = inj_min + ui * (inj_max - inj_min)

        # Interpolate to monthly values per group
        prod_monthly = spline_knots_to_months(prod_knots, args.months, knot_months)  # (months, kprod)
        inj_monthly = spline_knots_to_months(inj_knots, args.months, knot_months)    # (months, kinj)

        # Expand group values to per-well order (producers first then injectors)
        prod_vals = np.zeros((args.months, Np))
        for g in range(kprod):
            prod_vals[:, prod_labels==g] = prod_monthly[:, [g]]
        inj_vals = np.zeros((args.months, Ni))
        for g in range(kinj):
            inj_vals[:, inj_labels==g] = inj_monthly[:, [g]]

        scn = {}
        if args.prod_mode == 'bhp':
            scn['producer_bhp'] = prod_vals.tolist()
        else:
            scn['producer_choke'] = prod_vals.tolist()
        scn['injector_rates'] = inj_vals.tolist()

        with open(outdir / f'doe_{i:04d}.json', 'w') as f:
            json.dump(scn, f)

    print(f'Wrote {args.n} scenarios to {outdir}')


if __name__ == '__main__':
    main()


