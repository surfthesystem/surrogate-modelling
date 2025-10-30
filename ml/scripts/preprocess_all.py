#!/usr/bin/env python
"""
Preprocess all simulation data for GNN-LSTM training.

Builds well connectivity graphs and computes static edge features.
Dynamic features (pressure/saturation gradients) computed on-the-fly during training.

Usage:
    python ml/scripts/preprocess_all.py --data_dir results/training_data --output_dir ml/data/preprocessed
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.data.graph_builder import build_well_graphs


def load_well_locations(csv_path):
    """Load well locations from CSV file."""
    df = pd.read_csv(csv_path)

    # Separate producers and injectors
    producers = df[df['well_type'] == 'producer'].copy()
    injectors = df[df['well_type'] == 'injector'].copy()

    print(f"Loaded {len(producers)} producers, {len(injectors)} injectors")

    # Get coordinates in feet (already in the CSV)
    prod_coords = producers[['x_ft', 'y_ft']].values
    inj_coords = injectors[['x_ft', 'y_ft']].values

    return prod_coords, inj_coords, producers, injectors


def compute_static_features(graph_data, producers, injectors):
    """
    Compute static features for nodes and edges.

    These don't change with time (permeability, distance, etc.)
    Dynamic features (pressure, saturation) computed during training.
    """
    # Node features (static permeability and porosity at well locations)
    producer_static = np.column_stack([
        producers['perm_mD'].values,
        producers['porosity'].values,
    ])

    injector_static = np.column_stack([
        injectors['perm_mD'].values,
        injectors['porosity'].values,
    ])

    # Edge features (distance-based, will be enhanced with dynamic features during training)
    # For now, just compute distances
    prod_coords = producers[['x_ft', 'y_ft']].values
    inj_coords = injectors[['x_ft', 'y_ft']].values

    # P2P edge distances
    edge_index_p2p = graph_data['edge_index_p2p']
    p2p_distances = np.zeros(edge_index_p2p.shape[1])
    for i in range(edge_index_p2p.shape[1]):
        src, dst = edge_index_p2p[:, i]
        p2p_distances[i] = np.linalg.norm(prod_coords[src] - prod_coords[dst])

    # I2P edge distances
    edge_index_i2p = graph_data['edge_index_i2p']
    i2p_distances = np.zeros(edge_index_i2p.shape[1])
    for i in range(edge_index_i2p.shape[1]):
        inj_idx, prod_idx = edge_index_i2p[:, i]
        i2p_distances[i] = np.linalg.norm(inj_coords[inj_idx] - prod_coords[prod_idx])

    return {
        'producer_static': producer_static,
        'injector_static': injector_static,
        'p2p_distances': p2p_distances,
        'i2p_distances': i2p_distances,
    }


def main():
    parser = argparse.ArgumentParser(description='Preprocess simulation data')
    parser.add_argument('--data_dir', default='results/training_data', help='Directory with NPZ files')
    parser.add_argument('--output_dir', default='ml/data/preprocessed', help='Output directory')
    parser.add_argument('--wells_csv', default='data/impes_input/selected_wells.csv', help='Well locations CSV')
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("GNN-LSTM Preprocessing Pipeline")
    print("="*80)

    # Load well locations
    print("\n1. Loading well locations...")
    prod_coords, inj_coords, producers, injectors = load_well_locations(args.wells_csv)

    print(f"   Producers: {len(prod_coords)}")
    print(f"   Injectors: {len(inj_coords)}")
    print(f"   Domain size: {prod_coords[:, 0].max():.0f} ft × {prod_coords[:, 1].max():.0f} ft")

    # Build graphs
    print("\n2. Building well connectivity graphs...")
    graph_data = build_well_graphs(
        prod_coords,
        inj_coords,
        p2p_mode='voronoi',  # Voronoi for producer-producer
        i2p_mode='full',     # Full bipartite for injector-producer
    )

    print(f"   P2P edges (Voronoi): {graph_data['edge_index_p2p'].shape[1]}")
    print(f"   I2P edges (bipartite): {graph_data['edge_index_i2p'].shape[1]}")
    avg_p2p = graph_data['edge_index_p2p'].shape[1] / len(prod_coords)
    print(f"   Average P2P connectivity: {avg_p2p:.1f} edges/well")

    # Compute static features
    print("\n3. Computing static features...")
    static_features = compute_static_features(graph_data, producers, injectors)

    print(f"   Producer static features: {static_features['producer_static'].shape}")
    print(f"   Injector static features: {static_features['injector_static'].shape}")

    # Verify data availability
    print("\n4. Verifying simulation data...")
    data_dir = Path(args.data_dir)
    npz_files = sorted(list(data_dir.glob('doe_*/doe_*.npz')))
    print(f"   Found {len(npz_files)} NPZ files")

    if len(npz_files) == 0:
        print("   ERROR: No NPZ files found!")
        return

    # Load one file to verify structure
    sample_data = np.load(str(npz_files[0]))
    print(f"   Sample file: {npz_files[0].name}")
    print(f"   Keys: {list(sample_data.keys())}")
    print(f"   Time steps: {sample_data['t'].shape[0]}")
    print(f"   Wells: {sample_data['well_oil_rate_stb'].shape[0]}")

    # Save preprocessed data
    print("\n5. Saving preprocessed data...")
    save_path = output_path / 'graph_data.npz'
    np.savez(
        save_path,
        # Graph structure
        edge_index_p2p=graph_data['edge_index_p2p'],
        edge_index_i2p=graph_data['edge_index_i2p'],
        # Static features
        producer_static=static_features['producer_static'],
        injector_static=static_features['injector_static'],
        p2p_distances=static_features['p2p_distances'],
        i2p_distances=static_features['i2p_distances'],
        # Coordinates
        producer_coords=prod_coords,
        injector_coords=inj_coords,
        # Metadata
        num_producers=len(prod_coords),
        num_injectors=len(inj_coords),
        num_scenarios=len(npz_files),
    )

    print(f"   ✓ Saved to: {save_path}")
    print(f"   File size: {save_path.stat().st_size / 1024:.1f} KB")

    # Save scenario file list
    scenario_list = [str(f) for f in npz_files]
    list_path = output_path / 'scenario_list.txt'
    with open(list_path, 'w') as f:
        f.write('\n'.join(scenario_list))
    print(f"   ✓ Saved scenario list: {list_path}")

    print("\n" + "="*80)
    print("✓ Preprocessing complete!")
    print("="*80)
    print(f"\nNext step: Run training with:")
    print(f"  python ml/scripts/train.py --config ml/training/config.yaml")


if __name__ == '__main__':
    main()
