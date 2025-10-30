"""
PyTorch Dataset for reservoir simulation surrogate modeling.

Loads NPZ files from simulation results, extracts features, and provides
batched data for GNN-LSTM training.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Dict, List, Optional, Tuple
import glob
import os
from pathlib import Path

from .preprocessing import (
    compute_static_edge_features,
    compute_dynamic_edge_features,
    interpolate_to_wells,
)
from .normalizers import FeatureNormalizer


class ReservoirDataset(Dataset):
    """
    PyTorch Dataset for reservoir simulation data.

    Loads simulation NPZ files and prepares features for GNN-LSTM model.
    """

    def __init__(
        self,
        scenario_paths: List[str],
        well_graph_data: Dict,
        normalizers: Optional[Dict[str, FeatureNormalizer]] = None,
        grid_spacing_m: float = 50.0,
        precomputed_static_edges: Optional[Dict] = None,
        precomputed_time_lag_corr: Optional[Dict] = None,
    ):
        """
        Args:
            scenario_paths: List of paths to scenario NPZ files
            well_graph_data: Dict with 'edge_index_p2p', 'edge_index_i2p',
                            'producer_coords', 'injector_coords'
            normalizers: Dict of fitted FeatureNormalizer objects (if None, no normalization)
            grid_spacing_m: Grid cell size in meters
            precomputed_static_edges: Pre-computed static edge features (optional)
            precomputed_time_lag_corr: Pre-computed time-lagged correlations (optional)
        """
        self.scenario_paths = scenario_paths
        self.well_graph_data = well_graph_data
        self.normalizers = normalizers if normalizers is not None else {}
        self.grid_spacing_m = grid_spacing_m
        self.precomputed_static_edges = precomputed_static_edges
        self.precomputed_time_lag_corr = precomputed_time_lag_corr

        self.num_producers = well_graph_data['num_producers']
        self.num_injectors = well_graph_data['num_injectors']

        print(f"ReservoirDataset initialized with {len(scenario_paths)} scenarios")
        print(f"  Producers: {self.num_producers}, Injectors: {self.num_injectors}")
        print(f"  P2P edges: {well_graph_data['num_edges_p2p']}")
        print(f"  I2P edges: {well_graph_data['num_edges_i2p']}")

    def __len__(self) -> int:
        return len(self.scenario_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load and process a single scenario.

        Returns:
            Dict with keys:
                - producer_features: (T, num_prod, 10)
                - injector_features: (T, num_inj, 8)
                - edge_features_p2p: (T, num_edges_p2p, 10)
                - edge_features_i2p: (T, num_edges_i2p, 10)
                - edge_index_p2p: (2, num_edges_p2p) - static
                - edge_index_i2p: (2, num_edges_i2p) - static
                - targets_oil: (T, num_prod)
                - targets_water: (T, num_prod)
        """
        # Load NPZ file
        data = np.load(self.scenario_paths[idx])

        T = len(data['t'])  # Number of timesteps (typically 61)

        # === Extract Well Data ===
        # Well rates: shape (num_wells, T) where first 5 are injectors, last 10 are producers
        # NOTE: Need to check actual data structure in your NPZ files
        oil_rates = data['well_oil_rate_stb'].T  # (T, 15) -> (T, num_prod)
        water_rates = data['well_water_rate_stb'].T  # (T, 15)

        # Separate producers (last 10 wells) and injectors (first 5 wells)
        # IMPORTANT: Adjust indices based on your actual well ordering
        producer_oil_rates = oil_rates[:, -self.num_producers:]  # (T, 10)
        producer_water_rates = water_rates[:, -self.num_producers:]  # (T, 10)
        injector_water_rates = water_rates[:, :self.num_injectors]  # (T, 5)

        # === Extract Controls ===
        # well_constrainttime: shape (T, num_wells)
        controls = data['well_constrainttime']  # (T-1, 15) or (T, 15)

        # Pad to T timesteps if needed
        if controls.shape[0] < T:
            controls = np.vstack([controls, controls[-1:]])  # Repeat last timestep

        producer_bhp = controls[:, -self.num_producers:]  # (T, 10)
        injector_rates_control = controls[:, :self.num_injectors]  # (T, 5)

        # === Extract Grid Data (for dynamic edge features) ===
        pressure_fields = data['P_plot']  # (num_cells, T) -> reshape to (T, Ny, Nx)
        saturation_fields = data['Sw_plot']  # (num_cells, T)

        # Reshape to 2D grid (100x100)
        Nx, Ny = 100, 100
        pressure_fields = pressure_fields.T.reshape(T, Ny, Nx)  # (T, 100, 100)
        saturation_fields = saturation_fields.T.reshape(T, Ny, Nx)

        # === Build Node Features ===
        producer_features_list = []
        injector_features_list = []

        # Cumulative production (for autoregressive features)
        cum_oil = np.cumsum(producer_oil_rates, axis=0)  # (T, 10)
        cum_water_prod = np.cumsum(producer_water_rates, axis=0)
        cum_water_inj = np.cumsum(injector_water_rates, axis=0)  # (T, 5)

        # Well coordinates (static, from well_graph_data)
        prod_coords = self.well_graph_data['producer_coords']  # (10, 2)
        inj_coords = self.well_graph_data['injector_coords']  # (5, 2)

        # Normalize coordinates to [0, 1] for neural network
        coord_norm = 1.0 / 5000.0  # Domain is ~5000m

        for t in range(T):
            # Producer node features (10-dim)
            prod_feat_t = np.column_stack([
                producer_bhp[t, :],                          # [0] BHP control
                self.well_graph_data.get('perm_at_prod', np.ones(self.num_producers) * 100),  # [1] k
                self.well_graph_data.get('poro_at_prod', np.ones(self.num_producers) * 0.15), # [2] φ
                np.ones(self.num_producers) * 7500,          # [3] depth (constant)
                prod_coords[:, 0] * coord_norm,              # [4] x_norm
                prod_coords[:, 1] * coord_norm,              # [5] y_norm
                cum_oil[t, :],                               # [6] cum_oil
                cum_water_prod[t, :],                        # [7] cum_water
                producer_oil_rates[max(0, t-1), :],          # [8] prev_oil_rate
                producer_water_rates[max(0, t-1), :],        # [9] prev_water_rate
            ])  # Shape: (10, 10)

            # Injector node features (8-dim)
            inj_feat_t = np.column_stack([
                injector_rates_control[t, :],                # [0] injection rate
                self.well_graph_data.get('perm_at_inj', np.ones(self.num_injectors) * 100),  # [1] k
                self.well_graph_data.get('poro_at_inj', np.ones(self.num_injectors) * 0.15), # [2] φ
                np.ones(self.num_injectors) * 7500,          # [3] depth
                inj_coords[:, 0] * coord_norm,               # [4] x_norm
                inj_coords[:, 1] * coord_norm,               # [5] y_norm
                cum_water_inj[t, :],                         # [6] cum_inj
                injector_water_rates[max(0, t-1), :],        # [7] prev_inj_rate
            ])  # Shape: (5, 8)

            producer_features_list.append(prod_feat_t)
            injector_features_list.append(inj_feat_t)

        producer_features = np.array(producer_features_list)  # (T, 10, 10)
        injector_features = np.array(injector_features_list)  # (T, 5, 8)

        # === Build Edge Features ===
        edge_features_p2p_list = []
        edge_features_i2p_list = []

        # Load permeability/porosity fields (static, same for all timesteps)
        # These should be loaded from reservoir config, not from NPZ
        # For now, use dummy values (replace with actual loading)
        perm_field = np.ones((Ny, Nx)) * 100  # TODO: Load from reservoir config
        poro_field = np.ones((Ny, Nx)) * 0.15

        for t in range(T):
            # === P2P edge features ===
            if self.precomputed_static_edges is not None:
                # Use pre-computed static features
                static_p2p = self.precomputed_static_edges['p2p']  # (num_edges, 7)
            else:
                # Compute on-the-fly (slower)
                static_p2p = compute_static_edge_features(
                    self.well_graph_data['edge_index_p2p'],
                    prod_coords,
                    perm_field,
                    poro_field,
                    self.grid_spacing_m
                )

            # Compute dynamic features (pressure/saturation gradients)
            dynamic_p2p = compute_dynamic_edge_features(
                self.well_graph_data['edge_index_p2p'],
                prod_coords,
                pressure_fields[t],
                saturation_fields[t],
                self.grid_spacing_m
            )

            # Combine static (7) + dynamic (3) = 10-dim
            edge_feat_p2p_t = np.concatenate([static_p2p, dynamic_p2p], axis=1)  # (num_edges_p2p, 10)
            edge_features_p2p_list.append(edge_feat_p2p_t)

            # === I2P edge features (similar) ===
            # For I2P, nodes include both injectors and producers
            all_coords = np.vstack([inj_coords, prod_coords])  # (15, 2)

            if self.precomputed_static_edges is not None:
                static_i2p = self.precomputed_static_edges['i2p']
            else:
                static_i2p = compute_static_edge_features(
                    self.well_graph_data['edge_index_i2p'],
                    all_coords,
                    perm_field,
                    poro_field,
                    self.grid_spacing_m
                )

            dynamic_i2p = compute_dynamic_edge_features(
                self.well_graph_data['edge_index_i2p'],
                all_coords,
                pressure_fields[t],
                saturation_fields[t],
                self.grid_spacing_m
            )

            edge_feat_i2p_t = np.concatenate([static_i2p, dynamic_i2p], axis=1)
            edge_features_i2p_list.append(edge_feat_i2p_t)

        edge_features_p2p = np.array(edge_features_p2p_list)  # (T, num_edges_p2p, 10)
        edge_features_i2p = np.array(edge_features_i2p_list)  # (T, num_edges_i2p, 10)

        # === Apply Normalization ===
        if 'BHP' in self.normalizers:
            producer_features[:, :, 0] = self.normalizers['BHP'].transform(producer_features[:, :, 0])
        if 'injection_rate' in self.normalizers:
            injector_features[:, :, 0] = self.normalizers['injection_rate'].transform(injector_features[:, :, 0])
        if 'oil_rate' in self.normalizers:
            producer_oil_rates = self.normalizers['oil_rate'].transform(producer_oil_rates)
        if 'water_rate' in self.normalizers:
            producer_water_rates = self.normalizers['water_rate'].transform(producer_water_rates)

        # === Convert to PyTorch Tensors ===
        return {
            'producer_features': torch.FloatTensor(producer_features),  # (T, 10, 10)
            'injector_features': torch.FloatTensor(injector_features),  # (T, 5, 8)
            'edge_features_p2p': torch.FloatTensor(edge_features_p2p),  # (T, num_edges, 10)
            'edge_features_i2p': torch.FloatTensor(edge_features_i2p),  # (T, num_edges, 10)
            'edge_index_p2p': torch.LongTensor(self.well_graph_data['edge_index_p2p']),  # (2, num_edges)
            'edge_index_i2p': torch.LongTensor(self.well_graph_data['edge_index_i2p']),  # (2, num_edges)
            'targets_oil': torch.FloatTensor(producer_oil_rates),  # (T, 10)
            'targets_water': torch.FloatTensor(producer_water_rates),  # (T, 10)
            'scenario_name': os.path.basename(self.scenario_paths[idx]).replace('.npz', ''),
        }


def create_dataloaders(
    data_dir: str,
    well_graph_data: Dict,
    batch_size: int = 8,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    num_workers: int = 4,
    normalizers: Optional[Dict[str, FeatureNormalizer]] = None,
    random_seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.

    Args:
        data_dir: Directory containing scenario NPZ files
        well_graph_data: Well graph connectivity data
        batch_size: Batch size for training
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        test_split: Fraction of data for testing
        num_workers: Number of dataloader workers
        normalizers: Fitted normalizers (if None, no normalization)
        random_seed: Random seed for reproducibility

    Returns:
        train_loader, val_loader, test_loader
    """
    # Find all scenario NPZ files
    scenario_paths = sorted(glob.glob(os.path.join(data_dir, "doe_*/doe_*.npz")))

    if len(scenario_paths) == 0:
        raise FileNotFoundError(f"No NPZ files found in {data_dir}")

    print(f"Found {len(scenario_paths)} scenario files")

    # Create full dataset
    full_dataset = ReservoirDataset(
        scenario_paths=scenario_paths,
        well_graph_data=well_graph_data,
        normalizers=normalizers,
    )

    # Split into train/val/test
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size

    torch.manual_seed(random_seed)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    print(f"Split: {train_size} train / {val_size} val / {test_size} test")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset loading
    print("Testing ReservoirDataset...")

    from graph_builder import build_well_graphs

    # Load well locations
    import pandas as pd
    wells_df = pd.read_csv('../../data/impes_input/selected_wells.csv')

    prod_wells = wells_df[wells_df['type'] == 'producer']
    inj_wells = wells_df[wells_df['type'] == 'injector']

    prod_coords = prod_wells[['x_m', 'y_m']].values * 3.28084  # m to ft
    inj_coords = inj_wells[['x_m', 'y_m']].values * 3.28084

    # Build graphs
    graph_data = build_well_graphs(prod_coords, inj_coords, p2p_mode='voronoi', i2p_mode='full')
    graph_data['producer_coords'] = prod_coords
    graph_data['injector_coords'] = inj_coords

    # Create dataset
    scenario_paths = sorted(glob.glob('../../results/training_data/doe_*/doe_*.npz'))[:5]

    dataset = ReservoirDataset(
        scenario_paths=scenario_paths,
        well_graph_data=graph_data,
    )

    print(f"\nDataset size: {len(dataset)}")

    # Load first sample
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Producer features shape: {sample['producer_features'].shape}")
    print(f"Targets oil shape: {sample['targets_oil'].shape}")

    print("\n✓ Dataset test passed!")
