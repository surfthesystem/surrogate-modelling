"""
Simplified dataset for getting training started quickly.

This dataset creates basic features from NPZ files to match the model's expected format.
For production, use the full ReservoirDataset with all edge features.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class SimpleReservoirDataset(Dataset):
    """
    Simplified dataset that loads NPZ files and creates basic features.

    Creates minimal features needed for training:
    - Producer node features: [oil_rate, water_rate, pressure, saturation, perm, porosity]
    - Injector node features: [injection_rate, pressure, saturation, perm, porosity]
    - Edge features: [distance, avg_perm]
    """

    def __init__(self, scenario_paths, graph_data, normalize=True):
        """
        Args:
            scenario_paths: List of paths to NPZ files
            graph_data: Loaded graph data from preprocessed/graph_data.npz
            normalize: Whether to normalize features
        """
        self.scenario_paths = scenario_paths
        self.normalize = normalize

        # Extract graph structure
        self.edge_index_p2p = torch.from_numpy(graph_data['edge_index_p2p']).long()
        self.edge_index_i2p = torch.from_numpy(graph_data['edge_index_i2p']).long()
        self.p2p_distances = graph_data['p2p_distances']
        self.i2p_distances = graph_data['i2p_distances']
        self.producer_static = graph_data['producer_static']  # [perm, poro]
        self.injector_static = graph_data['injector_static']

        self.num_producers = int(graph_data['num_producers'])
        self.num_injectors = int(graph_data['num_injectors'])

        print(f"SimpleReservoirDataset: {len(scenario_paths)} scenarios")
        print(f"  Producers: {self.num_producers}, Injectors: {self.num_injectors}")
        print(f"  P2P edges: {self.edge_index_p2p.shape[1]}, I2P edges: {self.edge_index_i2p.shape[1]}")

    def __len__(self):
        return len(self.scenario_paths)

    def __getitem__(self, idx):
        """Load and prepare one scenario."""
        # Load NPZ
        data = np.load(self.scenario_paths[idx])

        T = len(data['t'])  # Number of timesteps

        # === Extract well data ===
        # Wells are indexed: 0-9 producers, 10-14 injectors
        oil_rates = data['well_oil_rate_stb'].T  # (T, 15)
        water_rates = data['well_water_rate_stb'].T  # (T, 15)

        # Split by well type
        producer_oil = oil_rates[:, :self.num_producers]  # (T, 10)
        producer_water = water_rates[:, :self.num_producers]  # (T, 10)
        injector_water = water_rates[:, self.num_producers:]  # (T, 5)

        # Get constraint pressures from producer_pwf if available (constant per well)
        if 'producer_pwf' in data:
            # producer_pwf is (15, 1), take first num_producers
            producer_pwf_const = data['producer_pwf'][:self.num_producers, 0]  # (10,)
        else:
            producer_pwf_const = np.ones(self.num_producers) * 4000  # Default 4000 psi

        # === Build producer features ===
        # Features: [oil_rate, water_rate, pwf_constraint, perm, poro] = 5 features
        # Expand to 10 features by adding zeros for now
        producer_features = np.zeros((T, self.num_producers, 10), dtype=np.float32)
        producer_features[:, :, 0] = producer_oil
        producer_features[:, :, 1] = producer_water
        # Repeat constant pwf for all timesteps
        for t in range(T):
            producer_features[t, :, 2] = producer_pwf_const

        # Add static properties (repeated across time)
        for t in range(T):
            producer_features[t, :, 3] = self.producer_static[:, 0]  # perm
            producer_features[t, :, 4] = self.producer_static[:, 1]  # poro

        # === Build injector features ===
        # Features: [injection_rate, perm, poro] expanded to 8
        injector_features = np.zeros((T, self.num_injectors, 8), dtype=np.float32)
        injector_features[:, :, 0] = injector_water
        for t in range(T):
            injector_features[t, :, 1] = self.injector_static[:, 0]  # perm
            injector_features[t, :, 2] = self.injector_static[:, 1]  # poro

        # === Build edge features ===
        # P2P edge features: [distance, avg_perm, ...] expanded to 10
        num_p2p_edges = self.edge_index_p2p.shape[1]
        edge_features_p2p = np.zeros((T, num_p2p_edges, 10), dtype=np.float32)
        for t in range(T):
            edge_features_p2p[t, :, 0] = self.p2p_distances / 10000  # Normalize distance
            # Compute avg perm between connected producers
            for e in range(num_p2p_edges):
                src, dst = self.edge_index_p2p[:, e]
                avg_perm = (self.producer_static[src, 0] + self.producer_static[dst, 0]) / 2
                edge_features_p2p[t, e, 1] = avg_perm / 1000  # Normalize

        # I2P edge features: [distance, avg_perm, ...] expanded to 10
        num_i2p_edges = self.edge_index_i2p.shape[1]
        edge_features_i2p = np.zeros((T, num_i2p_edges, 10), dtype=np.float32)
        for t in range(T):
            edge_features_i2p[t, :, 0] = self.i2p_distances / 10000  # Normalize distance
            # Compute avg perm between injector and producer
            for e in range(num_i2p_edges):
                inj_idx, prod_idx = self.edge_index_i2p[:, e]
                avg_perm = (self.injector_static[inj_idx, 0] + self.producer_static[prod_idx, 0]) / 2
                edge_features_i2p[t, e, 1] = avg_perm / 1000  # Normalize

        # === Targets ===
        target_oil = producer_oil  # (T, num_producers)
        target_water = producer_water  # (T, num_producers)

        # Convert to tensors
        return {
            'producer_features': torch.from_numpy(producer_features).float(),
            'injector_features': torch.from_numpy(injector_features).float(),
            'edge_features_p2p': torch.from_numpy(edge_features_p2p).float(),
            'edge_features_i2p': torch.from_numpy(edge_features_i2p).float(),
            'edge_index_p2p': self.edge_index_p2p,
            'edge_index_i2p': self.edge_index_i2p,
            'target_oil_rate': torch.from_numpy(target_oil).float(),
            'target_water_rate': torch.from_numpy(target_water).float(),
        }


def collate_batch(batch):
    """
    Custom collate function to handle batching of graph data.

    Edge indices are the same for all samples, so we don't batch them.
    """
    # Stack features (all have same dimensions)
    producer_features = torch.stack([item['producer_features'] for item in batch])
    injector_features = torch.stack([item['injector_features'] for item in batch])
    edge_features_p2p = torch.stack([item['edge_features_p2p'] for item in batch])
    edge_features_i2p = torch.stack([item['edge_features_i2p'] for item in batch])
    target_oil = torch.stack([item['target_oil_rate'] for item in batch])
    target_water = torch.stack([item['target_water_rate'] for item in batch])

    # Edge indices are the same for all samples
    edge_index_p2p = batch[0]['edge_index_p2p']
    edge_index_i2p = batch[0]['edge_index_i2p']

    return {
        'producer_features': producer_features,
        'injector_features': injector_features,
        'edge_features_p2p': edge_features_p2p,
        'edge_features_i2p': edge_features_i2p,
        'edge_index_p2p': edge_index_p2p,
        'edge_index_i2p': edge_index_i2p,
        'target_oil_rate': target_oil,
        'target_water_rate': target_water,
    }
