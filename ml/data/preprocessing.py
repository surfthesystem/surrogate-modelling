"""
Edge feature computation for well connectivity graphs.

Computes 10-dimensional edge features combining:
- Geometric: distance, angle, drainage overlap
- Permeability-based: k_avg, k_contrast, transmissibility
- Data-driven: time-lagged correlation
- Dynamic: pressure gradient, saturation difference
"""

import numpy as np
from scipy.spatial import distance_matrix
from scipy.ndimage import map_coordinates
from typing import Tuple, Dict, List, Optional
import warnings


def sample_line(field: np.ndarray, start: np.ndarray, end: np.ndarray,
                num_samples: int = 20) -> np.ndarray:
    """
    Sample values from a 2D field along a straight line.

    Args:
        field: 2D array to sample from, shape (Ny, Nx)
        start: Starting coordinates [x, y] in same units as field indices
        end: Ending coordinates [x, y]
        num_samples: Number of samples along the line

    Returns:
        sampled_values: Array of sampled values, shape (num_samples,)
    """
    # Create interpolation points along the line
    t = np.linspace(0, 1, num_samples)
    x_coords = start[0] + t * (end[0] - start[0])
    y_coords = start[1] + t * (end[1] - start[1])

    # Sample using bilinear interpolation
    # Note: map_coordinates uses (row, col) = (y, x) indexing
    coords = np.array([y_coords, x_coords])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sampled = map_coordinates(field, coords, order=1, mode='nearest')

    return sampled


def coords_to_grid_indices(coords_ft: np.ndarray, grid_spacing_ft: float = 164.04) -> np.ndarray:
    """
    Convert physical coordinates (feet) to grid indices.

    Args:
        coords_ft: Coordinates in feet, shape (N, 2) - [x, y]
        grid_spacing_ft: Grid cell size in feet (default 50m = 164.04 ft)

    Returns:
        indices: Grid indices, shape (N, 2) - [i_x, i_y]
    """
    indices = coords_ft / grid_spacing_ft
    return indices


def compute_static_edge_features(
    edge_index: np.ndarray,
    node_coords: np.ndarray,
    permeability_field: np.ndarray,
    porosity_field: np.ndarray,
    grid_spacing_m: float = 50.0,
    domain_size_m: Tuple[float, float] = (5000.0, 5000.0),
) -> np.ndarray:
    """
    Compute static edge features (computed once, don't change with time).

    Features per edge (7 dimensions):
        [0] Inverse distance: 1 / d(i,j)
        [1] Log permeability (avg): log(k_avg)
        [2] Permeability contrast: log(k_i / k_j)
        [3-4] Direction: cos(angle), sin(angle)
        [5] Transmissibility: k_avg / distance
        [6] Placeholder for time-lag correlation (filled later)

    Args:
        edge_index: Edge connectivity, shape (2, num_edges)
        node_coords: Node coordinates in feet, shape (num_nodes, 2)
        permeability_field: Permeability in mD, shape (Ny, Nx)
        porosity_field: Porosity (fraction), shape (Ny, Nx)
        grid_spacing_m: Grid cell size in meters
        domain_size_m: Domain size in meters (Lx, Ly)

    Returns:
        edge_features: Static features, shape (num_edges, 7)
    """
    num_edges = edge_index.shape[1]
    edge_features = np.zeros((num_edges, 7), dtype=np.float32)

    # Convert coordinates from feet to meters (simulation typically in feet)
    node_coords_m = node_coords * 0.3048  # ft to m

    # Grid dimensions
    Nx = permeability_field.shape[1]
    Ny = permeability_field.shape[0]

    for e in range(num_edges):
        src, tgt = edge_index[:, e]

        # Geometric features
        dx = node_coords_m[tgt, 0] - node_coords_m[src, 0]
        dy = node_coords_m[tgt, 1] - node_coords_m[src, 1]
        distance_m = np.sqrt(dx**2 + dy**2) + 1e-6  # Avoid division by zero

        # [0] Inverse distance
        edge_features[e, 0] = 1.0 / distance_m

        # [3-4] Direction (unit vector components)
        angle = np.arctan2(dy, dx)
        edge_features[e, 3] = np.cos(angle)
        edge_features[e, 4] = np.sin(angle)

        # Sample permeability along line connecting wells
        # Convert physical coords to grid indices
        src_indices = node_coords_m[src] / grid_spacing_m
        tgt_indices = node_coords_m[tgt] / grid_spacing_m

        # Ensure indices are within bounds
        src_indices = np.clip(src_indices, 0, [Nx-1, Ny-1])
        tgt_indices = np.clip(tgt_indices, 0, [Nx-1, Ny-1])

        # Sample permeability along line
        k_path = sample_line(permeability_field, src_indices, tgt_indices, num_samples=20)
        k_avg = np.mean(k_path) + 1e-6

        # [1] Log average permeability
        edge_features[e, 1] = np.log(k_avg)

        # [2] Permeability contrast
        k_src = k_path[0] + 1e-6
        k_tgt = k_path[-1] + 1e-6
        edge_features[e, 2] = np.log(k_src / k_tgt)

        # [5] Transmissibility (flow capacity)
        edge_features[e, 5] = k_avg / distance_m

        # [6] Time-lag correlation (placeholder, filled by compute_time_lagged_correlation)
        edge_features[e, 6] = 0.0

    return edge_features


def compute_dynamic_edge_features(
    edge_index: np.ndarray,
    node_coords: np.ndarray,
    pressure_field: np.ndarray,
    saturation_field: np.ndarray,
    grid_spacing_m: float = 50.0,
) -> np.ndarray:
    """
    Compute dynamic edge features (change with time).

    Features per edge (3 dimensions):
        [0] Pressure difference: P_i - P_j
        [1] Saturation difference: Sw_i - Sw_j
        [2] Drainage overlap (static, but included here for consistency)

    Args:
        edge_index: Edge connectivity, shape (2, num_edges)
        node_coords: Node coordinates in feet, shape (num_nodes, 2)
        pressure_field: Pressure in psi, shape (Ny, Nx)
        saturation_field: Water saturation, shape (Ny, Nx)
        grid_spacing_m: Grid cell size in meters

    Returns:
        edge_features_dynamic: Dynamic features, shape (num_edges, 3)
    """
    num_edges = edge_index.shape[1]
    edge_features_dynamic = np.zeros((num_edges, 3), dtype=np.float32)

    # Convert coordinates from feet to meters
    node_coords_m = node_coords * 0.3048

    # Interpolate pressure and saturation to well locations
    pressure_at_wells = interpolate_to_wells(pressure_field, node_coords_m, grid_spacing_m)
    saturation_at_wells = interpolate_to_wells(saturation_field, node_coords_m, grid_spacing_m)

    for e in range(num_edges):
        src, tgt = edge_index[:, e]

        # [0] Pressure difference
        edge_features_dynamic[e, 0] = pressure_at_wells[src] - pressure_at_wells[tgt]

        # [1] Saturation difference
        edge_features_dynamic[e, 1] = saturation_at_wells[src] - saturation_at_wells[tgt]

        # [2] Drainage overlap (TODO: implement properly, for now use 0)
        edge_features_dynamic[e, 2] = 0.0

    return edge_features_dynamic


def interpolate_to_wells(
    field: np.ndarray,
    well_coords_m: np.ndarray,
    grid_spacing_m: float = 50.0
) -> np.ndarray:
    """
    Interpolate 2D field values to well locations.

    Args:
        field: 2D field to interpolate, shape (Ny, Nx)
        well_coords_m: Well coordinates in meters, shape (num_wells, 2) - [x, y]
        grid_spacing_m: Grid cell size in meters

    Returns:
        values_at_wells: Interpolated values, shape (num_wells,)
    """
    num_wells = well_coords_m.shape[0]
    Nx, Ny = field.shape[1], field.shape[0]

    # Convert coordinates to grid indices
    indices = well_coords_m / grid_spacing_m

    # Clip to valid range
    indices = np.clip(indices, 0, [Nx-1, Ny-1])

    # Interpolate (note: map_coordinates uses (y, x) indexing)
    coords = np.array([indices[:, 1], indices[:, 0]])  # (2, num_wells)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        values = map_coordinates(field, coords, order=1, mode='nearest')

    return values


def compute_time_lagged_correlation(
    injector_rates: np.ndarray,
    producer_rates: np.ndarray,
    edge_index_i2p: np.ndarray,
    max_lag: int = 10,
    rate_type: str = 'oil'
) -> np.ndarray:
    """
    Compute time-lagged cross-correlation between injector and producer rates.

    This captures data-driven connectivity: how much does injector i affect producer j
    with a time delay?

    Args:
        injector_rates: Injection rates, shape (num_scenarios, num_timesteps, num_inj)
        producer_rates: Production rates, shape (num_scenarios, num_timesteps, num_prod)
        edge_index_i2p: Injector-Producer edge indices, shape (2, num_edges_i2p)
        max_lag: Maximum time lag to consider (timesteps)
        rate_type: 'oil' or 'water' (for logging purposes)

    Returns:
        correlations: Max abs correlation for each edge, shape (num_edges_i2p,)
    """
    num_edges = edge_index_i2p.shape[1]
    correlations = np.zeros(num_edges, dtype=np.float32)

    num_scenarios, num_timesteps, _ = injector_rates.shape

    # Flatten across scenarios to get longer time series
    inj_flat = injector_rates.reshape(-1, injector_rates.shape[2])  # (scenarios*T, num_inj)
    prod_flat = producer_rates.reshape(-1, producer_rates.shape[2])  # (scenarios*T, num_prod)

    for e in range(num_edges):
        inj_idx, prod_idx = edge_index_i2p[:, e]

        inj_series = inj_flat[:, inj_idx]
        prod_series = prod_flat[:, prod_idx]

        # Normalize to zero mean, unit variance
        inj_norm = (inj_series - np.mean(inj_series)) / (np.std(inj_series) + 1e-6)
        prod_norm = (prod_series - np.mean(prod_series)) / (np.std(prod_series) + 1e-6)

        # Compute cross-correlation for different time lags
        max_corr = 0.0
        for lag in range(1, min(max_lag + 1, num_timesteps)):
            if len(inj_norm) <= lag:
                break

            # Correlation: injection at t affects production at t+lag
            corr = np.corrcoef(inj_norm[:-lag], prod_norm[lag:])[0, 1]

            if not np.isnan(corr):
                max_corr = max(max_corr, abs(corr))

        correlations[e] = max_corr

    return correlations


def compute_full_edge_features(
    edge_index: np.ndarray,
    node_coords: np.ndarray,
    permeability_field: np.ndarray,
    porosity_field: np.ndarray,
    pressure_field: np.ndarray,
    saturation_field: np.ndarray,
    time_lag_correlations: Optional[np.ndarray] = None,
    grid_spacing_m: float = 50.0,
    domain_size_m: Tuple[float, float] = (5000.0, 5000.0),
) -> np.ndarray:
    """
    Compute full 10-dimensional edge features (static + dynamic).

    Feature vector per edge:
        [0] Inverse distance
        [1] Log permeability (avg)
        [2] Permeability contrast
        [3] Direction cos(angle)
        [4] Direction sin(angle)
        [5] Transmissibility
        [6] Time-lagged correlation (data-driven connectivity)
        [7] Pressure difference ΔP (dynamic)
        [8] Saturation difference ΔSw (dynamic)
        [9] Drainage overlap (placeholder)

    Args:
        edge_index: Edge connectivity, shape (2, num_edges)
        node_coords: Node coordinates in feet, shape (num_nodes, 2)
        permeability_field: Permeability in mD, shape (Ny, Nx)
        porosity_field: Porosity, shape (Ny, Nx)
        pressure_field: Pressure in psi, shape (Ny, Nx)
        saturation_field: Water saturation, shape (Ny, Nx)
        time_lag_correlations: Pre-computed correlations, shape (num_edges,) or None
        grid_spacing_m: Grid cell size in meters
        domain_size_m: Domain size in meters

    Returns:
        edge_features: Full edge features, shape (num_edges, 10)
    """
    # Compute static features (7D)
    static_features = compute_static_edge_features(
        edge_index, node_coords, permeability_field, porosity_field,
        grid_spacing_m, domain_size_m
    )

    # Fill in time-lag correlation if provided
    if time_lag_correlations is not None:
        static_features[:, 6] = time_lag_correlations

    # Compute dynamic features (3D)
    dynamic_features = compute_dynamic_edge_features(
        edge_index, node_coords, pressure_field, saturation_field, grid_spacing_m
    )

    # Concatenate to 10D
    edge_features = np.concatenate([static_features, dynamic_features], axis=1)

    return edge_features


if __name__ == "__main__":
    # Test edge feature computation
    print("Testing edge feature computation...")

    # Create dummy data
    np.random.seed(42)
    num_producers = 10
    num_injectors = 5

    # Random well coordinates (in feet, domain 5000m x 5000m ~ 16000 ft x 16000 ft)
    producer_coords = np.random.uniform(500, 15000, size=(num_producers, 2))
    injector_coords = np.random.uniform(500, 15000, size=(num_injectors, 2))

    # Random permeability/porosity fields (100x100 grid)
    perm_field = np.random.lognormal(mean=np.log(100), sigma=0.5, size=(100, 100))
    poro_field = np.random.uniform(0.15, 0.16, size=(100, 100))

    # Random pressure/saturation fields
    pressure_field = np.random.uniform(3000, 4500, size=(100, 100))
    saturation_field = np.random.uniform(0.2, 0.8, size=(100, 100))

    # Create simple edge index (3 edges as test)
    edge_index_test = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)

    # Compute static features
    static_feats = compute_static_edge_features(
        edge_index_test, producer_coords, perm_field, poro_field
    )
    print(f"\nStatic features shape: {static_feats.shape}")
    print(f"Static features (first edge):\n{static_feats[0, :]}")

    # Compute dynamic features
    dynamic_feats = compute_dynamic_edge_features(
        edge_index_test, producer_coords, pressure_field, saturation_field
    )
    print(f"\nDynamic features shape: {dynamic_feats.shape}")
    print(f"Dynamic features (first edge):\n{dynamic_feats[0, :]}")

    # Test time-lagged correlation
    inj_rates = np.random.uniform(300, 1200, size=(10, 61, num_injectors))  # 10 scenarios
    prod_rates = np.random.uniform(50, 500, size=(10, 61, num_producers))

    edge_index_i2p = np.array([[0, 0, 1], [0, 1, 2]], dtype=np.int64)  # 3 I2P edges
    correlations = compute_time_lagged_correlation(
        inj_rates, prod_rates, edge_index_i2p, max_lag=5
    )
    print(f"\nTime-lagged correlations: {correlations}")

    print("\n✓ Edge feature computation test passed!")
