"""
Graph construction utilities for well networks.

Builds spatial connectivity graphs for Producer-Producer and Injector-Producer pairs
using Voronoi diagrams, k-nearest neighbors, or distance thresholds.
"""

import numpy as np
from scipy.spatial import Voronoi, distance_matrix
from scipy.sparse import csr_matrix
from typing import Tuple, Dict, Optional
import networkx as nx


def compute_voronoi_connectivity(coords: np.ndarray, periodic: bool = False) -> np.ndarray:
    """
    Compute well connectivity based on Voronoi diagram adjacency.

    Args:
        coords: Well coordinates, shape (num_wells, 2) - [x, y] in feet or meters
        periodic: Whether to use periodic boundary conditions (not implemented)

    Returns:
        edge_index: Connectivity matrix, shape (2, num_edges)
                   edge_index[0, :] = source nodes
                   edge_index[1, :] = target nodes
    """
    num_wells = coords.shape[0]

    if num_wells < 4:
        # Voronoi requires at least 4 points, fall back to k-nearest
        return knearest_connectivity(coords, k=min(3, num_wells-1))

    # Compute Voronoi diagram
    vor = Voronoi(coords)

    # Build adjacency from Voronoi ridge points
    edges = []
    for ridge_points in vor.ridge_points:
        i, j = ridge_points
        if i < num_wells and j < num_wells:  # Valid well indices
            edges.append((i, j))
            edges.append((j, i))  # Add reverse edge for undirected graph

    if len(edges) == 0:
        # Fallback if Voronoi fails
        return knearest_connectivity(coords, k=3)

    edges = np.array(edges, dtype=np.int64).T  # Shape: (2, num_edges)

    # Remove duplicates
    edges = np.unique(edges, axis=1)

    return edges


def knearest_connectivity(
    coords: np.ndarray,
    k: int = 5,
    max_distance: Optional[float] = None
) -> np.ndarray:
    """
    Compute k-nearest neighbor connectivity.

    Args:
        coords: Well coordinates, shape (num_wells, 2)
        k: Number of nearest neighbors to connect
        max_distance: Maximum distance for connection (None = no limit)

    Returns:
        edge_index: Connectivity matrix, shape (2, num_edges)
    """
    num_wells = coords.shape[0]
    k = min(k, num_wells - 1)  # Can't have more neighbors than wells-1

    # Compute pairwise distances
    dist_matrix = distance_matrix(coords, coords)

    edges = []
    for i in range(num_wells):
        # Get k nearest neighbors (excluding self)
        neighbors = np.argsort(dist_matrix[i, :])[1:k+1]

        for j in neighbors:
            if max_distance is None or dist_matrix[i, j] <= max_distance:
                edges.append((i, j))

    edges = np.array(edges, dtype=np.int64).T  # Shape: (2, num_edges)

    return edges


def distance_threshold_connectivity(
    coords: np.ndarray,
    threshold: float
) -> np.ndarray:
    """
    Connect wells within a distance threshold.

    Args:
        coords: Well coordinates, shape (num_wells, 2)
        threshold: Maximum distance for connection (feet or meters)

    Returns:
        edge_index: Connectivity matrix, shape (2, num_edges)
    """
    num_wells = coords.shape[0]
    dist_matrix = distance_matrix(coords, coords)

    # Find pairs within threshold (excluding diagonal)
    i_idx, j_idx = np.where((dist_matrix < threshold) & (dist_matrix > 0))
    edges = np.array([i_idx, j_idx], dtype=np.int64)

    return edges


def bipartite_connectivity(
    coords_source: np.ndarray,
    coords_target: np.ndarray,
    mode: str = 'full',
    k: int = 5,
    max_distance: Optional[float] = None
) -> np.ndarray:
    """
    Build bipartite graph connectivity (e.g., Injector-to-Producer).

    Args:
        coords_source: Source node coordinates (e.g., injectors), shape (num_source, 2)
        coords_target: Target node coordinates (e.g., producers), shape (num_target, 2)
        mode: 'full' (all-to-all), 'knearest' (k nearest targets per source)
        k: Number of nearest neighbors (for knearest mode)
        max_distance: Maximum connection distance

    Returns:
        edge_index: Connectivity matrix, shape (2, num_edges)
                   edge_index[0, :] = source indices (0 to num_source-1)
                   edge_index[1, :] = target indices (0 to num_target-1)
    """
    num_source = coords_source.shape[0]
    num_target = coords_target.shape[0]

    if mode == 'full':
        # Full bipartite graph (all injectors connect to all producers)
        src_idx = np.repeat(np.arange(num_source), num_target)
        tgt_idx = np.tile(np.arange(num_target), num_source)
        edges = np.array([src_idx, tgt_idx], dtype=np.int64)

        # Apply distance threshold if specified
        if max_distance is not None:
            dist_matrix = distance_matrix(coords_source, coords_target)
            valid = dist_matrix[src_idx, tgt_idx] <= max_distance
            edges = edges[:, valid]

        return edges

    elif mode == 'knearest':
        # k-nearest targets for each source
        k = min(k, num_target)
        dist_matrix = distance_matrix(coords_source, coords_target)

        edges = []
        for i in range(num_source):
            # Get k nearest target nodes
            nearest = np.argsort(dist_matrix[i, :])[:k]

            for j in nearest:
                if max_distance is None or dist_matrix[i, j] <= max_distance:
                    edges.append((i, j))

        edges = np.array(edges, dtype=np.int64).T
        return edges

    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'full' or 'knearest'.")


def build_well_graphs(
    producer_coords: np.ndarray,
    injector_coords: np.ndarray,
    p2p_mode: str = 'voronoi',
    i2p_mode: str = 'full',
    k_p2p: int = 5,
    k_i2p: int = 5,
    max_distance_p2p: Optional[float] = None,
    max_distance_i2p: Optional[float] = None
) -> Dict[str, np.ndarray]:
    """
    Build both Producer-Producer and Injector-Producer graphs.

    Args:
        producer_coords: Producer coordinates, shape (num_prod, 2)
        injector_coords: Injector coordinates, shape (num_inj, 2)
        p2p_mode: 'voronoi', 'knearest', or 'distance'
        i2p_mode: 'full' or 'knearest'
        k_p2p: Number of neighbors for P2P knearest
        k_i2p: Number of nearest producers per injector
        max_distance_p2p: Max distance for P2P connections
        max_distance_i2p: Max distance for I2P connections

    Returns:
        Dictionary with keys:
            - 'edge_index_p2p': Producer-Producer edges, shape (2, num_edges_p2p)
            - 'edge_index_i2p': Injector-Producer edges, shape (2, num_edges_i2p)
            - 'num_producers': Number of producer wells
            - 'num_injectors': Number of injector wells
    """
    num_producers = producer_coords.shape[0]
    num_injectors = injector_coords.shape[0]

    # Producer-to-Producer graph
    if p2p_mode == 'voronoi':
        edge_index_p2p = compute_voronoi_connectivity(producer_coords)

        # Apply distance threshold if specified
        if max_distance_p2p is not None:
            dist_matrix = distance_matrix(producer_coords, producer_coords)
            src, tgt = edge_index_p2p
            valid = dist_matrix[src, tgt] <= max_distance_p2p
            edge_index_p2p = edge_index_p2p[:, valid]

    elif p2p_mode == 'knearest':
        edge_index_p2p = knearest_connectivity(
            producer_coords, k=k_p2p, max_distance=max_distance_p2p
        )
    elif p2p_mode == 'distance':
        if max_distance_p2p is None:
            raise ValueError("max_distance_p2p must be specified for 'distance' mode")
        edge_index_p2p = distance_threshold_connectivity(
            producer_coords, threshold=max_distance_p2p
        )
    else:
        raise ValueError(f"Unknown p2p_mode: {p2p_mode}")

    # Injector-to-Producer bipartite graph
    edge_index_i2p = bipartite_connectivity(
        injector_coords, producer_coords,
        mode=i2p_mode, k=k_i2p, max_distance=max_distance_i2p
    )

    return {
        'edge_index_p2p': edge_index_p2p,
        'edge_index_i2p': edge_index_i2p,
        'num_producers': num_producers,
        'num_injectors': num_injectors,
        'num_edges_p2p': edge_index_p2p.shape[1],
        'num_edges_i2p': edge_index_i2p.shape[1],
    }


def visualize_graphs(
    producer_coords: np.ndarray,
    injector_coords: np.ndarray,
    edge_index_p2p: np.ndarray,
    edge_index_i2p: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Visualize well connectivity graphs.

    Args:
        producer_coords: Producer coordinates, shape (num_prod, 2)
        injector_coords: Injector coordinates, shape (num_inj, 2)
        edge_index_p2p: Producer-Producer edges
        edge_index_i2p: Injector-Producer edges (rows are inj indices, cols are prod indices)
        save_path: Path to save figure (None = display only)
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Plot 1: Producer-to-Producer
    ax = axes[0]
    ax.scatter(producer_coords[:, 0], producer_coords[:, 1],
              c='red', s=200, marker='o', label='Producers', zorder=3)

    # Draw P2P edges
    for i in range(edge_index_p2p.shape[1]):
        src, tgt = edge_index_p2p[:, i]
        ax.plot([producer_coords[src, 0], producer_coords[tgt, 0]],
               [producer_coords[src, 1], producer_coords[tgt, 1]],
               'k-', alpha=0.3, linewidth=1, zorder=1)

    ax.set_title(f'Producer-to-Producer Graph ({edge_index_p2p.shape[1]} edges)',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('X (ft)')
    ax.set_ylabel('Y (ft)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Plot 2: Injector-to-Producer
    ax = axes[1]
    ax.scatter(producer_coords[:, 0], producer_coords[:, 1],
              c='red', s=200, marker='o', label='Producers', zorder=3)
    ax.scatter(injector_coords[:, 0], injector_coords[:, 1],
              c='blue', s=200, marker='^', label='Injectors', zorder=3)

    # Draw I2P edges
    for i in range(edge_index_i2p.shape[1]):
        inj_idx, prod_idx = edge_index_i2p[:, i]
        ax.plot([injector_coords[inj_idx, 0], producer_coords[prod_idx, 0]],
               [injector_coords[inj_idx, 1], producer_coords[prod_idx, 1]],
               'g-', alpha=0.2, linewidth=0.8, zorder=1)

    ax.set_title(f'Injector-to-Producer Graph ({edge_index_i2p.shape[1]} edges)',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('X (ft)')
    ax.set_ylabel('Y (ft)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Graph visualization saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    # Test with example well configuration
    np.random.seed(42)

    # Simulate 10 producers and 5 injectors in a 5000x5000 ft domain
    producer_coords = np.random.uniform(500, 4500, size=(10, 2))
    injector_coords = np.random.uniform(500, 4500, size=(5, 2))

    # Build graphs
    graphs = build_well_graphs(
        producer_coords, injector_coords,
        p2p_mode='voronoi',
        i2p_mode='knearest',
        k_i2p=5
    )

    print("Graph Construction Test:")
    print(f"  Producers: {graphs['num_producers']}")
    print(f"  Injectors: {graphs['num_injectors']}")
    print(f"  P2P edges: {graphs['num_edges_p2p']}")
    print(f"  I2P edges: {graphs['num_edges_i2p']}")
    print(f"  Avg P2P connectivity: {graphs['num_edges_p2p'] / graphs['num_producers']:.1f} edges/well")

    # Visualize
    visualize_graphs(
        producer_coords, injector_coords,
        graphs['edge_index_p2p'], graphs['edge_index_i2p']
    )
