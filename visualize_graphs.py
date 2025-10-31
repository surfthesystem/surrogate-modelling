"""Visualize the GNN graph structures (P2P and I2P edges)."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load graph data
graph_data = np.load('ml/data/preprocessed/graph_data.npz')

p2p_edges = graph_data['edge_index_p2p']  # (2, num_p2p_edges)
i2p_edges = graph_data['edge_index_i2p']  # (2, num_i2p_edges)
producer_locs = graph_data['producer_coords']  # (10, 2)
injector_locs = graph_data['injector_coords']  # (5, 2)

print(f"P2P edges: {p2p_edges.shape}")
print(f"I2P edges: {i2p_edges.shape}")
print(f"Producer locations: {producer_locs.shape}")
print(f"Injector locations: {injector_locs.shape}")

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Plot 1: Producer-to-Producer Graph (Voronoi-based)
ax1.set_title('Producer-to-Producer Graph (Voronoi)', fontsize=14, fontweight='bold')

# Plot P2P edges
for i in range(p2p_edges.shape[1]):
    src_idx = p2p_edges[0, i]
    dst_idx = p2p_edges[1, i]

    x_coords = [producer_locs[src_idx, 0], producer_locs[dst_idx, 0]]
    y_coords = [producer_locs[src_idx, 1], producer_locs[dst_idx, 1]]

    ax1.plot(x_coords, y_coords, 'b-', alpha=0.3, linewidth=1)

# Plot producer nodes
ax1.scatter(producer_locs[:, 0], producer_locs[:, 1],
           c='blue', s=200, marker='o', edgecolors='darkblue', linewidths=2,
           label='Producers', zorder=10)

# Add node labels
for i, (x, y) in enumerate(producer_locs):
    ax1.text(x, y, f'P{i+1}', fontsize=9, ha='center', va='center',
            color='white', fontweight='bold', zorder=11)

ax1.set_xlabel('X (ft)', fontsize=12)
ax1.set_ylabel('Y (ft)', fontsize=12)
ax1.legend(fontsize=11, loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')
ax1.text(0.02, 0.98, f'{p2p_edges.shape[1]} edges',
        transform=ax1.transAxes, fontsize=11, va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 2: Injector-to-Producer Graph (Bipartite)
ax2.set_title('Injector-to-Producer Graph (Bipartite)', fontsize=14, fontweight='bold')

# Plot I2P edges
for i in range(i2p_edges.shape[1]):
    inj_idx = i2p_edges[0, i]  # Injector index (0-4)
    prod_idx = i2p_edges[1, i]  # Producer index (0-9)

    x_coords = [injector_locs[inj_idx, 0], producer_locs[prod_idx, 0]]
    y_coords = [injector_locs[inj_idx, 1], producer_locs[prod_idx, 1]]

    ax2.plot(x_coords, y_coords, 'g-', alpha=0.3, linewidth=1)

# Plot producer nodes
ax2.scatter(producer_locs[:, 0], producer_locs[:, 1],
           c='blue', s=200, marker='o', edgecolors='darkblue', linewidths=2,
           label='Producers', zorder=10)

# Plot injector nodes
ax2.scatter(injector_locs[:, 0], injector_locs[:, 1],
           c='red', s=250, marker='s', edgecolors='darkred', linewidths=2,
           label='Injectors', zorder=10)

# Add producer labels
for i, (x, y) in enumerate(producer_locs):
    ax2.text(x, y, f'P{i+1}', fontsize=9, ha='center', va='center',
            color='white', fontweight='bold', zorder=11)

# Add injector labels
for i, (x, y) in enumerate(injector_locs):
    ax2.text(x, y, f'I{i+1}', fontsize=9, ha='center', va='center',
            color='white', fontweight='bold', zorder=11)

ax2.set_xlabel('X (ft)', fontsize=12)
ax2.set_ylabel('Y (ft)', fontsize=12)
ax2.legend(fontsize=11, loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')
ax2.text(0.02, 0.98, f'{i2p_edges.shape[1]} edges',
        transform=ax2.transAxes, fontsize=11, va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

# Save figure
output_path = Path('results/evaluation/normalized_run/graph_structures.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Graph visualization saved to: {output_path}")

plt.close()
