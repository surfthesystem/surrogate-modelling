"""
Enhanced Graph Neural Network for well connectivity.

Uses message passing with edge features to encode spatial relationships
between producers and injectors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from typing import Optional


class EdgeConv(MessagePassing):
    """
    Graph convolution layer with edge features.

    Implements message passing where messages depend on both node and edge features.
    """

    def __init__(self, node_dim: int, edge_dim: int, out_dim: int):
        """
        Args:
            node_dim: Dimension of node features
            edge_dim: Dimension of edge features
            out_dim: Output dimension
        """
        super(EdgeConv, self).__init__(aggr='add')  # Sum aggregation

        # Message network: combines source node, target node, and edge
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, out_dim),
            nn.PReLU(),
            nn.Linear(out_dim, out_dim),
        )

        # Update network: combines aggregated messages with self-features
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim + out_dim, out_dim),
            nn.PReLU(),
        )

    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x: Node features, shape (num_nodes, node_dim)
            edge_index: Edge connectivity, shape (2, num_edges)
            edge_attr: Edge features, shape (num_edges, edge_dim)

        Returns:
            out: Updated node features, shape (num_nodes, out_dim)
        """
        # Propagate messages
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        # Update with self-connection
        out = self.update_mlp(torch.cat([x, out], dim=-1))

        return out

    def message(self, x_i, x_j, edge_attr):
        """
        Construct messages for each edge.

        Args:
            x_i: Target node features, shape (num_edges, node_dim)
            x_j: Source node features, shape (num_edges, node_dim)
            edge_attr: Edge features, shape (num_edges, edge_dim)

        Returns:
            messages: Shape (num_edges, out_dim)
        """
        # Concatenate source, target, and edge features
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)

        # Transform through MLP
        messages = self.message_mlp(msg_input)

        return messages


class EnhancedGNN(nn.Module):
    """
    Enhanced GNN with multi-layer message passing and edge features.

    Encodes spatial well connectivity using permeability, pressure gradients,
    and geometric features.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
    ):
        """
        Args:
            node_dim: Dimension of input node features
            edge_dim: Dimension of edge features
            hidden_dim: Hidden layer dimension
            num_layers: Number of message passing layers
            dropout: Dropout rate for regularization
        """
        super(EnhancedGNN, self).__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Node embedding: map variable-size input to hidden_dim
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, 64),
            nn.PReLU(),
            nn.Linear(64, hidden_dim),
        )

        # Edge embedding: map edge features to hidden_dim
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, 64),
            nn.PReLU(),
            nn.Linear(64, hidden_dim),
        )

        # Message passing layers
        self.conv_layers = nn.ModuleList([
            EdgeConv(hidden_dim, hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, node_features, edge_features, edge_index):
        """
        Forward pass through GNN.

        Args:
            node_features: Node features, shape (num_nodes, node_dim) or (batch, num_nodes, node_dim)
            edge_features: Edge features, shape (num_edges, edge_dim) or (batch, num_edges, edge_dim)
            edge_index: Edge connectivity, shape (2, num_edges)

        Returns:
            node_embeddings: shape (num_nodes, hidden_dim) or (batch, num_nodes, hidden_dim)
        """
        # Handle batched input
        if node_features.dim() == 3:
            batch_size, num_nodes, _ = node_features.shape
            num_edges = edge_features.shape[1]

            # Process each sample in batch independently
            outputs = []
            for b in range(batch_size):
                out_b = self._forward_single(
                    node_features[b], edge_features[b], edge_index
                )
                outputs.append(out_b)

            return torch.stack(outputs, dim=0)  # (batch, num_nodes, hidden_dim)

        else:
            # Single sample (no batch dimension)
            return self._forward_single(node_features, edge_features, edge_index)

    def _forward_single(self, node_features, edge_features, edge_index):
        """Forward pass for a single sample"""
        # Encode nodes and edges to hidden dimension
        h_nodes = self.node_encoder(node_features)  # (num_nodes, hidden_dim)
        h_edges = self.edge_encoder(edge_features)  # (num_edges, hidden_dim)

        # Message passing with residual connections
        for conv in self.conv_layers:
            h_nodes_new = conv(h_nodes, edge_index, h_edges)
            h_nodes = h_nodes + h_nodes_new  # Residual
            h_nodes = self.dropout(h_nodes)

        # Output projection
        output = self.output_proj(h_nodes)

        return output  # (num_nodes, hidden_dim)


class DualGraphGNN(nn.Module):
    """
    Dual-graph GNN: separate processing for P2P and I2P graphs.

    Processes Producer-Producer and Injector-Producer graphs separately,
    then fuses their outputs.
    """

    def __init__(
        self,
        producer_node_dim: int = 10,
        injector_node_dim: int = 8,
        edge_dim: int = 10,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
    ):
        """
        Args:
            producer_node_dim: Producer node feature dimension
            injector_node_dim: Injector node feature dimension
            edge_dim: Edge feature dimension
            hidden_dim: Hidden dimension for GNN
            num_layers: Number of message passing layers
            dropout: Dropout rate
        """
        super(DualGraphGNN, self).__init__()

        # Producer-to-Producer GNN
        self.gnn_p2p = EnhancedGNN(
            node_dim=producer_node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Injector-to-Producer GNN (operates on combined node set)
        # For I2P, we need to handle both injector and producer nodes
        # We'll use a unified node dimension by padding
        max_node_dim = max(producer_node_dim, injector_node_dim)

        self.gnn_i2p = EnhancedGNN(
            node_dim=max_node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Feature fusion: combine P2P and I2P embeddings
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.PReLU(),
        )

        self.producer_node_dim = producer_node_dim
        self.injector_node_dim = injector_node_dim
        self.max_node_dim = max_node_dim

    def forward(self, batch_data):
        """
        Forward pass through dual-graph GNN.

        Args:
            batch_data: Dict with keys:
                - producer_features: (batch, num_prod, prod_dim)
                - injector_features: (batch, num_inj, inj_dim)
                - edge_features_p2p: (batch, num_edges_p2p, edge_dim)
                - edge_features_i2p: (batch, num_edges_i2p, edge_dim)
                - edge_index_p2p: (2, num_edges_p2p)
                - edge_index_i2p: (2, num_edges_i2p)

        Returns:
            producer_embeddings: (batch, num_prod, hidden_dim)
        """
        producer_feats = batch_data['producer_features']
        injector_feats = batch_data['injector_features']

        batch_size = producer_feats.shape[0]
        num_prod = producer_feats.shape[1]
        num_inj = injector_feats.shape[1]

        # === Producer-to-Producer graph ===
        h_p2p = self.gnn_p2p(
            producer_feats,
            batch_data['edge_features_p2p'],
            batch_data['edge_index_p2p']
        )  # (batch, num_prod, hidden_dim)

        # === Injector-to-Producer graph ===
        # Pad features to max dimension
        prod_padded = F.pad(producer_feats, (0, self.max_node_dim - self.producer_node_dim))
        inj_padded = F.pad(injector_feats, (0, self.max_node_dim - self.injector_node_dim))

        # Concatenate nodes: [injectors, producers]
        # This matches edge_index_i2p where source indices are injectors (0-4)
        # and target indices are producers (5-14) in a combined node set
        combined_nodes = torch.cat([inj_padded, prod_padded], dim=1)  # (batch, num_inj+num_prod, max_dim)

        # Forward through I2P GNN
        h_combined = self.gnn_i2p(
            combined_nodes,
            batch_data['edge_features_i2p'],
            batch_data['edge_index_i2p']
        )  # (batch, num_inj+num_prod, hidden_dim)

        # Extract producer embeddings (last num_prod nodes)
        h_i2p = h_combined[:, num_inj:, :]  # (batch, num_prod, hidden_dim)

        # === Fuse P2P and I2P embeddings ===
        h_fused = self.fusion(torch.cat([h_p2p, h_i2p], dim=-1))  # (batch, num_prod, hidden_dim)

        return h_fused


if __name__ == "__main__":
    # Test GNN modules
    print("Testing Enhanced GNN...")

    torch.manual_seed(42)

    batch_size = 4
    num_prod = 10
    num_inj = 5
    num_edges_p2p = 30
    num_edges_i2p = 50

    # Create dummy data
    batch_data = {
        'producer_features': torch.randn(batch_size, num_prod, 10),
        'injector_features': torch.randn(batch_size, num_inj, 8),
        'edge_features_p2p': torch.randn(batch_size, num_edges_p2p, 10),
        'edge_features_i2p': torch.randn(batch_size, num_edges_i2p, 10),
        'edge_index_p2p': torch.randint(0, num_prod, (2, num_edges_p2p)),
        'edge_index_i2p': torch.cat([
            torch.randint(0, num_inj, (1, num_edges_i2p)),  # Source: injectors (0-4)
            torch.randint(num_inj, num_inj+num_prod, (1, num_edges_i2p))  # Target: producers (5-14)
        ], dim=0),
    }

    # Test single GNN
    print("\n1. Testing single EnhancedGNN...")
    gnn = EnhancedGNN(node_dim=10, edge_dim=10, hidden_dim=128, num_layers=3)
    out_single = gnn(
        batch_data['producer_features'],
        batch_data['edge_features_p2p'],
        batch_data['edge_index_p2p']
    )
    print(f"   Output shape: {out_single.shape}")  # Should be (batch, num_prod, 128)

    # Test dual-graph GNN
    print("\n2. Testing DualGraphGNN...")
    dual_gnn = DualGraphGNN(
        producer_node_dim=10,
        injector_node_dim=8,
        edge_dim=10,
        hidden_dim=128,
        num_layers=3
    )

    producer_embeddings = dual_gnn(batch_data)
    print(f"   Producer embeddings shape: {producer_embeddings.shape}")  # (batch, num_prod, 128)

    # Test gradient flow
    print("\n3. Testing gradient flow...")
    loss = producer_embeddings.sum()
    loss.backward()
    print(f"   ✓ Gradients computed successfully")

    # Count parameters
    num_params = sum(p.numel() for p in dual_gnn.parameters())
    print(f"\n4. Model size: {num_params:,} parameters ({num_params/1e6:.2f}M)")

    print("\n✓ GNN tests passed!")
