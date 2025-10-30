"""
Full GNN-LSTM Surrogate Model for reservoir simulation.

Combines spatial encoding (GNN) with temporal dynamics (LSTM) to predict
well production rates given controls and reservoir properties.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gnn import DualGraphGNN
from .lstm import WellAwareLSTM


class RateDecoder(nn.Module):
    """
    Decoder network that maps LSTM output to well production rates.

    Outputs oil and water rates for each producer well.
    """

    def __init__(
        self,
        lstm_hidden_dim: int = 256,
        num_producers: int = 10,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        """
        Args:
            lstm_hidden_dim: Dimension of LSTM output
            num_producers: Number of producer wells
            hidden_dim: Hidden dimension for decoder MLP
            dropout: Dropout rate
        """
        super(RateDecoder, self).__init__()

        self.num_producers = num_producers

        # Decoder MLP
        self.decoder = nn.Sequential(
            nn.Linear(lstm_hidden_dim, hidden_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, num_producers * 2),  # 2 rates per well (oil, water)
        )

    def forward(self, lstm_output):
        """
        Decode LSTM output to production rates.

        Args:
            lstm_output: LSTM output, shape (batch, T, lstm_hidden_dim)

        Returns:
            Dict with:
                - oil_rates: (batch, T, num_producers)
                - water_rates: (batch, T, num_producers)
        """
        batch_size, T, _ = lstm_output.shape

        # Decode to rates
        rates = self.decoder(lstm_output)  # (batch, T, num_producers * 2)

        # Reshape to (batch, T, num_producers, 2)
        rates = rates.view(batch_size, T, self.num_producers, 2)

        # Split into oil and water rates
        oil_rates = rates[..., 0]  # (batch, T, num_producers)
        water_rates = rates[..., 1]  # (batch, T, num_producers)

        # Apply softplus to ensure positive rates
        oil_rates = F.softplus(oil_rates)
        water_rates = F.softplus(water_rates)

        return {
            'oil_rates': oil_rates,
            'water_rates': water_rates,
        }


class GNN_LSTM_Surrogate(nn.Module):
    """
    Full GNN-LSTM surrogate model for reservoir simulation.

    Architecture:
        1. Spatial encoding: DualGraphGNN processes well connectivity at each timestep
        2. Temporal encoding: LSTM models evolution of well states
        3. Rate prediction: Decoder maps LSTM output to oil/water rates

    Inputs:
        - Well controls (BHP, injection rates)
        - Reservoir properties (permeability, porosity)
        - Well connectivity (graphs)
        - Edge features (distance, perm, pressure, etc.)

    Outputs:
        - Oil production rates per well
        - Water production rates per well
    """

    def __init__(
        self,
        # GNN parameters
        producer_node_dim: int = 10,
        injector_node_dim: int = 8,
        edge_dim: int = 10,
        gnn_hidden_dim: int = 128,
        gnn_num_layers: int = 3,
        gnn_dropout: float = 0.2,
        # LSTM parameters
        lstm_hidden_dim: int = 256,
        lstm_num_layers: int = 2,
        lstm_dropout: float = 0.2,
        # Decoder parameters
        num_producers: int = 10,
        decoder_hidden_dim: int = 64,
        decoder_dropout: float = 0.1,
    ):
        """
        Initialize full surrogate model.

        Args:
            See individual module docstrings for parameter descriptions.
        """
        super(GNN_LSTM_Surrogate, self).__init__()

        self.producer_node_dim = producer_node_dim
        self.injector_node_dim = injector_node_dim
        self.edge_dim = edge_dim
        self.num_producers = num_producers

        # === Spatial Encoder (GNN) ===
        self.gnn = DualGraphGNN(
            producer_node_dim=producer_node_dim,
            injector_node_dim=injector_node_dim,
            edge_dim=edge_dim,
            hidden_dim=gnn_hidden_dim,
            num_layers=gnn_num_layers,
            dropout=gnn_dropout,
        )

        # === Temporal Encoder (LSTM) ===
        self.lstm = WellAwareLSTM(
            num_wells=num_producers,
            well_embedding_dim=gnn_hidden_dim,
            hidden_dim=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            dropout=lstm_dropout,
        )

        # === Rate Decoder ===
        self.decoder = RateDecoder(
            lstm_hidden_dim=lstm_hidden_dim,
            num_producers=num_producers,
            hidden_dim=decoder_hidden_dim,
            dropout=decoder_dropout,
        )

        self.gnn_hidden_dim = gnn_hidden_dim
        self.lstm_hidden_dim = lstm_hidden_dim

    def forward(self, batch_data):
        """
        Forward pass through full surrogate model.

        Args:
            batch_data: Dict with keys:
                - producer_features: (batch, T, num_prod, prod_dim)
                - injector_features: (batch, T, num_inj, inj_dim)
                - edge_features_p2p: (batch, T, num_edges_p2p, edge_dim)
                - edge_features_i2p: (batch, T, num_edges_i2p, edge_dim)
                - edge_index_p2p: (2, num_edges_p2p)
                - edge_index_i2p: (2, num_edges_i2p)

        Returns:
            Dict with:
                - oil_rates: Predicted oil rates, (batch, T, num_producers)
                - water_rates: Predicted water rates, (batch, T, num_producers)
                - producer_embeddings: GNN embeddings (optional, for visualization)
        """
        batch_size = batch_data['producer_features'].shape[0]
        T = batch_data['producer_features'].shape[1]

        # === 1. Spatial Encoding with GNN (per timestep) ===
        # Process each timestep independently with GNN
        producer_embeddings_list = []

        for t in range(T):
            # Extract features at timestep t
            batch_t = {
                'producer_features': batch_data['producer_features'][:, t, :, :],  # (batch, num_prod, dim)
                'injector_features': batch_data['injector_features'][:, t, :, :],  # (batch, num_inj, dim)
                'edge_features_p2p': batch_data['edge_features_p2p'][:, t, :, :],  # (batch, edges, dim)
                'edge_features_i2p': batch_data['edge_features_i2p'][:, t, :, :],  # (batch, edges, dim)
                'edge_index_p2p': batch_data['edge_index_p2p'],  # Static
                'edge_index_i2p': batch_data['edge_index_i2p'],  # Static
            }

            # GNN forward: encode well connectivity
            producer_emb_t = self.gnn(batch_t)  # (batch, num_prod, gnn_hidden_dim)
            producer_embeddings_list.append(producer_emb_t)

        # Stack into sequence
        producer_embeddings = torch.stack(producer_embeddings_list, dim=1)  # (batch, T, num_prod, gnn_hidden_dim)

        # === 2. Temporal Encoding with LSTM ===
        lstm_output, _ = self.lstm(producer_embeddings)  # (batch, T, lstm_hidden_dim)

        # === 3. Decode to Rates ===
        rates = self.decoder(lstm_output)  # Dict with oil_rates, water_rates

        # Add producer embeddings for visualization/analysis
        rates['producer_embeddings'] = producer_embeddings

        return rates

    def predict(self, batch_data):
        """
        Prediction mode (same as forward, but with explicit name for clarity).
        """
        return self.forward(batch_data)

    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test full surrogate model
    print("Testing GNN-LSTM Surrogate Model...")

    torch.manual_seed(42)

    batch_size = 4
    T = 61  # 61 timesteps
    num_prod = 10
    num_inj = 5
    num_edges_p2p = 30
    num_edges_i2p = 50

    # Create dummy batch data
    batch_data = {
        'producer_features': torch.randn(batch_size, T, num_prod, 10),
        'injector_features': torch.randn(batch_size, T, num_inj, 8),
        'edge_features_p2p': torch.randn(batch_size, T, num_edges_p2p, 10),
        'edge_features_i2p': torch.randn(batch_size, T, num_edges_i2p, 10),
        'edge_index_p2p': torch.randint(0, num_prod, (2, num_edges_p2p)),
        'edge_index_i2p': torch.cat([
            torch.randint(0, num_inj, (1, num_edges_i2p)),
            torch.randint(num_inj, num_inj + num_prod, (1, num_edges_i2p))
        ], dim=0),
    }

    # Initialize model
    print("\n1. Initializing model...")
    model = GNN_LSTM_Surrogate(
        producer_node_dim=10,
        injector_node_dim=8,
        edge_dim=10,
        gnn_hidden_dim=128,
        gnn_num_layers=3,
        lstm_hidden_dim=256,
        lstm_num_layers=2,
        num_producers=10,
    )

    num_params = model.count_parameters()
    print(f"   Total parameters: {num_params:,} ({num_params/1e6:.2f}M)")

    # Forward pass
    print("\n2. Testing forward pass...")
    outputs = model(batch_data)

    print(f"   Oil rates shape: {outputs['oil_rates'].shape}")  # (batch, T, num_prod)
    print(f"   Water rates shape: {outputs['water_rates'].shape}")
    print(f"   Producer embeddings shape: {outputs['producer_embeddings'].shape}")

    # Check output ranges (should be positive due to softplus)
    print(f"\n3. Output statistics:")
    print(f"   Oil rates: min={outputs['oil_rates'].min():.4f}, max={outputs['oil_rates'].max():.4f}")
    print(f"   Water rates: min={outputs['water_rates'].min():.4f}, max={outputs['water_rates'].max():.4f}")

    # Test gradient flow
    print("\n4. Testing gradient flow...")
    loss = outputs['oil_rates'].sum() + outputs['water_rates'].sum()
    loss.backward()

    # Check if gradients exist
    has_grad = any(p.grad is not None for p in model.parameters())
    print(f"   Gradients computed: {has_grad}")

    # Test prediction mode
    print("\n5. Testing prediction mode...")
    with torch.no_grad():
        predictions = model.predict(batch_data)
        print(f"   Prediction oil rates shape: {predictions['oil_rates'].shape}")

    print("\n✓ Full surrogate model tests passed!")
