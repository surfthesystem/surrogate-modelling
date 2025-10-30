"""
Temporal LSTM module for capturing production dynamics over time.

Takes GNN-encoded well embeddings and models their temporal evolution.
"""

import torch
import torch.nn as nn


class TemporalLSTM(nn.Module):
    """
    LSTM module for modeling temporal evolution of well production.

    Processes sequences of well embeddings to capture production dynamics,
    depletion trends, and time-varying interference effects.
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        """
        Args:
            input_dim: Dimension of input features (well embeddings from GNN)
            hidden_dim: LSTM hidden state dimension
            num_layers: Number of stacked LSTM layers
            dropout: Dropout rate between LSTM layers
            bidirectional: Use bidirectional LSTM (not recommended for forecasting)
        """
        super(TemporalLSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Output dimension depends on bidirectionality
        self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim

    def forward(self, x, hidden_state=None):
        """
        Forward pass through LSTM.

        Args:
            x: Input sequence, shape (batch, seq_len, input_dim)
            hidden_state: Optional initial hidden state tuple (h_0, c_0)
                         h_0: (num_layers * num_directions, batch, hidden_dim)
                         c_0: (num_layers * num_directions, batch, hidden_dim)

        Returns:
            output: LSTM output for all timesteps, shape (batch, seq_len, hidden_dim)
            (h_n, c_n): Final hidden and cell states
        """
        # LSTM forward
        output, (h_n, c_n) = self.lstm(x, hidden_state)

        return output, (h_n, c_n)

    def init_hidden(self, batch_size, device='cpu'):
        """
        Initialize hidden state to zeros.

        Args:
            batch_size: Batch size
            device: Device ('cpu' or 'cuda')

        Returns:
            (h_0, c_0): Initial hidden and cell states
        """
        num_directions = 2 if self.bidirectional else 1

        h_0 = torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_dim,
            device=device
        )

        c_0 = torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_dim,
            device=device
        )

        return (h_0, c_0)


class WellAwareLSTM(nn.Module):
    """
    Well-aware LSTM that processes each well's embedding sequence.

    Instead of flattening all wells into a single sequence, this maintains
    well identities and processes them in parallel.
    """

    def __init__(
        self,
        num_wells: int,
        well_embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        """
        Args:
            num_wells: Number of wells (producers)
            well_embedding_dim: Dimension of well embeddings from GNN
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(WellAwareLSTM, self).__init__()

        self.num_wells = num_wells
        self.well_embedding_dim = well_embedding_dim
        self.hidden_dim = hidden_dim

        # Flatten well embeddings for LSTM input
        # Input: (batch, T, num_wells, well_dim) -> (batch, T, num_wells * well_dim)
        self.input_dim = num_wells * well_embedding_dim

        # LSTM
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.output_dim = hidden_dim

    def forward(self, well_embeddings, hidden_state=None):
        """
        Forward pass through well-aware LSTM.

        Args:
            well_embeddings: Well embeddings from GNN,
                            shape (batch, T, num_wells, well_embedding_dim)
            hidden_state: Optional initial hidden state

        Returns:
            output: LSTM output, shape (batch, T, hidden_dim)
            (h_n, c_n): Final hidden and cell states
        """
        batch_size, T, num_wells, well_dim = well_embeddings.shape

        # Flatten wells into input features
        x = well_embeddings.view(batch_size, T, num_wells * well_dim)

        # LSTM forward
        output, (h_n, c_n) = self.lstm(x, hidden_state)

        return output, (h_n, c_n)


class AttentionLSTM(nn.Module):
    """
    LSTM with attention mechanism over well embeddings.

    Uses attention to dynamically weight well contributions at each timestep.
    """

    def __init__(
        self,
        num_wells: int,
        well_embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        """
        Args:
            num_wells: Number of wells
            well_embedding_dim: Dimension of well embeddings
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(AttentionLSTM, self).__init__()

        self.num_wells = num_wells
        self.well_embedding_dim = well_embedding_dim
        self.hidden_dim = hidden_dim

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(well_embedding_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # LSTM processes attended embeddings
        self.lstm = nn.LSTM(
            input_size=well_embedding_dim,  # Attended embedding
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.output_dim = hidden_dim

    def forward(self, well_embeddings, hidden_state=None):
        """
        Forward pass with attention.

        Args:
            well_embeddings: shape (batch, T, num_wells, well_embedding_dim)
            hidden_state: Optional initial LSTM state

        Returns:
            output: LSTM output, shape (batch, T, hidden_dim)
            (h_n, c_n): Final states
            attention_weights: shape (batch, T, num_wells) for visualization
        """
        batch_size, T, num_wells, well_dim = well_embeddings.shape

        # Compute attention weights for each timestep
        # Reshape to (batch * T, num_wells, well_dim)
        emb_reshaped = well_embeddings.view(batch_size * T, num_wells, well_dim)

        # Attention scores: (batch * T, num_wells, 1)
        attn_scores = self.attention(emb_reshaped)

        # Softmax over wells: (batch * T, num_wells, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)

        # Weighted sum of embeddings: (batch * T, well_dim)
        attended_emb = (emb_reshaped * attn_weights).sum(dim=1)

        # Reshape back to sequence: (batch, T, well_dim)
        attended_emb = attended_emb.view(batch_size, T, well_dim)

        # LSTM forward
        output, (h_n, c_n) = self.lstm(attended_emb, hidden_state)

        # Reshape attention weights for output: (batch, T, num_wells)
        attn_weights_out = attn_weights.view(batch_size, T, num_wells)

        return output, (h_n, c_n), attn_weights_out


if __name__ == "__main__":
    # Test LSTM modules
    print("Testing LSTM modules...")

    torch.manual_seed(42)

    batch_size = 4
    seq_len = 61  # 61 timesteps (6 months)
    num_wells = 10
    well_embedding_dim = 128

    # Create dummy well embeddings (output from GNN)
    well_embeddings = torch.randn(batch_size, seq_len, num_wells, well_embedding_dim)

    # === Test 1: Basic TemporalLSTM ===
    print("\n1. Testing TemporalLSTM...")

    # Flatten for basic LSTM
    x_flat = well_embeddings.view(batch_size, seq_len, num_wells * well_embedding_dim)

    lstm = TemporalLSTM(
        input_dim=num_wells * well_embedding_dim,
        hidden_dim=256,
        num_layers=2,
        dropout=0.2
    )

    output, (h_n, c_n) = lstm(x_flat)
    print(f"   Input shape: {x_flat.shape}")
    print(f"   Output shape: {output.shape}")  # (batch, seq_len, 256)
    print(f"   Hidden state shape: {h_n.shape}")  # (2, batch, 256)

    # === Test 2: WellAwareLSTM ===
    print("\n2. Testing WellAwareLSTM...")

    well_lstm = WellAwareLSTM(
        num_wells=num_wells,
        well_embedding_dim=well_embedding_dim,
        hidden_dim=256,
        num_layers=2,
        dropout=0.2
    )

    output2, (h_n2, c_n2) = well_lstm(well_embeddings)
    print(f"   Input shape: {well_embeddings.shape}")
    print(f"   Output shape: {output2.shape}")  # (batch, seq_len, 256)

    # === Test 3: AttentionLSTM ===
    print("\n3. Testing AttentionLSTM...")

    attn_lstm = AttentionLSTM(
        num_wells=num_wells,
        well_embedding_dim=well_embedding_dim,
        hidden_dim=256,
        num_layers=2,
        dropout=0.2
    )

    output3, (h_n3, c_n3), attn_weights = attn_lstm(well_embeddings)
    print(f"   Input shape: {well_embeddings.shape}")
    print(f"   Output shape: {output3.shape}")  # (batch, seq_len, 256)
    print(f"   Attention weights shape: {attn_weights.shape}")  # (batch, seq_len, num_wells)

    # Check attention weights sum to 1
    attn_sum = attn_weights.sum(dim=2)  # Should be all 1s
    print(f"   Attention sums to 1: {torch.allclose(attn_sum, torch.ones_like(attn_sum))}")

    # === Test 4: Gradient flow ===
    print("\n4. Testing gradient flow...")
    loss = output3.sum()
    loss.backward()
    print(f"   ✓ Gradients computed successfully")

    # === Test 5: Parameter count ===
    print("\n5. Model sizes:")
    for name, model in [("TemporalLSTM", lstm), ("WellAwareLSTM", well_lstm), ("AttentionLSTM", attn_lstm)]:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"   {name}: {num_params:,} parameters ({num_params/1e6:.2f}M)")

    print("\n✓ LSTM tests passed!")
