"""
Feature normalization utilities for reservoir data.

Provides consistent scaling for pressures, rates, permeability, etc.
"""

import numpy as np
from typing import Dict, Optional, Tuple
import pickle


class FeatureNormalizer:
    """
    Flexible feature normalizer supporting multiple scaling modes.
    """

    def __init__(self, mode: str = 'minmax', eps: float = 1e-6):
        """
        Args:
            mode: 'minmax', 'standard', 'log', 'log1p', or 'robust'
            eps: Small constant for numerical stability
        """
        self.mode = mode
        self.eps = eps
        self.fitted = False

        # Statistics (filled during fit)
        self.min_val = None
        self.max_val = None
        self.mean_val = None
        self.std_val = None
        self.median_val = None
        self.q25_val = None
        self.q75_val = None

    def fit(self, data: np.ndarray) -> 'FeatureNormalizer':
        """
        Fit normalizer to training data.

        Args:
            data: Training data, any shape

        Returns:
            self (for chaining)
        """
        data_flat = data.flatten()

        if self.mode in ['minmax', 'log', 'log1p']:
            self.min_val = np.min(data_flat)
            self.max_val = np.max(data_flat)

        if self.mode == 'standard':
            self.mean_val = np.mean(data_flat)
            self.std_val = np.std(data_flat) + self.eps

        if self.mode == 'robust':
            self.median_val = np.median(data_flat)
            self.q25_val = np.percentile(data_flat, 25)
            self.q75_val = np.percentile(data_flat, 75)

        self.fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Apply normalization to data.

        Args:
            data: Data to normalize

        Returns:
            normalized_data: Normalized data, same shape as input
        """
        if not self.fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")

        if self.mode == 'minmax':
            # Scale to [0, 1]
            return (data - self.min_val) / (self.max_val - self.min_val + self.eps)

        elif self.mode == 'standard':
            # Z-score normalization
            return (data - self.mean_val) / self.std_val

        elif self.mode == 'log':
            # Log transform (assumes data > 0)
            return np.log(data + self.eps)

        elif self.mode == 'log1p':
            # Log(1+x) transform (handles zeros)
            return np.log1p(data)

        elif self.mode == 'robust':
            # Robust scaling using median and IQR
            iqr = self.q75_val - self.q25_val + self.eps
            return (data - self.median_val) / iqr

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Reverse normalization.

        Args:
            data: Normalized data

        Returns:
            original_data: De-normalized data
        """
        if not self.fitted:
            raise RuntimeError("Normalizer not fitted.")

        if self.mode == 'minmax':
            return data * (self.max_val - self.min_val + self.eps) + self.min_val

        elif self.mode == 'standard':
            return data * self.std_val + self.mean_val

        elif self.mode == 'log':
            return np.exp(data) - self.eps

        elif self.mode == 'log1p':
            return np.expm1(data)

        elif self.mode == 'robust':
            iqr = self.q75_val - self.q25_val + self.eps
            return data * iqr + self.median_val

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform in one step"""
        return self.fit(data).transform(data)

    def save(self, filepath: str):
        """Save normalizer state to pickle file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.__dict__, f)
        print(f"Normalizer saved to {filepath}")

    def load(self, filepath: str):
        """Load normalizer state from pickle file"""
        with open(filepath, 'rb') as f:
            self.__dict__.update(pickle.load(f))
        print(f"Normalizer loaded from {filepath}")
        return self


def create_normalizers(config: Optional[Dict] = None) -> Dict[str, FeatureNormalizer]:
    """
    Create normalizers for all reservoir features.

    Args:
        config: Optional configuration dict. If None, uses defaults.

    Returns:
        normalizers: Dictionary of FeatureNormalizer objects
    """
    if config is None:
        config = {
            'BHP': 'minmax',         # Producer bottom-hole pressure
            'injection_rate': 'minmax',  # Injector rate
            'permeability': 'log',    # Log-normal distribution
            'porosity': 'standard',   # Gaussian-like
            'pressure': 'minmax',     # Grid pressure
            'saturation': 'minmax',   # Water saturation [0.2, 0.8]
            'oil_rate': 'log1p',      # Log(1+rate) for stability
            'water_rate': 'log1p',    # Log(1+rate)
            'cum_oil': 'log1p',       # Cumulative production
            'cum_water': 'log1p',     # Cumulative production
            'cum_inj': 'log1p',       # Cumulative injection
            'edge_distance': 'robust',  # Edge feature: distance
            'edge_perm': 'standard',    # Edge feature: log k (already log)
            'edge_pressure': 'standard',  # Edge feature: ΔP
            'edge_saturation': 'standard',  # Edge feature: ΔSw
        }

    normalizers = {}
    for feature_name, mode in config.items():
        normalizers[feature_name] = FeatureNormalizer(mode=mode)

    return normalizers


def fit_normalizers_from_scenarios(
    normalizers: Dict[str, FeatureNormalizer],
    scenario_data: Dict[str, np.ndarray],
) -> Dict[str, FeatureNormalizer]:
    """
    Fit all normalizers to training data.

    Args:
        normalizers: Dict of unfitted normalizers
        scenario_data: Dict with keys matching normalizer names,
                      values are arrays to fit on

    Returns:
        normalizers: Dict of fitted normalizers
    """
    for feature_name, normalizer in normalizers.items():
        if feature_name in scenario_data:
            print(f"Fitting normalizer for '{feature_name}'...")
            normalizer.fit(scenario_data[feature_name])
        else:
            print(f"Warning: No data provided for '{feature_name}', skipping fit.")

    return normalizers


def save_normalizers(normalizers: Dict[str, FeatureNormalizer], save_dir: str):
    """
    Save all normalizers to a directory.

    Args:
        normalizers: Dict of fitted normalizers
        save_dir: Directory to save normalizer pickle files
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    for feature_name, normalizer in normalizers.items():
        filepath = os.path.join(save_dir, f"{feature_name}_normalizer.pkl")
        normalizer.save(filepath)

    print(f"All normalizers saved to {save_dir}")


def load_normalizers(save_dir: str) -> Dict[str, FeatureNormalizer]:
    """
    Load all normalizers from a directory.

    Args:
        save_dir: Directory containing normalizer pickle files

    Returns:
        normalizers: Dict of loaded normalizers
    """
    import os
    import glob

    normalizers = {}
    pkl_files = glob.glob(os.path.join(save_dir, "*_normalizer.pkl"))

    for pkl_file in pkl_files:
        feature_name = os.path.basename(pkl_file).replace('_normalizer.pkl', '')
        normalizer = FeatureNormalizer()
        normalizer.load(pkl_file)
        normalizers[feature_name] = normalizer

    print(f"Loaded {len(normalizers)} normalizers from {save_dir}")
    return normalizers


if __name__ == "__main__":
    # Test normalizers
    print("Testing normalizers...")

    # Create test data
    np.random.seed(42)
    bhp_data = np.random.uniform(900, 1400, size=(100, 61, 10))  # 100 scenarios, 61 timesteps, 10 wells
    perm_data = np.random.lognormal(mean=np.log(100), sigma=0.5, size=(100, 100))

    # Test individual normalizer
    print("\n1. MinMax Normalizer (BHP):")
    bhp_norm = FeatureNormalizer(mode='minmax')
    bhp_norm.fit(bhp_data)
    bhp_normalized = bhp_norm.transform(bhp_data)
    print(f"   Original range: [{bhp_data.min():.1f}, {bhp_data.max():.1f}]")
    print(f"   Normalized range: [{bhp_normalized.min():.4f}, {bhp_normalized.max():.4f}]")

    # Test inverse transform
    bhp_reconstructed = bhp_norm.inverse_transform(bhp_normalized)
    error = np.abs(bhp_reconstructed - bhp_data).max()
    print(f"   Reconstruction error: {error:.6f}")

    # Test log normalizer
    print("\n2. Log Normalizer (Permeability):")
    perm_norm = FeatureNormalizer(mode='log')
    perm_norm.fit(perm_data)
    perm_normalized = perm_norm.transform(perm_data)
    print(f"   Original range: [{perm_data.min():.1f}, {perm_data.max():.1f}] mD")
    print(f"   Log-normalized range: [{perm_normalized.min():.2f}, {perm_normalized.max():.2f}]")

    # Test creating full normalizer set
    print("\n3. Create full normalizer set:")
    normalizers = create_normalizers()
    print(f"   Created {len(normalizers)} normalizers:")
    for name, norm in list(normalizers.items())[:5]:
        print(f"     - {name}: {norm.mode}")

    print("\n✓ Normalizer tests passed!")
