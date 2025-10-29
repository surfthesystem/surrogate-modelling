"""
Reservoir Model Setup and Configuration
Generates heterogeneous permeability/porosity fields and well placements
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import lognorm
import yaml
import json
import os
from pathlib import Path


class ReservoirModel:
    """
    2D Heterogeneous Reservoir Model

    Generates:
    - Spatially correlated permeability field (log-normal distribution)
    - Correlated porosity field
    - Strategic well placements with spacing constraints
    """

    def __init__(self, config_path="config.yaml"):
        """Initialize reservoir model with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.nx = self.config['reservoir']['grid']['nx']
        self.ny = self.config['reservoir']['grid']['ny']
        self.dx = self.config['reservoir']['grid']['dx']
        self.dy = self.config['reservoir']['grid']['dy']

        # Physical domain size in meters
        self.Lx = self.nx * self.dx
        self.Ly = self.ny * self.dy

        # Create coordinate arrays
        self.x = np.linspace(self.dx/2, self.Lx - self.dx/2, self.nx)
        self.y = np.linspace(self.dy/2, self.Ly - self.dy/2, self.ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # Initialize fields
        self.permeability = None
        self.porosity = None
        self.producer_locations = None
        self.injector_locations = None

        print(f"Initialized {self.nx}x{self.ny} reservoir model")
        print(f"Domain size: {self.Lx/1000:.1f} km x {self.Ly/1000:.1f} km")

    def generate_permeability_field(self, seed=42):
        """
        Generate spatially correlated log-normal permeability field

        Uses Gaussian Random Field with exponential covariance function:
        C(r) = σ² * exp(-r / correlation_length)

        Args:
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)

        perm_config = self.config['reservoir']['rock']['permeability']
        mean_perm = perm_config['mean']
        min_perm = perm_config['min']
        max_perm = perm_config['max']
        corr_length = perm_config['correlation_length']

        print(f"\nGenerating permeability field...")
        print(f"  Mean: {mean_perm} mD, Range: [{min_perm}, {max_perm}] mD")
        print(f"  Correlation length: {corr_length} m")

        # Create distance matrix for all grid points
        # This is computationally expensive for large grids, so we use a
        # spectral method (FFT) for efficiency

        # For log-normal distribution:
        # If X ~ N(μ, σ²), then Y = exp(X) ~ LogNormal
        # We want E[Y] = mean_perm
        # For lognormal: E[Y] = exp(μ + σ²/2)
        # We'll use σ² = 1.0 for reasonable heterogeneity

        sigma_gaussian = 1.0
        mu_gaussian = np.log(mean_perm) - 0.5 * sigma_gaussian**2

        # Generate correlated Gaussian field using FFT method
        # This is much faster than computing full covariance matrix
        gaussian_field = self._generate_gaussian_random_field(
            corr_length, sigma_gaussian
        )

        # Add mean
        gaussian_field += mu_gaussian

        # Transform to log-normal
        perm_field = np.exp(gaussian_field)

        # Clip to specified range
        perm_field = np.clip(perm_field, min_perm, max_perm)

        self.permeability = perm_field

        print(f"  Generated field statistics:")
        print(f"    Mean: {np.mean(perm_field):.1f} mD")
        print(f"    Std:  {np.std(perm_field):.1f} mD")
        print(f"    Min:  {np.min(perm_field):.1f} mD")
        print(f"    Max:  {np.max(perm_field):.1f} mD")

        return perm_field

    def _generate_gaussian_random_field(self, correlation_length, sigma):
        """
        Generate spatially correlated Gaussian random field using FFT

        Uses spectral method:
        1. Generate white noise in Fourier space
        2. Multiply by square root of power spectrum
        3. Inverse FFT to get correlated field
        """
        # Generate white noise
        white_noise = np.random.randn(self.ny, self.nx)

        # Compute wavenumbers
        kx = np.fft.fftfreq(self.nx, self.dx) * 2 * np.pi
        ky = np.fft.fftfreq(self.ny, self.dy) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky)
        K = np.sqrt(KX**2 + KY**2)

        # Power spectrum for exponential covariance
        # S(k) = (2π σ² L²) / (1 + (kL)²)^(3/2)
        # For 2D exponential covariance
        L = correlation_length
        power_spectrum = (2 * np.pi * sigma**2 * L**2) / (1 + (K * L)**2)**(3/2)

        # Avoid division issues at k=0
        power_spectrum[0, 0] = 0

        # FFT of white noise
        noise_fft = np.fft.fft2(white_noise)

        # Multiply by sqrt of power spectrum
        field_fft = noise_fft * np.sqrt(power_spectrum)

        # Inverse FFT to get correlated field
        correlated_field = np.real(np.fft.ifft2(field_fft))

        # Normalize to have specified standard deviation
        correlated_field = correlated_field / np.std(correlated_field) * sigma

        return correlated_field

    def generate_porosity_field(self):
        """
        Generate porosity field correlated with permeability

        Uses Carman-Kozeny-like relationship:
        porosity = 0.15 + 0.0005 * sqrt(permeability)
        """
        if self.permeability is None:
            raise ValueError("Must generate permeability field first")

        poro_config = self.config['reservoir']['rock']['porosity']
        min_poro = poro_config['min']
        max_poro = poro_config['max']

        print(f"\nGenerating porosity field...")

        # Carman-Kozeny relationship
        porosity_field = 0.15 + 0.0005 * np.sqrt(self.permeability)

        # Clip to physical range
        porosity_field = np.clip(porosity_field, min_poro, max_poro)

        self.porosity = porosity_field

        print(f"  Mean: {np.mean(porosity_field):.3f}")
        print(f"  Std:  {np.std(porosity_field):.3f}")
        print(f"  Min:  {np.min(porosity_field):.3f}")
        print(f"  Max:  {np.max(porosity_field):.3f}")

        # Compute correlation with permeability
        corr = np.corrcoef(self.permeability.flatten(), porosity_field.flatten())[0, 1]
        print(f"  Correlation with permeability: {corr:.3f}")

        return porosity_field

    def place_wells(self, seed=42):
        """
        Place producer and injector wells with spatial constraints

        Strategy:
        - Producers: Stratified sampling (divide domain into regions)
        - Injectors: Place between producer clusters
        - Constraints: Minimum spacing, avoid low permeability zones

        Args:
            seed: Random seed for reproducibility
        """
        if self.permeability is None:
            raise ValueError("Must generate permeability field first")

        np.random.seed(seed)

        n_producers = self.config['wells']['producers']['count']
        n_injectors = self.config['wells']['injectors']['count']
        min_spacing = self.config['wells']['producers']['min_spacing']

        print(f"\nPlacing wells...")
        print(f"  Producers: {n_producers}")
        print(f"  Injectors: {n_injectors}")
        print(f"  Minimum spacing: {min_spacing} m")

        # Place producers using stratified sampling
        self.producer_locations = self._place_producers_stratified(
            n_producers, min_spacing, seed
        )

        # Place injectors between producers
        self.injector_locations = self._place_injectors(
            n_injectors, min_spacing, seed + 1
        )

        print(f"  Successfully placed {len(self.producer_locations)} producers")
        print(f"  Successfully placed {len(self.injector_locations)} injectors")

        return self.producer_locations, self.injector_locations

    def _place_producers_stratified(self, n_producers, min_spacing, seed):
        """
        Place producers using stratified sampling

        Divide reservoir into grid regions and place one well per region
        with random offset, avoiding low permeability zones
        """
        np.random.seed(seed)

        # Determine grid layout (as close to square as possible)
        n_rows = int(np.sqrt(n_producers))
        n_cols = int(np.ceil(n_producers / n_rows))

        # Region size
        region_width = self.Lx / n_cols
        region_height = self.Ly / n_rows

        producers = []
        max_attempts = 1000

        for i in range(n_rows):
            for j in range(n_cols):
                if len(producers) >= n_producers:
                    break

                # Region boundaries
                x_min = j * region_width
                x_max = (j + 1) * region_width
                y_min = i * region_height
                y_max = (i + 1) * region_height

                # Try to place well in this region
                placed = False
                for attempt in range(max_attempts):
                    # Random location within region (with some margin)
                    margin = min_spacing * 0.2
                    x = np.random.uniform(x_min + margin, x_max - margin)
                    y = np.random.uniform(y_min + margin, y_max - margin)

                    # Check permeability at this location
                    ix = int(x / self.dx)
                    iy = int(y / self.dy)
                    ix = np.clip(ix, 0, self.nx - 1)
                    iy = np.clip(iy, 0, self.ny - 1)

                    perm = self.permeability[iy, ix]

                    # Avoid very low permeability zones
                    if perm < 20:
                        continue

                    # Check spacing from existing wells
                    if len(producers) > 0:
                        distances = cdist([[x, y]], producers)[0]
                        if np.min(distances) < min_spacing:
                            continue

                    # Valid location found
                    producers.append([x, y])
                    placed = True
                    break

                if not placed:
                    print(f"    Warning: Could not place producer in region ({i}, {j})")

        return np.array(producers)

    def _place_injectors(self, n_injectors, min_spacing, seed):
        """
        Place injectors to provide good sweep

        Strategy: Place injectors in regions with good permeability
        that are not too close to producers
        """
        np.random.seed(seed)

        injectors = []
        max_attempts = 5000

        # Preferred distance from producers (not too close, not too far)
        preferred_dist_min = min_spacing * 1.2
        preferred_dist_max = min_spacing * 3.0

        for i in range(n_injectors):
            placed = False

            for attempt in range(max_attempts):
                # Random location
                x = np.random.uniform(min_spacing, self.Lx - min_spacing)
                y = np.random.uniform(min_spacing, self.Ly - min_spacing)

                # Check permeability
                ix = int(x / self.dx)
                iy = int(y / self.dy)
                ix = np.clip(ix, 0, self.nx - 1)
                iy = np.clip(iy, 0, self.ny - 1)

                perm = self.permeability[iy, ix]

                # Prefer moderate to high permeability
                if perm < 50:
                    continue

                # Check spacing from existing injectors
                if len(injectors) > 0:
                    dist_to_injectors = cdist([[x, y]], injectors)[0]
                    if np.min(dist_to_injectors) < min_spacing:
                        continue

                # Check distance from producers
                dist_to_producers = cdist([[x, y]], self.producer_locations)[0]
                min_dist_prod = np.min(dist_to_producers)

                # Should not be too close to producers
                if min_dist_prod < min_spacing:
                    continue

                # Prefer locations that are at good distance from producers
                # This is a soft constraint
                if min_dist_prod > preferred_dist_min:
                    injectors.append([x, y])
                    placed = True
                    break
                elif attempt > max_attempts * 0.8:  # Relax constraint if struggling
                    injectors.append([x, y])
                    placed = True
                    break

            if not placed:
                print(f"    Warning: Could not place injector {i+1}")

        return np.array(injectors)

    def visualize_reservoir(self, save_path="results/reservoir_visualization.png"):
        """
        Create comprehensive visualization of reservoir model

        Shows:
        - Permeability field
        - Porosity field
        - Well locations
        """
        if self.permeability is None or self.porosity is None:
            raise ValueError("Must generate fields first")

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Permeability plot
        ax1 = axes[0]
        im1 = ax1.imshow(
            self.permeability,
            extent=[0, self.Lx/1000, 0, self.Ly/1000],
            origin='lower',
            cmap='jet',
            aspect='equal'
        )

        # Add wells
        if self.producer_locations is not None:
            ax1.scatter(
                self.producer_locations[:, 0]/1000,
                self.producer_locations[:, 1]/1000,
                c='red', s=100, marker='o',
                edgecolors='white', linewidths=1.5,
                label=f'Producers ({len(self.producer_locations)})',
                zorder=10
            )

        if self.injector_locations is not None:
            ax1.scatter(
                self.injector_locations[:, 0]/1000,
                self.injector_locations[:, 1]/1000,
                c='blue', s=100, marker='^',
                edgecolors='white', linewidths=1.5,
                label=f'Injectors ({len(self.injector_locations)})',
                zorder=10
            )

        ax1.set_xlabel('X (km)', fontsize=12)
        ax1.set_ylabel('Y (km)', fontsize=12)
        ax1.set_title('Permeability Field with Well Locations', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('Permeability (mD)', fontsize=11)

        # Porosity plot
        ax2 = axes[1]
        im2 = ax2.imshow(
            self.porosity,
            extent=[0, self.Lx/1000, 0, self.Ly/1000],
            origin='lower',
            cmap='viridis',
            aspect='equal'
        )

        # Add wells
        if self.producer_locations is not None:
            ax2.scatter(
                self.producer_locations[:, 0]/1000,
                self.producer_locations[:, 1]/1000,
                c='red', s=100, marker='o',
                edgecolors='white', linewidths=1.5,
                zorder=10
            )

        if self.injector_locations is not None:
            ax2.scatter(
                self.injector_locations[:, 0]/1000,
                self.injector_locations[:, 1]/1000,
                c='cyan', s=100, marker='^',
                edgecolors='white', linewidths=1.5,
                zorder=10
            )

        ax2.set_xlabel('X (km)', fontsize=12)
        ax2.set_ylabel('Y (km)', fontsize=12)
        ax2.set_title('Porosity Field with Well Locations', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label('Porosity', fontsize=11)

        plt.tight_layout()

        # Save figure
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}")

        return fig

    def save_reservoir_data(self, data_dir="data"):
        """Save reservoir data to files"""
        os.makedirs(data_dir, exist_ok=True)

        # Save numpy arrays
        np.save(os.path.join(data_dir, 'permeability_field.npy'), self.permeability)
        np.save(os.path.join(data_dir, 'porosity_field.npy'), self.porosity)

        # Save well locations as CSV
        import pandas as pd

        # Producers
        prod_df = pd.DataFrame(
            self.producer_locations,
            columns=['x_m', 'y_m']
        )
        prod_df['well_type'] = 'producer'
        prod_df['well_id'] = [f'PROD_{i:03d}' for i in range(len(prod_df))]

        # Get permeability and porosity at well locations
        prod_df['perm_mD'] = self._interpolate_field_at_wells(
            self.permeability, self.producer_locations
        )
        prod_df['porosity'] = self._interpolate_field_at_wells(
            self.porosity, self.producer_locations
        )

        # Injectors
        inj_df = pd.DataFrame(
            self.injector_locations,
            columns=['x_m', 'y_m']
        )
        inj_df['well_type'] = 'injector'
        inj_df['well_id'] = [f'INJ_{i:03d}' for i in range(len(inj_df))]

        inj_df['perm_mD'] = self._interpolate_field_at_wells(
            self.permeability, self.injector_locations
        )
        inj_df['porosity'] = self._interpolate_field_at_wells(
            self.porosity, self.injector_locations
        )

        # Combine and save
        wells_df = pd.concat([prod_df, inj_df], ignore_index=True)
        wells_df.to_csv(os.path.join(data_dir, 'well_locations.csv'), index=False)

        # Save reservoir configuration as JSON
        config_dict = {
            'grid': {
                'nx': self.nx,
                'ny': self.ny,
                'dx': self.dx,
                'dy': self.dy,
                'Lx': self.Lx,
                'Ly': self.Ly
            },
            'permeability': {
                'mean': float(np.mean(self.permeability)),
                'std': float(np.std(self.permeability)),
                'min': float(np.min(self.permeability)),
                'max': float(np.max(self.permeability))
            },
            'porosity': {
                'mean': float(np.mean(self.porosity)),
                'std': float(np.std(self.porosity)),
                'min': float(np.min(self.porosity)),
                'max': float(np.max(self.porosity))
            },
            'wells': {
                'n_producers': len(self.producer_locations),
                'n_injectors': len(self.injector_locations)
            }
        }

        with open(os.path.join(data_dir, 'reservoir_config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)

        print(f"\nReservoir data saved to: {data_dir}/")
        print(f"  - permeability_field.npy")
        print(f"  - porosity_field.npy")
        print(f"  - well_locations.csv")
        print(f"  - reservoir_config.json")

    def _interpolate_field_at_wells(self, field, well_locations):
        """Interpolate field values at well locations"""
        values = []
        for x, y in well_locations:
            ix = int(x / self.dx)
            iy = int(y / self.dy)
            ix = np.clip(ix, 0, self.nx - 1)
            iy = np.clip(iy, 0, self.ny - 1)
            values.append(field[iy, ix])
        return values


def main():
    """Main function to set up reservoir model"""
    print("=" * 70)
    print("GNN-LSTM RESERVOIR SURROGATE MODEL - PHASE 1")
    print("Reservoir Model Setup")
    print("=" * 70)

    # Initialize model
    model = ReservoirModel(config_path="config.yaml")

    # Generate permeability field
    model.generate_permeability_field(seed=42)

    # Generate porosity field
    model.generate_porosity_field()

    # Place wells
    model.place_wells(seed=42)

    # Visualize
    model.visualize_reservoir()

    # Save data
    model.save_reservoir_data()

    print("\n" + "=" * 70)
    print("PHASE 1 COMPLETE: Reservoir model setup successful!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Review reservoir_visualization.png in results/")
    print("  2. Check well_locations.csv in data/")
    print("  3. Proceed to Phase 2: Training data generation")


if __name__ == "__main__":
    main()
