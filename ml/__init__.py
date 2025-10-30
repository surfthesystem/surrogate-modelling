"""
Enhanced GNN-LSTM Surrogate Model for Reservoir Simulation

This package implements a graph neural network (GNN) combined with LSTM
for fast and accurate prediction of well production rates in reservoir simulation.

Key improvements over the baseline paper (SPE-215842):
- Rich edge features: permeability, pressure gradients, flow connectivity
- Data-driven connectivity from time-lagged correlations
- Hybrid static + dynamic feature engineering
"""

__version__ = "0.1.0"
__author__ = "Reservoir Surrogate Team"
