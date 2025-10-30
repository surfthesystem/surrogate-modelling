"""
Loss functions for training the GNN-LSTM surrogate model.

Includes weighted losses for oil/water rates and physics-based constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def weighted_l1_loss(
    pred_oil: torch.Tensor,
    pred_water: torch.Tensor,
    target_oil: torch.Tensor,
    target_water: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 80.0,
) -> torch.Tensor:
    """
    Weighted L1 loss for oil and water production rates.

    From SPE-215842 paper: Beta=80 for oil (more important economically),
    alpha=1 for water.

    Args:
        pred_oil: Predicted oil rates, shape (batch, T, num_producers)
        pred_water: Predicted water rates, shape (batch, T, num_producers)
        target_oil: Ground truth oil rates, same shape
        target_water: Ground truth water rates, same shape
        alpha: Weight for water loss (default: 1.0)
        beta: Weight for oil loss (default: 80.0, from paper)

    Returns:
        loss: Weighted L1 loss (scalar)
    """
    loss_oil = F.l1_loss(pred_oil, target_oil)
    loss_water = F.l1_loss(pred_water, target_water)

    total_loss = beta * loss_oil + alpha * loss_water

    return total_loss


def weighted_mse_loss(
    pred_oil: torch.Tensor,
    pred_water: torch.Tensor,
    target_oil: torch.Tensor,
    target_water: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 80.0,
) -> torch.Tensor:
    """
    Weighted MSE loss for oil and water production rates.

    Alternative to L1 loss - MSE penalizes large errors more heavily.

    Args:
        Same as weighted_l1_loss

    Returns:
        loss: Weighted MSE loss (scalar)
    """
    loss_oil = F.mse_loss(pred_oil, target_oil)
    loss_water = F.mse_loss(pred_water, target_water)

    total_loss = beta * loss_oil + alpha * loss_water

    return total_loss


def cumulative_production_loss(
    pred_oil: torch.Tensor,
    pred_water: torch.Tensor,
    target_oil: torch.Tensor,
    target_water: torch.Tensor,
    weight: float = 0.1,
) -> torch.Tensor:
    """
    Loss on cumulative production to ensure mass balance consistency.

    Cumulative production is often more important for economic evaluation
    than instantaneous rates.

    Args:
        pred_oil: Predicted oil rates, shape (batch, T, num_producers)
        pred_water: Predicted water rates
        target_oil: Target oil rates
        target_water: Target water rates
        weight: Weight for cumulative loss

    Returns:
        loss: Cumulative production loss (scalar)
    """
    # Compute cumulative production (sum over time)
    cum_pred_oil = torch.cumsum(pred_oil, dim=1)
    cum_target_oil = torch.cumsum(target_oil, dim=1)

    cum_pred_water = torch.cumsum(pred_water, dim=1)
    cum_target_water = torch.cumsum(target_water, dim=1)

    # MSE on cumulative production
    loss_cum_oil = F.mse_loss(cum_pred_oil, cum_target_oil)
    loss_cum_water = F.mse_loss(cum_pred_water, cum_target_water)

    total_loss = weight * (loss_cum_oil + loss_cum_water)

    return total_loss


def physics_loss(
    pred_oil: torch.Tensor,
    pred_water: torch.Tensor,
    target_oil: torch.Tensor,
    target_water: torch.Tensor,
    weight: float = 0.01,
) -> torch.Tensor:
    """
    Physics-based constraints on production rates.

    Ensures:
    1. Cumulative production is monotonically increasing
    2. Rates remain non-negative (enforced by softplus activation)
    3. Smooth rate changes (penalize large discontinuities)

    Args:
        pred_oil: Predicted oil rates, shape (batch, T, num_producers)
        pred_water: Predicted water rates
        target_oil: Target oil rates (for reference)
        target_water: Target water rates
        weight: Weight for physics loss

    Returns:
        loss: Physics constraint violation penalty (scalar)
    """
    loss = 0.0

    # 1. Monotonicity constraint on cumulative production
    # Cumulative production should not decrease over time
    cum_oil = torch.cumsum(pred_oil, dim=1)  # (batch, T, num_prod)
    cum_water = torch.cumsum(pred_water, dim=1)

    # Compute differences between consecutive timesteps
    diff_cum_oil = cum_oil[:, 1:, :] - cum_oil[:, :-1, :]  # (batch, T-1, num_prod)
    diff_cum_water = cum_water[:, 1:, :] - cum_water[:, :-1, :]

    # Penalize negative differences (violations of monotonicity)
    monotonicity_viol_oil = F.relu(-diff_cum_oil)  # Penalize if < 0
    monotonicity_viol_water = F.relu(-diff_cum_water)

    loss += (monotonicity_viol_oil.mean() + monotonicity_viol_water.mean()) * weight

    # 2. Smoothness constraint: penalize large rate changes
    # Rates shouldn't jump drastically between timesteps
    diff_oil = pred_oil[:, 1:, :] - pred_oil[:, :-1, :]  # (batch, T-1, num_prod)
    diff_water = pred_water[:, 1:, :] - pred_water[:, :-1, :]

    # Penalize large absolute changes
    smoothness_penalty = (diff_oil.abs().mean() + diff_water.abs().mean()) * weight * 0.1

    loss += smoothness_penalty

    return loss


def relative_error_loss(
    pred_oil: torch.Tensor,
    pred_water: torch.Tensor,
    target_oil: torch.Tensor,
    target_water: torch.Tensor,
    eps: float = 1e-6,
    weight: float = 1.0,
) -> torch.Tensor:
    """
    Relative error loss (MAPE-style).

    Focuses on relative accuracy rather than absolute errors,
    which is better when rates vary over orders of magnitude.

    Args:
        pred_oil: Predicted oil rates
        pred_water: Predicted water rates
        target_oil: Target oil rates
        target_water: Target water rates
        eps: Small constant to avoid division by zero
        weight: Loss weight

    Returns:
        loss: Mean relative error (scalar)
    """
    rel_error_oil = torch.abs((pred_oil - target_oil) / (target_oil + eps))
    rel_error_water = torch.abs((pred_water - target_water) / (target_water + eps))

    loss = weight * (rel_error_oil.mean() + rel_error_water.mean())

    return loss


def combined_loss(
    predictions: dict,
    targets: dict,
    alpha: float = 1.0,
    beta: float = 80.0,
    use_cumulative: bool = True,
    cumulative_weight: float = 0.1,
    use_physics: bool = True,
    physics_weight: float = 0.01,
    use_relative: bool = False,
    relative_weight: float = 0.5,
    loss_type: str = 'l1',
) -> tuple:
    """
    Combined loss function with multiple components.

    Args:
        predictions: Dict with 'oil_rates' and 'water_rates' tensors
        targets: Dict with 'targets_oil' and 'targets_water' tensors
        alpha: Weight for water in main loss
        beta: Weight for oil in main loss (default: 80.0 from paper)
        use_cumulative: Include cumulative production loss
        cumulative_weight: Weight for cumulative loss
        use_physics: Include physics-based constraints
        physics_weight: Weight for physics loss
        use_relative: Include relative error loss
        relative_weight: Weight for relative error
        loss_type: 'l1' or 'mse' for main loss

    Returns:
        total_loss: Combined loss (scalar)
        loss_dict: Dictionary of individual loss components
    """
    pred_oil = predictions['oil_rates']
    pred_water = predictions['water_rates']
    target_oil = targets['targets_oil']
    target_water = targets['targets_water']

    loss_dict = {}

    # Main loss (L1 or MSE)
    if loss_type == 'l1':
        main_loss = weighted_l1_loss(pred_oil, pred_water, target_oil, target_water, alpha, beta)
    elif loss_type == 'mse':
        main_loss = weighted_mse_loss(pred_oil, pred_water, target_oil, target_water, alpha, beta)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Use 'l1' or 'mse'.")

    loss_dict['main_loss'] = main_loss.item()
    total_loss = main_loss

    # Cumulative production loss
    if use_cumulative:
        cum_loss = cumulative_production_loss(
            pred_oil, pred_water, target_oil, target_water, cumulative_weight
        )
        loss_dict['cumulative_loss'] = cum_loss.item()
        total_loss = total_loss + cum_loss

    # Physics constraints
    if use_physics:
        phys_loss = physics_loss(
            pred_oil, pred_water, target_oil, target_water, physics_weight
        )
        loss_dict['physics_loss'] = phys_loss.item()
        total_loss = total_loss + phys_loss

    # Relative error loss
    if use_relative:
        rel_loss = relative_error_loss(
            pred_oil, pred_water, target_oil, target_water, weight=relative_weight
        )
        loss_dict['relative_loss'] = rel_loss.item()
        total_loss = total_loss + rel_loss

    loss_dict['total_loss'] = total_loss.item()

    return total_loss, loss_dict


if __name__ == "__main__":
    # Test loss functions
    print("Testing loss functions...")

    torch.manual_seed(42)

    batch_size = 4
    T = 61
    num_prod = 10

    # Create dummy predictions and targets
    pred_oil = torch.rand(batch_size, T, num_prod) * 100  # 0-100 STB/day
    pred_water = torch.rand(batch_size, T, num_prod) * 50  # 0-50 STB/day

    # Targets (similar but slightly different)
    target_oil = pred_oil + torch.randn_like(pred_oil) * 10
    target_water = pred_water + torch.randn_like(pred_water) * 5

    # Test weighted L1 loss
    print("\n1. Testing weighted_l1_loss...")
    loss_l1 = weighted_l1_loss(pred_oil, pred_water, target_oil, target_water)
    print(f"   L1 Loss: {loss_l1.item():.4f}")

    # Test cumulative loss
    print("\n2. Testing cumulative_production_loss...")
    loss_cum = cumulative_production_loss(pred_oil, pred_water, target_oil, target_water)
    print(f"   Cumulative Loss: {loss_cum.item():.4f}")

    # Test physics loss
    print("\n3. Testing physics_loss...")
    loss_phys = physics_loss(pred_oil, pred_water, target_oil, target_water)
    print(f"   Physics Loss: {loss_phys.item():.6f}")

    # Test relative error loss
    print("\n4. Testing relative_error_loss...")
    loss_rel = relative_error_loss(pred_oil, pred_water, target_oil, target_water)
    print(f"   Relative Error Loss: {loss_rel.item():.4f}")

    # Test combined loss
    print("\n5. Testing combined_loss...")
    predictions = {'oil_rates': pred_oil, 'water_rates': pred_water}
    targets = {'targets_oil': target_oil, 'targets_water': target_water}

    total_loss, loss_dict = combined_loss(
        predictions, targets,
        use_cumulative=True,
        use_physics=True,
        use_relative=True,
    )

    print(f"   Total Loss: {total_loss.item():.4f}")
    print(f"   Loss components:")
    for name, value in loss_dict.items():
        print(f"     - {name}: {value:.4f}")

    # Test gradient flow
    print("\n6. Testing gradient flow...")
    total_loss.backward()
    print(f"   ✓ Gradients computed successfully")

    print("\n✓ Loss function tests passed!")
