import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Callable

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_comparison_plot(
    model: nn.Module,
    nu_fixed: float,
    noise_args: list[torch.Tensor],
    analytical_mean_solution: Callable,
    *,
    vartheta: float = 1.0,
    d: int = 2,
    T: float = 1.0,
    res: int = 50,
    forcing_type: str = "linear",
    scaling: float = 1.0
):
    """
    2×3 figure comparing PINN vs. analytical mean for selected times (2-D).
    """
    if d != 2:        # only implemented for 2 spatial dimensions
        return None

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    times = (0.0, T / 4, T / 2)
    noise_args = [torch.full((res**2, 1), n.mean(), device=DEVICE) for n in noise_args]
    # Create uniform (x1,x2) grid
    x1 = torch.linspace(0, 1, res)
    x2 = torch.linspace(0, 1, res)
    X1, X2 = torch.meshgrid(x1, x2, indexing="ij")

    for col, t_val in enumerate(times):
        t_grid = torch.full_like(X1, t_val)
        nu_grid = torch.full_like(X1, nu_fixed)
        grid_pts = torch.stack(
            [t_grid.flatten(), X1.flatten(), X2.flatten(), nu_grid.flatten()],
            dim=1,
        ).to(DEVICE)

        # PINN prediction
        pred = model(torch.cat([grid_pts, *noise_args], dim=1)).detach().cpu().view(res, res)
        pred = pred * scaling
        # Analytical mean
        true = analytical_mean_solution(
            t=torch.full((res**2, 1), t_val),
            x=torch.stack([X1.flatten(), X2.flatten()], dim=1),
            nu=torch.full((res**2, 1), nu_fixed),
            vartheta=vartheta,
            noise_args=noise_args,
            d=d,
            forcing=forcing_type
        ).view(res, res).cpu().numpy()

        # ───── plot ─────
        im1 = axes[0, col].imshow(pred, cmap="viridis", extent=[0, 1, 0, 1], origin="lower")
        axes[0, col].set_title(f"PINN  t={t_val:.2f}")
        axes[0, col].set_xlabel("x₁")
        axes[0, col].set_ylabel("x₂")
        plt.colorbar(im1, ax=axes[0, col])

        im2 = axes[1, col].imshow(true, cmap="viridis", extent=[0, 1, 0, 1], origin="lower")
        axes[1, col].set_title(f"Exact t={t_val:.2f}")
        axes[1, col].set_xlabel("x₁")
        axes[1, col].set_ylabel("x₂")
        plt.colorbar(im2, ax=axes[1, col])

    plt.suptitle(f"ν={nu_fixed:.4f}", fontsize=16)
    plt.tight_layout()
    return fig
