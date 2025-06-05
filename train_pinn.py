"""
PINN for a heat-type PDE with time-dependent stochastic forcing.

Key features
------------
*   Physics-informed loss: interior PDE residual + Dirichlet BC + IC.
*   Forcing term     W(xi(t))  with   xi(t) = μ t + σ W_t.
*   Optional exact-solution monitor for a fixed viscosity ν.
*   TensorBoard logging of scalars and comparison plots.

"""
from __future__ import annotations
from typing import Callable
import os

# ───────────────────── 3rd-party imports ──────────────────────
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# ──────────────────── project-local imports ───────────────────
from utils.model_utils import FNN
from utils.viz_utils import create_comparison_plot
from utils.sampling_utils import (
    time_dependant_gaussian_white_noise,
    ou_process,
    compound_poisson_process,
    sample_boundary_points,
    sample_initial_points,
    sample_points,
    sample_noise_args,
)
from utils.pde_utils import (
    time_shifted_gaussian_analytical_mean_solution,
    compound_poisson_analytical_mean_solution,
    ou_analytical_mean_solution,
)

# ────────────────────────── device ────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Using device: {DEVICE}")


# ======================================================================
#                            Loss functions
# ======================================================================

def _laplacian(u: torch.Tensor, pts: torch.Tensor, d: int) -> torch.Tensor:
    """Compute ∇²u via autograd for the *spatial* coordinates only."""
    grads = torch.autograd.grad(u.sum(), pts, create_graph=True)[0]
    lap = 0.0
    for i in range(1, d + 1):                  # skip time column
        grad_i = grads[:, i: i + 1]
        lap += torch.autograd.grad(grad_i.sum(), pts, create_graph=True)[0][:, i: i + 1]
    return lap


def compute_pde_residual(
    model: nn.Module,
    pts: torch.Tensor,
    w_t: torch.Tensor,
    vartheta: float,
    noise_args: list[torch.Tensor],
    *,
    forcing_type: str = "linear",
    scaling: float = 1.0,
) -> torch.Tensor:
    """
    Return PDE residual  r = u_t − ν ∇²u − f·vartheta·∏ sin(π x_i).
    """
    pts.requires_grad_(True)                 # enable autograd w.r.t. inputs
    model_in = torch.cat([pts, *noise_args], dim=1)
    u = model(model_in) * scaling  # scale output to match PDE units

    nu = pts[:, -1:]                    # viscosity column
    d = pts.shape[1] - 2               # #spatial dims
    x_space = pts[:, 1: 1 + d]
    t = pts[:, 0:1]                  # time column

    du_dt = torch.autograd.grad(u.sum(), pts, create_graph=True)[0][:, 0:1]
    lap_u = _laplacian(u, pts, d)

    # Forcing  f(t)·vartheta·G(x)
    Gx = torch.prod(torch.sin(np.pi * x_space), dim=1, keepdim=True)
    if forcing_type == "linear":
        forcing = w_t * vartheta * Gx
    elif forcing_type == "exp_linear":
        forcing = torch.exp(-t) * w_t * vartheta * Gx
    elif forcing_type == "square":
        forcing = (w_t**2) * vartheta * Gx
    else:  # pragma: no cover
        raise ValueError(f"Unsupported forcing type: {forcing_type}")

    return (du_dt - nu * lap_u - forcing)/scaling      # residual r(t,x,ν)


# ----------------------------------------------------------------------

def compute_losses(
    model: nn.Module,
    noise_args: list[torch.Tensor],
    noise_function: Callable,
    *,
    vartheta: float = 1.0,
    d: int = 2,
    T: float = 1.0,
    nu_range: list[float] = (0.01, 0.1),
    n_pde: int = 1000,
    n_bc: int = 100,
    n_ic: int = 100,
    m_samples: int = 10,
    forcing_type: str = "linear",  # Add forcing_type parameter
    scaling: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return (ℒ_pde, ℒ_bc, ℒ_ic) – each is a *scalar* tensor.
    """
    # ───── Interior loss (Monte-Carlo average) ─────
    pde_loss = 0.0
    p_pts = sample_points(n_pde, d, T, nu_range).to(DEVICE)
    for _ in range(m_samples):
        t = p_pts[:, 0:1]
        w_t = noise_function(t, *noise_args)
        res = compute_pde_residual(
            model, p_pts, w_t, vartheta, noise_args, forcing_type=forcing_type
        )
        pde_loss += (res.pow(2)).mean()/scaling
    pde_loss /= m_samples

    # ───── Boundary (Dirichlet) loss : u = 0 on ∂Ω ─────
    bc_pts = sample_boundary_points(n_bc, d, T, nu_range).to(DEVICE)
    bc_noise_args = [na[: bc_pts.shape[0]] for na in noise_args]
    bc_pred = model(torch.cat([bc_pts, *bc_noise_args], dim=1)) * \
        scaling  # scale output to match PDE units
    bc_loss = (bc_pred.pow(2)/scaling).mean()

    # ───── Initial-time loss : u(0,x) = vartheta·G(x) ─────
    ic_pts, ic_true = sample_initial_points(n_ic, d, vartheta, nu_range)
    ic_pts, ic_true = ic_pts.to(DEVICE), ic_true.to(DEVICE)
    ic_noise_args = [na[: n_ic] for na in noise_args]
    ic_pred = model(torch.cat([ic_pts, *ic_noise_args], dim=1)) * \
        scaling  # scale output to match PDE units
    ic_loss = ((ic_pred - ic_true).pow(2)/scaling).mean()

    return pde_loss, bc_loss, ic_loss

# ======================================================================
#                   Evaluation / visualisation helpers
# ======================================================================


@torch.no_grad()
def compute_l2_error(
    model: nn.Module,
    nu_fixed: float,
    noise_args: list[torch.Tensor],
    analytical_mean_solution: Callable,
    *,
    vartheta: float = 1.0,
    d: int = 2,
    T: float = 1.0,
    n_test: int = 1_000,
    forcing_type: str = "linear",
    scaling: float = 1.0,
) -> float:
    """
    Relative L2 error ‖u_pred − u_true‖₂ / ‖u_true‖₂  on random test set.
    """
    t = torch.rand(n_test, 1, device=DEVICE) * T
    x = torch.rand(n_test, d, device=DEVICE)
    nu = torch.full((n_test, 1), nu_fixed, device=DEVICE)

    pts = torch.cat([t, x, nu, *noise_args], dim=1)
    pred = model(pts)
    pred = pred * scaling  # scale output to match PDE units

    true = analytical_mean_solution(
        t=t.squeeze(),
        x=x,
        nu=torch.full_like(t.squeeze(), nu_fixed),
        vartheta=vartheta,
        noise_args=noise_args,
        d=d,
        forcing=forcing_type
    )
    return torch.norm(pred - true) / torch.norm(true)


# ======================================================================
#                              Training
# ======================================================================

def train_pinn(
    noise_function: Callable,
    analytical_mean_solution: Callable,
    spatial_dims: int = 2,
    add_dims: int = 3,
    forcing_type: str = "linear",
    lambda_pde: float = 1.0,
    lambda_bc: float = 10.0,
    lambda_ic: float = 10.0,
    hidden_dim: int = 128,
    num_layers: int = 6,
    T: float = 1.0,
    lr: float = 1e-2,
    m_samples: int = 10,
    epochs: int = 10_000,
    nu_range: tuple[float, float] = (0.01, 0.1),
    scheduler_gamma: float = 0.9995,
    n_pde: int = 1000,
    n_bc: int = 100,
    n_ic: int = 100,
    n_test: int = 1000,
    log_interval: int = 10,
    tensorboard_interval: int = 100,
    plot_interval: int = 500,
    noise_boundaries: tuple[tuple[float, float], ...] = ((0.1, 0.5), (0.1, 1.0)),
    save_model: bool = True,
) -> nn.Module:
    """
    Main training loop for Physics-Informed Neural Network (PINN).

    Parameters
    ----------
    spatial_dims : int, default=2
        Number of spatial dimensions
    forcing_type : str, default="linear"
        Type of forcing function: {"linear", "exp_linear", "square"}
    lambda_pde : float, default=1.0
        Weight for PDE residual loss
    lambda_bc : float, default=10.0
        Weight for boundary condition loss
    lambda_ic : float, default=10.0
        Weight for initial condition loss
    hidden_dim : int, default=128
        Hidden layer dimension for the neural network
    num_layers : int, default=6
        Number of hidden layers in the neural network
    T : float, default=1.0
        Final time for the domain
    lr : float, default=1e-2
        Learning rate for Adam optimizer
    m_samples : int, default=10
        Number of Monte Carlo samples for PDE loss computation
    epochs : int, default=10_000
        Number of training epochs
    nu_range : tuple[float, float], default=(0.01, 0.1)
        Range of viscosity values (min, max)
    scheduler_gamma : float, default=0.9995
        Decay factor for exponential learning rate scheduler
    n_pde : int, default=1000
        Number of interior points for PDE residual
    n_bc : int, default=100
        Number of boundary points
    n_ic : int, default=100
        Number of initial condition points
    n_test : int, default=1000
        Number of test points for L2 error computation
    log_interval : int, default=10
        Interval for logging progress
    tensorboard_interval : int, default=100
        Interval for TensorBoard logging
    plot_interval : int, default=500
        Interval for generating comparison plots
    noise_boundaries : tuple[tuple[float, float], tuple[float, float]], default=((0.1, 0.5), (0.1, 1.0))
        Boundaries for sampling noise parameters (mu_range, sigma_range)
    random_seed : int, default=42
        Random seed for reproducibility
    save_model : bool, default=True
        Whether to save the trained model

    Returns
    -------
    nn.Module
        Trained PINN model
    """
    # ───── Problem setup ─────
    d = spatial_dims
    scaling = 1/max(10*round(np.sqrt(d)-1), 1.0)  # scale output to match PDE units

    # Compute vartheta
    vartheta = np.sqrt(d) * np.log10(np.exp(d))

    # Validate forcing type
    valid_forcing_types = {"linear", "exp_linear", "square"}
    if forcing_type not in valid_forcing_types:
        raise ValueError(f"forcing_type must be one of {valid_forcing_types}, got {forcing_type}")

    # ───── Network ─────
    input_dim = d + add_dims         # t + x_d + ν + μ + σ
    model = FNN(input_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(DEVICE)

    #  ───── Noise Function ─────
    noise_name = noise_function.__name__

    print(f"🏗️  Model architecture: {input_dim} → {hidden_dim} (×{num_layers}) → 1")
    print(f"📊  Problem setup: {d}D, T={T}, forcing='{forcing_type}', vartheta={vartheta:.4f}")
    print(f"🔊  Noise function: {noise_name}")
    print(f"🔧  Scaling: {scaling:.4f}")

    # ───── Optimisation ─────
    optimiser = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimiser, gamma=scheduler_gamma)

    # ───── Loss weights ─────
    λ_pde, λ_bc, λ_ic = lambda_pde, lambda_bc, lambda_ic
    print(f"⚖️  Loss weights: λ_pde={λ_pde}, λ_bc={λ_bc}, λ_ic={λ_ic}")

    # ───── Logging ─────

    tb_dir = f"runs/{datetime.now():%Y%m%d_%H%M%S}_pinn_heat_{d}d_{forcing_type}"
    writer = SummaryWriter(tb_dir)
    print(f"📝  TensorBoard logs: {tb_dir}")
    print("🏋️  Training PINN …")

    #
    stamp = f"pinn_heat_{d}d_{forcing_type}_mc_samples_{m_samples}_noise_{noise_name}.pth"

    # ───── Training loop ─────
    bar = tqdm(range(epochs), desc="Epoch")
    for epoch in bar:
        model.train()

        # Fresh Gaussian parameters each epoch
        noise_args = sample_noise_args(
            n_points=max(n_pde, n_bc, n_ic),
            n_args=2,
            boundaries=noise_boundaries,
            device=DEVICE
        )

        # Forward + losses
        ℒ_pde, ℒ_bc, ℒ_ic = compute_losses(
            model=model,
            noise_args=noise_args,
            noise_function=noise_function,
            vartheta=vartheta,
            d=d,
            T=T,
            nu_range=nu_range,
            n_pde=n_pde,
            n_bc=n_bc,
            n_ic=n_ic,
            m_samples=m_samples,
            forcing_type=forcing_type,  # Pass forcing_type to compute_losses
            scaling=scaling,
        )
        loss = λ_pde * ℒ_pde + λ_bc * ℒ_bc + λ_ic * ℒ_ic

        # Backward
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        scheduler.step()

        # Periodic diagnostics
        if epoch % log_interval == 0:
            test_noise_args = sample_noise_args(
                n_points=n_test,
                n_args=2,
                boundaries=noise_boundaries,
                device=DEVICE
            )
            ν_mid = sum(nu_range) / 2
            l2 = compute_l2_error(
                model=model,
                nu_fixed=ν_mid,
                noise_args=test_noise_args,
                analytical_mean_solution=analytical_mean_solution,
                vartheta=vartheta,
                d=d,
                T=T,
                n_test=n_test,
                forcing_type=forcing_type
            )
            bar.set_postfix(loss=f"{loss:.3e}", L2=f"{l2:.2e}")

        # TensorBoard logging
        if epoch % tensorboard_interval == 0:
            writer.add_scalar("Loss/PDE", ℒ_pde.item(), epoch)
            writer.add_scalar("Loss/BC", ℒ_bc.item(), epoch)
            writer.add_scalar("Loss/IC", ℒ_ic.item(), epoch)
            writer.add_scalar("Loss/Total", loss.item(), epoch)
            writer.add_scalar("Metrics/L2", l2, epoch)
            writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

        # Comparison plots (only for 2D problems)
        if epoch % plot_interval == 0 and d == 2:
            ν_mid = sum(nu_range) / 2
            fig = create_comparison_plot(
                model,
                ν_mid,
                noise_args=test_noise_args,
                analytical_mean_solution=analytical_mean_solution,
                vartheta=vartheta,
                d=d,
                T=T,
                scaling=scaling,)
            if fig:
                writer.add_figure("Comparison", fig, epoch)
                plt.close(fig)

    # ───── Final evaluation + save ─────
    print("\n✔️  Training complete")

    if save_model:
        os.makedirs("models", exist_ok=True)  # Ensure models directory exists
        print("💾  Saving model ...")
        time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"{time_stamp}_{stamp}.pth"
        save_path = os.path.join("models", model_path)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "hyperparams": {
                    "spatial_dims": d,
                    "T": T,
                    "vartheta": vartheta,
                    "nu_range": nu_range,
                    "forcing_type": forcing_type,
                    "lambda_pde": λ_pde,
                    "lambda_bc": λ_bc,
                    "lambda_ic": λ_ic,
                    "hidden_dim": hidden_dim,
                    "num_layers": num_layers,
                    "lr": lr,
                    "m_samples": m_samples,
                    "epochs": epochs,
                },
            },
            save_path,
        )
        print(f"💾  Model saved: {save_path}")

    writer.close()
    return model


# ======================================================================
#                             Entrypoint
# ======================================================================

if __name__ == "__main__":
    # torch.manual_seed(42)
    # np.random.seed(42)
    # for forcing_type in ["linear", "exp_linear", "square"]:
    #     train_pinn(forcing_type=forcing_type, n_pde=10000)
    # train_pinn(analytical_mean_solution=time_shifted_gaussian_analytical_mean_solution,
    #            noise_function=time_dependant_gaussian_white_noise,
    #            forcing_type="square",
    #            n_pde=1000,
    #            spatial_dims=16,
    #            add_dims=4
    #            )
    train_pinn(
        noise_function=ou_process,
        analytical_mean_solution=ou_analytical_mean_solution,
        add_dims=5,
        forcing_type="linear",
        noise_boundaries=((0.5, 2.5), (0.1, 0.5), (0.1, 1.0)),
        spatial_dims=16
    )
