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
import json
from datetime import datetime


# ───────────────────── 3rd-party imports ──────────────────────
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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
from utils.lap_utils import UltraLaplacianComputer

# ────────────────────────── device ────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Using device: {DEVICE}")


# ======================================================================
#                            Loss functions
# ======================================================================

_laplacian = UltraLaplacianComputer()


def compute_pde_residual(
    model: nn.Module,
    pts: torch.Tensor,
    w_t: torch.Tensor,
    vartheta: float,
    noise_args: list[torch.Tensor],
    *,
    forcing_type: str = "linear",
    scaling: float = 1.0,
    buffer: torch.Tensor = None,  # Optional pre-allocated buffer
) -> torch.Tensor:
    """
    Aggressively memory-optimized version using pre-allocated buffers.
    """
    pts.requires_grad_(True)

    # Use pre-allocated buffer for model input if provided
    if buffer is not None and noise_args:
        buffer[:, :pts.shape[1]] = pts
        start_idx = pts.shape[1]
        for noise_arg in noise_args:
            end_idx = start_idx + noise_arg.shape[1]
            buffer[:, start_idx:end_idx] = noise_arg
            start_idx = end_idx
        model_in = buffer[:, :start_idx]
    else:
        model_in = torch.cat([pts, *noise_args], dim=1) if noise_args else pts

    u = model(model_in)
    if scaling != 1.0:
        u.mul_(scaling)  # in-place scaling

    # Extract components (views, no memory allocation)
    nu = pts[:, -1:]
    d = pts.shape[1] - 2
    x_space = pts[:, 1: 1 + d]
    t = pts[:, 0:1]

    # Efficient gradient computation
    grad_outputs = torch.ones_like(u)
    du_dt = torch.autograd.grad(u, pts, grad_outputs, create_graph=True)[0][:, 0:1]

    lap_u = _laplacian(u, pts, d)

    # Efficient forcing computation using temporary tensor
    temp = x_space * np.pi
    torch.sin_(temp)
    Gx = torch.prod(temp, dim=1, keepdim=True)

    # Build forcing term step by step to minimize allocations
    if forcing_type == "linear":
        forcing = w_t * vartheta
    elif forcing_type == "exp_linear":
        forcing = torch.exp(-t)
        forcing *= w_t * vartheta
    elif forcing_type == "square":
        forcing = w_t * w_t * vartheta
    else:
        raise ValueError(f"Unsupported forcing type: {forcing_type}")

    forcing *= Gx

    # Final computation
    residual = du_dt - nu * lap_u - forcing
    if scaling != 1.0:
        residual.div_(scaling)

    return residual

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
        pde_loss += (res.pow(2)).mean()
    pde_loss /= m_samples

    # ───── Boundary (Dirichlet) loss : u = 0 on ∂Ω ─────
    bc_pts = sample_boundary_points(n_bc, d, T, nu_range).to(DEVICE)
    bc_noise_args = [na[: bc_pts.shape[0]] for na in noise_args]
    bc_pred = model(torch.cat([bc_pts, *bc_noise_args], dim=1))
    bc_loss = (bc_pred.pow(2)/scaling).mean()

    # ───── Initial-time loss : u(0,x) = vartheta·G(x) ─────
    ic_pts, ic_true = sample_initial_points(n_ic, d, vartheta, nu_range)
    ic_pts, ic_true = ic_pts.to(DEVICE), ic_true.to(DEVICE)
    ic_noise_args = [na[: n_ic] for na in noise_args]
    ic_pred = model(torch.cat([ic_pts, *ic_noise_args], dim=1))
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
    run_path: str = "runs",
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
    run_path : str, default="runs"
        Directory to save TensorBoard logs and model checkpoints

    Returns
    -------
    nn.Module
        Trained PINN model
    """
    # ───── Problem setup ─────
    d = spatial_dims
    scaling = d ** (-np.log(10) / np.log(2)) * 10  # scale output to for better training stability

    # Compute vartheta
    vartheta = np.sqrt(d) * np.log10(np.exp(d))

    # Validate forcing type
    valid_forcing_types = {"linear", "exp_linear", "square"}
    if forcing_type not in valid_forcing_types:
        raise ValueError(f"forcing_type must be one of {valid_forcing_types}, got {forcing_type}")

    # ───── Network ─────
    input_dim = d + add_dims
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

    tb_dir = os.path.join(
        run_path, f"{datetime.now():%Y%m%d_%H%M%S}_pinn_heat_{d}d_{forcing_type}_mc_samples_{m_samples}_noise_{noise_name}")
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

def pinn_scaling(d):
    """
    Compute scaling rules for PINNs as functions of dimension `d`.

    Returns:
        p (int): neurons per layer (width)
        L (int): number of layers (depth)
        n (int): number of collocation points
    """
    # Width scales like 128 * sqrt(d / 2)
    p = int(np.round(128 * np.sqrt(d / 2)))

    # Depth grows log-linearly: 6 + 2 * log2(d / 2)
    L = int(np.round(6 + 2 * np.log2(d / 2)))

    # Number of collocation points grows quadratically: 1000 * (d / 2)^2
    n = min(int(np.round(1000 * (d / 2) ** 2)), 5000)

    return p, L, n


if __name__ == "__main__":
    # Add system arguments for flexibility
    import argparse
    parser = argparse.ArgumentParser(
        description="Train a PINN for a heat-type PDE with stochastic forcing.")
    parser.add_argument("--spatial_dims", type=int, default=8,
                        help="Number of spatial dimensions (default: 2)")
    parser.add_argument("--forcing_type", type=str, default="linear", choices=["linear", "exp_linear", "square"],
                        help="Type of forcing function (default: linear)")
    parser.add_argument("--noise_function", type=str, default="ou_process", choices=["time_dependant_gaussian_white_noise", "ou_process", "compound_poisson_process"],
                        help="Noise function to use (default: ou_process)")
    parser.add_argument("--epochs", type=int, default=10_000,
                        help="Number of training epochs (default: 10_000)")
    parser.add_argument("--n_pde", type=int, default=1000,
                        help="Number of interior points for PDE residual (default: 1000)")
    parser.add_argument("--n_bc", type=int, default=100,
                        help="Number of boundary points (default: 100)")
    parser.add_argument("--n_ic", type=int, default=100,
                        help="Number of initial condition points (default: 100)")
    parser.add_argument("--n_test", type=int, default=10000,
                        help="Number of test points for L2 error computation (default: 10000)")
    parser.add_argument("--m_samples", type=int, default=10,
                        help="Number of Monte Carlo samples for PDE loss computation (default: 10)")
    parser.add_argument("--save_path", type=str, default="runs",
                        help="Directory to save TensorBoard logs (default: runs)")

    # Parse arguments
    args = parser.parse_args()
    # Select noise function based on argument
    noise_function_map = {
        "time_dependant_gaussian_white_noise": time_dependant_gaussian_white_noise,
        "ou_process": ou_process,
        "compound_poisson_process": compound_poisson_process,
    }
    analytical_solution_map = {
        "time_dependant_gaussian_white_noise": time_shifted_gaussian_analytical_mean_solution,
        "ou_process": ou_analytical_mean_solution,
        "compound_poisson_process": compound_poisson_analytical_mean_solution,
    }
    if args.noise_function == "time_dependant_gaussian_white_noise":
        add_dims = 4  # μ, σ, ν
        noise_boundaries = ((0.1, 0.5), (0.1, 1.0))
    elif args.noise_function == "ou_process":
        add_dims = 5
        noise_boundaries = ((0.5, 2.0), (0.1, 0.5), (0.1, 1.0))
    elif args.noise_function == "compound_poisson_process":
        add_dims = 5
        noise_boundaries = ((0.1, 5.0), (0.1, 0.5), (0.1, 1.0))
    else:  # pragma: no cover
        raise ValueError(f"Unsupported noise function: {args.noise_function}")

    noise_function = noise_function_map[args.noise_function]
    analytical_mean_solution = analytical_solution_map[args.noise_function]

    # Compute scaling rules based on spatial dimensions
    hidden_dim, num_layers, n_pde = pinn_scaling(args.spatial_dims)
    n_ic = n_bc = n_pde // 10  # 10% of PDE points for IC and BC

    # Confirm settings for the user
    print(f"🔊 Using noise function: {args.noise_function}"
          f" ({noise_function.__name__})")
    print(f"🔧 Using analytical mean solution: {analytical_mean_solution.__name__}")
    print(f"🔧 Using noise boundaries: {noise_boundaries}")
    print(f"🔧 Using add_dims: {add_dims} ({args.noise_function})")
    print(f"🔧 Using spatial_dims: {args.spatial_dims} ({args.noise_function})")
    print(f"🔧 Using forcing_type: {args.forcing_type}")
    print(f"🔧 Using epochs: {args.epochs}")
    print(f"🔧 Using n_pde: {n_pde}")
    print(f"🔧 Using n_ic: {n_ic}")
    print(f"🔧 Using n_bc: {n_bc}")
    print(f"🔧 Using n_test: {args.n_test}")
    print(f"🔧 Using m_samples: {args.m_samples}")
    print(f"🔧 Using hidden_dim: {hidden_dim}")
    print(f"🔧 Using num_layers: {num_layers}")
    print(f"🔧 Using add_dims: {add_dims}")

    # Train the PINN
    model = train_pinn(
        noise_function=noise_function,
        analytical_mean_solution=analytical_mean_solution,
        spatial_dims=args.spatial_dims,
        add_dims=add_dims,
        forcing_type=args.forcing_type,
        lambda_pde=1.0,
        lambda_bc=10.0,
        lambda_ic=10.0,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        T=1.0,  # Default final time
        lr=1e-3,  # Learning rate
        m_samples=args.m_samples,
        epochs=args.epochs,
        nu_range=(0.01, 0.1),  # Default viscosity range
        scheduler_gamma=0.9995,
        n_pde=n_pde,
        n_bc=n_bc,
        n_ic=n_ic,
        n_test=args.n_test,
        log_interval=10,
        tensorboard_interval=100,
        plot_interval=500,
        noise_boundaries=noise_boundaries,
        save_model=False,  # Set to True if you want to save the model
        run_path=args.save_path,
    )
    print("🏁 Training complete. Model ready for evaluation.")
