import torch
import numpy as np
import time


def shifted_gaussian_white_noise(t: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, seed: int = None):
    if seed is not None:
        torch.manual_seed(seed)
    noise = torch.randn_like(t) * torch.sqrt(t) * sigma + mu
    return noise


def time_dependant_gaussian_white_noise(t: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, seed: int = None):
    if seed is not None:
        torch.manual_seed(seed)
    noise = torch.randn_like(t) * torch.sqrt(t) * sigma + mu * t
    return noise  # Add a dimension to match the shape of t


def compound_poisson_process(
    t: torch.Tensor,
    lab: torch.Tensor,           # λ, event rate (shape: (*, 1))
    mu_j: torch.Tensor,          # μ_J, mean jump size (shape: (*, 1))
    sigma_j: torch.Tensor,       # σ_J, std dev of jumps (shape: (*, 1))
    seed: int = None
) -> torch.Tensor:
    if seed is not None:
        torch.manual_seed(seed)

    N = t.shape[0]
    t = t.view(-1, 1)
    lab = lab.view(-1, 1)
    mu_j = mu_j.view(-1, 1)
    sigma_j = sigma_j.view(-1, 1)

    # Sample number of jumps N_t ~ Poisson(λ t)
    pois_rate = lab * t
    n_jumps = torch.poisson(pois_rate).long().squeeze()  # (N,)

    # Total number of jumps needed
    total_jumps = n_jumps.sum().item()

    if total_jumps == 0:
        return torch.zeros(N, device=t.device)

    # Create indices for which sample each jump belongs to
    sample_indices = torch.repeat_interleave(torch.arange(N, device=t.device), n_jumps)

    # Generate all jumps at once
    mu_expanded = mu_j.squeeze()[sample_indices]
    sigma_expanded = sigma_j.squeeze()[sample_indices]
    jumps = torch.randn(total_jumps, device=t.device) * sigma_expanded + mu_expanded

    # Sum jumps for each sample using scatter_add
    noise = torch.zeros(N, device=t.device)
    noise.scatter_add_(0, sample_indices, jumps)

    return noise.unsqueeze(1)


def ou_process(
    t: torch.Tensor,
    theta: torch.Tensor,
    mu_ou: torch.Tensor,
    sigma_ou: torch.Tensor,
    x0: float | torch.Tensor = None,  # Changed default
    seed: int | None = None,
) -> torch.Tensor:
    randn = torch.randn_like(t)

    # ensure column tensors
    t, theta, mu_ou, sigma_ou = (z if z.ndim == 2 else z.view(-1, 1)
                                 for z in (t, theta, mu_ou, sigma_ou))

    # Default to equilibrium start
    if x0 is None:
        x0 = mu_ou  # Start at equilibrium!

    exp_term = torch.exp(-theta * t)
    mean = x0 * exp_term + mu_ou * (1.0 - exp_term)
    var_coeff = torch.sqrt((1.0 - exp_term ** 2) / (2.0 * theta))
    return mean + sigma_ou * var_coeff * randn


def sample_points(n_points, d=2, T=1.0, nu_range=[0.01, 0.1]):
    """Sample random points in [0,T] x [0,1]^d with random nu"""
    t = torch.rand(n_points, 1) * T
    x = torch.rand(n_points, d)
    nu = torch.rand(n_points, 1) * (nu_range[1] - nu_range[0]) + nu_range[0]
    return torch.cat([t, x, nu], dim=1)


def sample_boundary_points(n_points, d=2, T=1.0, nu_range=[0.01, 0.1]):
    """Sample points on the boundary of [0,T] x [0,1]^d with random nu"""
    points = []

    # For each spatial dimension, create boundary points
    for dim in range(d):
        # Lower boundary (x_i = 0)
        n_dim_points = n_points // (2 * d)
        t = torch.rand(n_dim_points, 1) * T
        x = torch.rand(n_dim_points, d)
        x[:, dim] = 0.0
        nu = torch.rand(n_dim_points, 1) * (nu_range[1] - nu_range[0]) + nu_range[0]
        points.append(torch.cat([t, x, nu], dim=1))

        # Upper boundary (x_i = 1)
        t = torch.rand(n_dim_points, 1) * T
        x = torch.rand(n_dim_points, d)
        x[:, dim] = 1.0
        nu = torch.rand(n_dim_points, 1) * (nu_range[1] - nu_range[0]) + nu_range[0]
        points.append(torch.cat([t, x, nu], dim=1))

    return torch.cat(points, dim=0)


def sample_initial_points(n_points, d=2, vartheta=1.0, nu_range=[0.01, 0.1]):
    """Sample points at t=0 with initial condition and random nu"""
    t = torch.zeros(n_points, 1)
    x = torch.rand(n_points, d)
    nu = torch.rand(n_points, 1) * (nu_range[1] - nu_range[0]) + nu_range[0]

    # Compute initial condition: vartheta * prod(sin(pi * x_i))
    u_initial = torch.ones(n_points, 1) * vartheta
    for i in range(d):
        u_initial *= torch.sin(np.pi * x[:, i:i+1])

    return torch.cat([t, x, nu], dim=1), u_initial


def sample_noise_args(n_points: int, n_args: int, boundaries=None, device=None):
    if boundaries is None:
        boundaries = [(0, 1)]*n_args
    args = []
    for bound in boundaries:
        sub_arg = torch.rand(n_points, 1) * (bound[1] - bound[0]) + bound[0]
        if device is not None:
            sub_arg = sub_arg.to(device)
        args.append(sub_arg)
    return args


if __name__ == "__main__":
    N = 100_000
    t = torch.rand(N, 1)
    theta = torch.rand(N, 1)*1.5 + 0.5         # (0.5, 2)
    mu_ou = torch.rand(N, 1)*0.4 + 0.1         # (0.1, 0.5)
    sigma_ou = torch.rand(N, 1)*0.9 + 0.1       # (0.1, 1.0)

    ξ = ou_process(t, theta, mu_ou, sigma_ou)
    # empirical check: mean should be close to μ_OU*(1-e^{-θt})
    print("mean error", (ξ.mean() - (mu_ou*(1-torch.exp(-theta*t))).mean()).abs().item())
