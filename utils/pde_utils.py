from __future__ import annotations
import numpy as np
import torch
from typing import Optional

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def time_shifted_gaussian_analytical_mean_solution(
    t: torch.Tensor,
    x: torch.Tensor,
    nu: torch.Tensor,
    vartheta: float = 1.0,
    noise_args: list[torch.Tensor] | None = None,
    d: int = 2,
    *,
    forcing: str = "linear",
) -> torch.Tensor:
    """
    Closed-form mean ⟨u(t,x)⟩ for separable forcing  G(x)·f(t).

    Shapes
    ------
    * t, x, nu broadcast across the batch.
    * x   : (*, d)
    * nu  : (*, 1)
    """
    if noise_args is None:
        raise ValueError("‘noise_args’ must be (μ, σ) tensors")

    mu, sigma = noise_args

    # Ensure 2-D tensors: (*, 1)
    def _to_2d(z: torch.Tensor) -> torch.Tensor:
        return z.to(DEVICE) if z.ndim == 2 else z.unsqueeze(1).to(DEVICE)

    t, nu, mu, sigma = map(_to_2d, (t, nu, mu, sigma))
    x = _to_2d(x.reshape(-1, d))

    # Spatial factor  G(x) = ∏ sin(π x_i)
    Gx = torch.prod(torch.sin(np.pi * x), dim=1, keepdim=True)

    lam = d * np.pi**2           # Laplacian eigenvalue
    k = nu * lam               # exponential decay rate
    # f = expected_forcing(t, mu, sigma, forcing)  # forcing in time

    # Time-convolution  ∫₀ᵗ e^{-k(t-s)} f(s) ds  – pre-derived primitives
    if forcing == "linear":
        integral = mu * (t / k - (1 - torch.exp(-k * t)) / k**2)
    elif forcing == "exp_linear":
        denom = (k - 1).clamp(min=1e-8)
        exp_term = torch.exp((k - 1) * t)
        integral = mu * torch.exp(-k * t) * (
            ((k - 1) * t - 1) * exp_term + 1
        ) / denom**2
    elif forcing == "square":
        term1 = mu**2 * (
            (t**2 / k) - (2 * t) / k**2 + 2 / k**3
            - 2 * torch.exp(-k * t) / k**3
        )
        term2 = sigma**2 * (
            t / k - (1 - torch.exp(-k * t)) / k**2
        )
        integral = term1 + term2
    else:  # pragma: no cover
        raise ValueError(f"Unsupported forcing: {forcing}")

    a_t = torch.exp(-k * t) + integral         # analytic coefficient
    return vartheta * Gx * a_t                 # final mean field


def compound_poisson_analytical_mean_solution(
    t: torch.Tensor,
    x: torch.Tensor,
    nu: torch.Tensor,
    *,
    noise_args: list[torch.Tensor] | None = None,
    vartheta: float = 1.0,
    d: int = 2,
    forcing: str = "linear",
) -> torch.Tensor:
    """
    Mean field ⟨u(t,x)⟩ for the compound-Poisson noise process (μ-term omitted).

    noise_args = [λ, m_J, σ_J]  – all (*,1) tensors broadcastable to t.
    """
    if noise_args is None or len(noise_args) != 3:
        raise ValueError("noise_args must be [λ, m_J, σ_J] tensors")

    λ, mJ, sJ = (z.to(DEVICE) if z.ndim == 2 else z.unsqueeze(1).to(DEVICE) for z in noise_args)
    t = t.to(DEVICE) if t.ndim == 2 else t.unsqueeze(1).to(DEVICE)
    nu = nu.to(DEVICE) if nu.ndim == 2 else nu.unsqueeze(1).to(DEVICE)
    x = x.reshape(-1, d).to(DEVICE)                # (*, d)

    # derived parameters
    α = λ * mJ
    β = λ * (sJ**2 + mJ**2)

    # spatial part  G(x) = ∏ sin(π x_i)
    Gx = torch.prod(torch.sin(torch.pi * x), dim=1, keepdim=True)

    a = nu * (torch.pi**2 * d)           # decay rate  a = ν π² d

    # ---------- time convolution ----------
    if forcing == "linear":
        integral = α * (t / a - (1 - torch.exp(-a * t)) / a**2)

    elif forcing == "exp_linear":
        denom = (a - 1).clamp(min=1e-8)          # avoid divide-by-zero
        exp_term = torch.exp((a - 1) * t)
        integral = α * torch.exp(-a * t) * (((a - 1) * t - 1) * exp_term + 1) / denom**2

    elif forcing == "square":
        term1 = α**2 * ((t**2 / a) - (2 * t) / a**2 + 2 / a**3 - 2 * torch.exp(-a * t) / a**3)
        term2 = β * (t / a - (1 - torch.exp(-a * t)) / a**2)
        integral = term1 + term2

    else:
        raise ValueError(f"Unsupported forcing: {forcing}")

    φ_t = torch.exp(-a * t) + integral
    return vartheta * Gx * φ_t.to(Gx.dtype)


def ou_analytical_mean_solution(
    t: torch.Tensor,
    x: torch.Tensor,
    nu: torch.Tensor,
    *,
    noise_args: list[torch.Tensor] | None = None,   # [θ, μ_OU, σ_OU]
    vartheta: float = 1.0,
    d: int = 2,
    forcing: str = "linear",
    xi_0: float = None,  # Initial value of OU process
) -> torch.Tensor:
    """
    Closed-form mean ⟨u(t,x)⟩ for OU noise process.

    OU process: dξ = θ(μ_OU - ξ)dt + σ_OU dW_t

    Parameters
    ----------
    noise_args : [θ, μ_OU, σ_OU] 
        - θ: mean reversion speed
        - μ_OU: long-term mean  
        - σ_OU: volatility
    xi_0 : float, optional
        Initial value of OU process. If None, assumes ξ(0) = μ_OU (equilibrium)
    """
    if noise_args is None or len(noise_args) != 3:
        raise ValueError("noise_args must be [θ, μ_OU, σ_OU]")

    θ, μ_ou, σ_ou = noise_args

    # Ensure consistent 2D tensor shapes and device
    def _to_2d_device(tensor):
        if tensor.ndim == 0:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.ndim == 1:
            tensor = tensor.unsqueeze(1)
        return tensor.to(DEVICE)

    t = _to_2d_device(t)
    nu = _to_2d_device(nu)
    θ = _to_2d_device(θ)
    μ_ou = _to_2d_device(μ_ou)
    σ_ou = _to_2d_device(σ_ou)

    # Handle spatial coordinates
    x = x.reshape(-1, d).to(DEVICE)

    # Set initial condition (default to equilibrium)
    if xi_0 is None:
        xi_0 = μ_ou  # Start at equilibrium
    else:
        xi_0 = torch.full_like(μ_ou, xi_0)

    # Spatial factor G(x) = ∏ sin(π x_i)
    Gx = torch.prod(torch.sin(np.pi * x), dim=1, keepdim=True)

    # PDE parameters
    lam = d * np.pi**2  # Laplacian eigenvalue
    a = nu * lam        # decay rate in PDE

    # Solve the ODE: φ'(t) + a*φ(t) = expected_forcing_t
    # Solution: φ(t) = e^(-at)[1 + ∫₀ᵗ e^(as) * expected_forcing_t(s) ds]

    # Compute the convolution integral ∫₀ᵗ e^(-a(t-s)) * expected_forcing_t(s) ds
    if forcing == "linear":
        # ∫₀ᵗ e^(-a(t-s)) * [μ_OU + (ξ_0 - μ_OU)e^(-θs)] ds
        # = μ_OU * ∫₀ᵗ e^(-a(t-s)) ds + (ξ_0 - μ_OU) * ∫₀ᵗ e^(-a(t-s))e^(-θs) ds
        # = μ_OU * (1 - e^(-at))/a + (ξ_0 - μ_OU) * e^(-at) * (e^((a-θ)t) - 1)/(a-θ)

        term1 = μ_ou * (1 - torch.exp(-a * t)) / a
        denom = (a - θ).clamp(min=1e-8)
        term2 = (xi_0 - μ_ou) * torch.exp(-a * t) * (torch.exp((a - θ) * t) - 1) / denom
        integral = term1 + term2

    elif forcing == "exp_linear":
        # More complex integral - needs careful derivation
        # ∫₀ᵗ e^(-a(t-s)) * e^(-s) * [μ_OU + (ξ_0 - μ_OU)e^(-θs)] ds

        term1 = μ_ou * torch.exp(-a * t) * (torch.exp((a - 1) * t) - 1) / (a - 1).clamp(min=1e-8)
        denom2 = (a - 1 - θ).clamp(min=1e-8)
        term2 = (xi_0 - μ_ou) * torch.exp(-a * t) * (torch.exp((a - 1 - θ) * t) - 1) / denom2
        integral = term1 + term2

    elif forcing == "square":
        # Exact solution for E[ξ²(s)] = μ² + σ²/(2θ)(1 - e^(-2θs))

        # Term 1: μ² contribution
        term1 = μ_ou**2 * (1 - torch.exp(-a * t)) / a

        # Term 2: constant variance contribution
        term2 = (σ_ou**2 / (2 * θ)) * (1 - torch.exp(-a * t)) / a

        # Term 3: time-dependent variance correction
        # Handle potential singularity when a ≈ 2θ
        alpha = a - 2 * θ
        is_singular = torch.abs(alpha) < 1e-6

        # For singular case: use limit = t
        # For regular case: use (e^(αt) - 1)/α
        correction = torch.where(
            is_singular,
            t,  # L'Hôpital's rule limit
            (torch.exp(alpha * t) - 1) / alpha
        )

        term3 = -(σ_ou**2 / (2 * θ)) * torch.exp(-a * t) * correction

        integral = term1 + term2 + term3
    else:
        raise ValueError(f"Unsupported forcing: {forcing}")

    # Final solution: φ(t) = e^(-at) + integral
    φ_t = torch.exp(-a * t) + integral

    return vartheta * Gx * φ_t
