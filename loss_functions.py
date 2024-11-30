import torch
from torch import nn


class HeatDGMLoss(nn.Module):
    def __init__(self, lambda1: float, lambda2: float, lambda3: float = 1.0):
        super().__init__()
        self.lambda1 = lambda1  # Penalty for initial condition
        self.lambda2 = lambda2  # Penalty for boundary condition
        self.lambda3 = lambda3  # Penalty for regularization
        self.lambda_physics = 1.0  # Penalty for physics constraints

    def compute_gradients(self, u, var, grad_outputs=None):
        """Safely compute gradients with graph retention"""
        if grad_outputs is None:
            grad_outputs = torch.ones_like(u)
        return torch.autograd.grad(
            u, var,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0]

    def raw_residual(self, u, t, W_d, X):
        """Compute PDE residual with retained computation graph"""
        # Ensure X has requires_grad=True
        if not X.requires_grad:
            X.requires_grad_(True)

        # Time derivative
        u_t = self.compute_gradients(u, t)

        # First spatial derivatives
        u_x = torch.zeros_like(X)
        u_x_full = self.compute_gradients(u, X)  # Compute gradients for all dimensions
        for i in range(X.shape[1]):
            u_x[:, i:i+1] = u_x_full[:, i:i+1]  # Extract gradient for specific dimension

        # Second spatial derivatives (diagonal terms of the Hessian)
        u_xx = torch.zeros_like(X)
        for i in range(X.shape[1]):
            u_xx[:, i:i+1] = self.compute_gradients(u_x[:, i:i+1], X)[:, i:i+1]  # Extract diagonal

        # Laplacian
        u_laplace = torch.sum(u_xx, dim=1, keepdim=True)
        return u_t - 0.1*u_laplace - u * W_d

    def compute_boundary_error(self, ub, X):
        """Compute Neumann boundary error (normal derivative = 0)"""
        # Ensure X has requires_grad=True
        if not X.requires_grad:
            X.requires_grad_(True)

        # Compute gradients for all dimensions at once
        boundary_grad_full = self.compute_gradients(ub, X)

        # Extract and assemble the gradient for each dimension
        boundary_grad = torch.zeros_like(X)
        for i in range(X.shape[1]):
            boundary_grad[:, i:i+1] = boundary_grad_full[:, i:i+1]  # Extract specific dimension

        # Compute Neumann boundary condition error (normal derivative = 0)
        return self.lambda2 * torch.mean(boundary_grad**2)

    def compute_initial_error(self, u0, X):
        """Compute initial condition error with fixed target"""
        # Fixed target for initial condition
        target = torch.prod(torch.cos(X * torch.pi), dim=1, keepdim=True)
        return self.lambda1 * torch.mean((u0 - target)**2)

    def compute_regularization(self, u):
        """Compute regularization terms for solution stability"""
        eps = 1e-6
        # Penalize nonzero magnitude
        zero_penalty = self.lambda3 * torch.mean(u**2)
        # Penalize variance to control spurious oscillations
        variance_penalty = self.lambda3 * torch.mean((u - u.mean())**2)

        return zero_penalty, variance_penalty

    def compute_physics_constraints(self, u, t):
        """Compute physics-informed constraints for stochastic heat equation"""
        # Sort points by time for energy computation
        sorted_idx = torch.argsort(t.squeeze())
        t_sorted = t[sorted_idx]
        u_sorted = u[sorted_idx]

        # Compute energy (L2 norm) at each unique time
        unique_times, inverse_indices = torch.unique(t_sorted.squeeze(), return_inverse=True)

        # Skip if we don't have enough time points
        if len(unique_times) <= 1:
            return torch.tensor(0.0, device=u.device)

        # Compute energies for each time point
        energies = []
        for i in range(len(unique_times)):
            mask = inverse_indices == i
            u_t = u_sorted[mask]
            if len(u_t) > 0:  # Only compute if we have points at this time
                energy_t = torch.mean(u_t**2)
                energies.append(energy_t)

        if not energies:  # If we couldn't compute any energies
            return torch.tensor(0.0, device=u.device)

        energies = torch.stack(energies)

        # Energy decay violation in expectation
        energy_violations = torch.relu(energies[1:] - energies[:-1])
        energy_penalty = torch.mean(energy_violations)

        # Maximum principle violation in expectation
        # Use the first time point instead of exactly t=0
        first_time_mask = inverse_indices == 0
        if torch.any(first_time_mask):
            init_max = torch.max(torch.abs(u_sorted[first_time_mask]))
            max_violation = torch.mean(torch.relu(torch.abs(u) - init_max))
        else:
            max_violation = torch.tensor(0.0, device=u.device)

        # Moment bound violation
        second_moment = torch.mean(u**2)
        if torch.any(first_time_mask):
            init_second_moment = torch.mean(u_sorted[first_time_mask]**2)
            moment_violation = torch.relu(second_moment - init_second_moment)
        else:
            moment_violation = torch.tensor(0.0, device=u.device)

        return energy_penalty + max_violation + moment_violation

    def forward(self, u, ub, u0, t, W_d, X, X_bound, X_init):
        """Forward pass with additional physics constraints"""
        # Ensure proper shapes
        if u.dim() == 1:
            u = u.unsqueeze(1)
        if ub.dim() == 1:
            ub = ub.unsqueeze(1)
        if u0.dim() == 1:
            u0 = u0.unsqueeze(1)
        if W_d.dim() == 1:
            W_d = W_d.unsqueeze(1)

        # Compute all components
        residual_error = torch.mean(self.raw_residual(u, t, W_d, X)**2)
        initial_error = self.compute_initial_error(u0, X_init)
        boundary_error = self.compute_boundary_error(ub, X_bound)
        zero_penalty, variance_penalty = self.compute_regularization(u)

        return residual_error, initial_error, boundary_error, zero_penalty, variance_penalty


class HeatMIMLoss(nn.Module):
    def __init__(self, lambda1: float = 1.0):
        super().__init__()
        self.lambda1 = lambda1  # Penalty for zero solutions

    def compute_gradients(self, u, var, grad_outputs=None):
        """Fixed gradient computation"""
        if grad_outputs is None:
            # Ensure grad_outputs matches u's shape
            grad_outputs = torch.ones(u.shape, device=u.device, dtype=u.dtype)
        try:
            grads = torch.autograd.grad(
                u, var,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )
            return grads[0] if grads[0] is not None else torch.zeros_like(var)
        except RuntimeError as e:
            if "does not require grad" in str(e):
                return torch.zeros_like(var)
            raise e

    def mim_residual(self, u, t, W_d, X):
        """Fixed Laplacian computation"""
        # Time derivative
        u_t = self.compute_gradients(u, t)

        # First spatial derivatives
        u_x = self.compute_gradients(u, X)

        # Proper second derivatives computation
        u_xx = torch.zeros_like(u)  # Correct shape for Laplacian
        for i in range(X.shape[1]):
            u_x_i = u_x[:, i].unsqueeze(-1)  # Ensure proper shape
            u_xx_i = self.compute_gradients(u_x_i, X)
            if u_xx_i is not None:
                u_xx = u_xx + u_xx_i[:, i].unsqueeze(-1)  # Accumulate Laplacian terms

        # Proper residual with reduction
        residual = u_t - 0.1*u_xx - u * W_d
        return residual  # Apply reduction

    def raw_residual(self, u, t, W_d, X):
        """
        Compute residual as if it were DGM (without gradient tracking)
        for comparison purposes
        """
        # Time derivative
        u_t = self.compute_gradients(u, t)

        # First spatial derivatives
        u_x = self.compute_gradients(u, X)

        # Second spatial derivatives
        u_xx = self.compute_gradients(u_x, X)

        # Laplacian
        u_laplace = torch.sum(u_xx, dim=1, keepdim=True)

        dgm_residual = u_t - 0.1*u_laplace - u * W_d
        return dgm_residual

    def compute_auxiliary_error(self, p, u, X):
        """Compute error between auxiliary variable p and actual gradients"""
        u_x = self.compute_gradients(u, X)
        if u_x is None:
            return torch.tensor(0.0, device=X.device)
        return torch.mean((p - u_x)**2)

    def compute_regularization(self, u):
        """Improved numerical stability"""
        eps = 1e-4  # Larger epsilon for better stability
        # Use more stable computations
        magnitude = torch.mean(torch.abs(u) + eps)
        variance = torch.var(u) + eps

        # Clip penalties to prevent explosions
        zero_penalty = torch.clamp(self.lambda1 / magnitude, 0, 1e6)
        variance_penalty = torch.clamp(self.lambda1 / variance, 0, 1e6)
        return zero_penalty, variance_penalty

    def forward(self, u, p, t, W_d, X):
        """Added shape checks and better error handling"""
        # Validate inputs
        if not (u.requires_grad and X.requires_grad and t.requires_grad):
            raise ValueError("Inputs u, X, and t must require gradients")

        # Ensure proper shapes
        if u.dim() == 1:
            u = u.unsqueeze(1)
        if W_d.dim() == 1:
            W_d = W_d.unsqueeze(1)

        # Compute components with proper error handling
        try:
            residual_error = torch.mean(self.mim_residual(u, t, W_d, X)**2)
            auxiliary_error = self.compute_auxiliary_error(p, u, X)
            zero_penalty, variance_penalty = self.compute_regularization(u)
        except RuntimeError as e:
            print(f"Error in forward pass: {str(e)}")
            raise e

        return residual_error, auxiliary_error, zero_penalty, variance_penalty
