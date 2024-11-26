import torch
from torch import nn


class HeatDGMLoss(nn.Module):
    def __init__(self, lambda1: float, lambda2: float, lambda3: float = 1.0):
        super().__init__()
        self.lambda1 = lambda1  # Penalty for initial condition
        self.lambda2 = lambda2  # Penalty for boundary condition
        self.lambda3 = lambda3  # Penalty for zero solutions

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
        # Time derivative
        u_t = self.compute_gradients(u, t)

        # First spatial derivatives
        u_x = self.compute_gradients(u, X)

        # Second spatial derivatives
        u_xx = torch.zeros_like(u_x)
        for i in range(X.shape[1]):
            u_xx_i = self.compute_gradients(u_x[:, i], X)
            if u_xx_i is not None:
                u_xx[:, i] = u_xx_i[:, i]

        # Laplacian
        u_laplace = torch.sum(u_xx, dim=1, keepdim=True)
        return u_t - u_laplace - u * W_d

    def compute_boundary_error(self, ub, X):
        """Compute boundary error with retained graph"""
        boundary_grads = []
        for i in range(X.shape[1]):
            grad_i = self.compute_gradients(ub, X)
            if grad_i is not None:
                boundary_grads.append(grad_i[:, i:i+1])

        boundary_grad = torch.cat(boundary_grads, dim=1)
        return self.lambda2 * torch.mean(boundary_grad**2)

    def compute_initial_error(self, u0, X):
        """Compute initial condition error"""
        target = torch.prod(torch.cos(X * torch.pi), dim=1, keepdim=True)
        return self.lambda1 * torch.mean((u0 - target)**2)

    def compute_regularization(self, u):
        """Compute regularization terms"""
        eps = 1e-6
        magnitude = torch.mean(torch.abs(u))
        variance = torch.var(u)

        zero_penalty = self.lambda3 * (1.0 / (magnitude + eps))
        variance_penalty = self.lambda3 * (1.0 / (variance + eps))

        return zero_penalty, variance_penalty

    def forward(self, u, ub, u0, t, W_d, X, X_bound, X_init):
        """Forward pass with proper shape handling and graph retention"""
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
        # initial_error = self.compute_initial_error(u0, X_init)
        boundary_error = self.compute_boundary_error(ub, X_bound)
        zero_penalty, variance_penalty = self.compute_regularization(u)

        return residual_error, boundary_error, zero_penalty, variance_penalty


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
        residual = u_t - u_xx - u * W_d
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

        dgm_residual = u_t - u_laplace - u * W_d
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


class BurgerMIMLoss(nn.Module):
    def __init__(self, lambda1: float = 1.0):
        super().__init__()
        self.lambda1 = lambda1  # Penalty for zero solutions

    def compute_gradients(self, u, var, grad_outputs=None):
        """Compute gradients with proper error handling"""
        if grad_outputs is None:
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

    def compute_advection(self, u, u_x):
        """Compute the nonlinear advection term u·∇u"""
        # For each spatial dimension, compute u * du/dx_i and sum
        advection = torch.sum(u * u_x, dim=1, keepdim=True)
        return advection

    def raw_residual(self, u, t, W_d, X, nu, alpha):
        """Compute the Burgers equation residual with stochastic forcing"""
        # Time derivative
        u_t = self.compute_gradients(u, t)

        # First spatial derivatives
        u_x = self.compute_gradients(u, X)

        # Compute advection term
        advection = self.compute_advection(u, u_x)

        # Compute Laplacian (sum of second derivatives)
        u_xx = torch.zeros_like(u)
        for i in range(X.shape[1]):
            u_x_i = u_x[:, i].unsqueeze(-1)
            u_xx_i = self.compute_gradients(u_x_i, X)
            if u_xx_i is not None:
                u_xx = u_xx + u_xx_i[:, i].unsqueeze(-1)

        # Number of spatial dimensions for u^n term
        n = X.shape[1]

        # Compute residual: du/dt + u·∇u - ν∆u - αu^n·dW
        residual = u_t + advection - nu * u_xx - alpha * (u**n) * W_d
        return torch.mean(residual**2, dim=0)

    def dgm_residual(self, u, t, W_d, X, nu, alpha):
        """
        Compute residual without gradient tracking for comparison
        """
        with torch.no_grad():
            # Time derivative
            u_t = torch.autograd.grad(
                u, t,
                grad_outputs=torch.ones_like(u),
                create_graph=False,
                retain_graph=True)[0]

            # First spatial derivatives
            u_x = torch.autograd.grad(
                u, X,
                grad_outputs=torch.ones_like(u),
                create_graph=True,
                retain_graph=True)[0]

            # Advection term
            advection = torch.sum(u * u_x, dim=1, keepdim=True)

            # Second spatial derivatives
            u_xx = torch.autograd.grad(
                u_x, X,
                grad_outputs=torch.ones_like(u_x),
                create_graph=False,
                retain_graph=True)[0]

            # Laplacian
            u_laplace = torch.sum(u_xx, dim=1, keepdim=True)

            # Number of spatial dimensions
            n = X.shape[1]

            dgm_residual = u_t + advection - nu * u_laplace - alpha * (u**n) * W_d
        return torch.mean(dgm_residual**2)

    def compute_auxiliary_error(self, p, u, X):
        """Compute error between auxiliary variable p and actual gradients"""
        u_x = self.compute_gradients(u, X)
        if u_x is None:
            return torch.tensor(0.0, device=X.device)
        return torch.mean((p - u_x)**2)

    def compute_regularization(self, u):
        """Compute regularization terms for numerical stability"""
        eps = 1e-4
        magnitude = torch.mean(torch.abs(u) + eps)
        variance = torch.var(u) + eps

        zero_penalty = torch.clamp(self.lambda1 / magnitude, 0, 1e6)
        variance_penalty = torch.clamp(self.lambda1 / variance, 0, 1e6)
        return zero_penalty, variance_penalty

    def forward(self, u, p, t, W_d, X, nu, alpha):
        """Forward pass with input validation"""
        # Validate inputs
        if not (u.requires_grad and X.requires_grad and t.requires_grad):
            raise ValueError("Inputs u, X, and t must require gradients")

        # Ensure proper shapes
        if u.dim() == 1:
            u = u.unsqueeze(1)
        if W_d.dim() == 1:
            W_d = W_d.unsqueeze(1)

        residual_error = torch.mean(self.raw_residual(u, t, W_d, X, nu, alpha))
        auxiliary_error = self.compute_auxiliary_error(p, u, X)
        zero_penalty, variance_penalty = self.compute_regularization(u)
        dgm_style_residual = self.dgm_residual(u, t, W_d, X, nu, alpha)

        return dgm_style_residual, residual_error, auxiliary_error, zero_penalty, variance_penalty
