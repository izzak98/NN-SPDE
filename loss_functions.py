import torch
from torch import nn


class HeatDgmLoss(nn.Module):
    def __init__(self, lambda1: float, lambda2: float, lambda3: float = 1.0):
        super().__init__()
        self.lambda1 = lambda1  # Penalty for initial condition
        self.lambda2 = lambda2  # Penalty for boundary condition
        self.lambda3 = lambda3  # Penalty for zero solutions

    def compute_gradients(self, u, var, grad_outputs=None):
        """Safely compute gradients with graph retention"""
        if grad_outputs is None:
            grad_outputs = torch.ones_like(u)
        try:
            return torch.autograd.grad(
                u, var,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )[0]
        except RuntimeError as e:
            if "does not require grad" in str(e):
                return torch.zeros_like(var)
            raise e

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

        if not boundary_grads:
            return torch.tensor(0.0, device=X.device)

        boundary_grad = torch.cat(boundary_grads, dim=1)
        return self.lambda2 * torch.mean(boundary_grad**2)

    def compute_initial_error(self, u0, X):
        """Compute initial condition error"""
        target = torch.prod(torch.sin(X * torch.pi), dim=1, keepdim=True)
        return self.lambda1 * torch.mean((u0 - target)**2)

    def compute_regularization(self, u):
        """Compute regularization terms"""
        eps = 1e-6
        magnitude = torch.mean(torch.abs(u))
        variance = torch.var(u)

        zero_penalty = self.lambda3 * (1.0 / (magnitude + eps))
        variance_penalty = self.lambda3 * (1.0 / (variance + eps))

        return zero_penalty, variance_penalty

    def forward(self, u, ub, u0, t, W_d, X, X_init):
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
        initial_error = self.compute_initial_error(u0, X_init)
        boundary_error = self.compute_boundary_error(ub, X)
        zero_penalty, variance_penalty = self.compute_regularization(u)

        return residual_error, initial_error, boundary_error, zero_penalty, variance_penalty


class HeatMIMLoss(nn.Module):
    def __init__(self, lambda1: float = 1.0, lambda_consistency: float = 1.0):
        super().__init__()
        self.lambda1 = lambda1  # Penalty for zero solutions
        # Penalty for consistency between u and p
        self.lambda_consistency = lambda_consistency

    def raw_residual(self, u, p, t, W_d, X):
        # Compute time derivative u_t = du/dt
        u_t = torch.autograd.grad(
            u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]

        # Compute the divergence of p (approximation of the Laplacian)
        div_p = torch.zeros_like(u)
        for i in range(X.shape[1]):
            p_x = torch.autograd.grad(
                p[:, i], X, grad_outputs=torch.ones_like(p[:, i]), create_graph=True)[0]
            div_p += p_x[:, i].unsqueeze(1)

        # Main MIM residual
        mim_residual = u_t - div_p - u * W_d

        # Compute dgm_residual without tracking gradients
        with torch.no_grad():
            u_x = torch.autograd.grad(
                u, X, grad_outputs=torch.ones_like(u), create_graph=True)[0]
            u_xx = torch.autograd.grad(
                u_x, X, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
            u_laplace = torch.sum(u_xx, dim=1, keepdim=True)
            dgm_residual = u_t - u_laplace - u * W_d

        return mim_residual, dgm_residual

    def forward(self, u, p, t, W_d, X):
        # Calculate MIM residual for optimization
        mim_residual, dgm_residual = self.raw_residual(u, p, t, W_d, X)
        residual_error = torch.mean(mim_residual**2)

        # Optional consistency penalty between u and p
        grad_u = torch.autograd.grad(
            u, X, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        consistency_error = torch.mean((grad_u - p)**2)

        # Optional zero-solution penalty (helps to avoid trivial solutions)
        magnitude = torch.mean(torch.abs(u))
        zero_penalty = self.lambda1 / (magnitude + 1e-6)

        # Return dgm_residual for evaluation purposes
        return torch.mean(dgm_residual**2), residual_error, consistency_error, zero_penalty


class BurgerMIMLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def raw_residual(self, u, p, t, W_d, X, nu, alpha):
        """
        Parameters:
        - u: torch.Tensor, the predicted solution by the neural network
        - p: torch.Tensor, the predicted gradient of u
        - t: torch.Tensor, time variable
        - W_d: torch.Tensor, space-time white noise
        - X: torch.Tensor, spatial variable in d dimensions
        - nu: torch.Tensor, viscosity coefficient
        - alpha: torch.Tensor, noise coefficient
        Returns:
        - residual: torch.Tensor, the raw PDE residual
        """
        # Compute time derivative u_t = du/dt
        u_t = torch.autograd.grad(
            u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]

        # Compute the divergence of p (which should equal Laplacian of u)
        div_p = torch.zeros_like(u)
        for i in range(X.shape[1]):  # Loop over spatial dimensions
            p_x = torch.autograd.grad(
                p[:, i], X, grad_outputs=torch.ones_like(p[:, i]), create_graph=True)[0]
            div_p += p_x[:, i].unsqueeze(1)

        # Compute convection term (u·p instead of u·∇u since p represents ∇u)
        convection_term = torch.sum(u * p, dim=1, keepdim=True)

        # Compute stochastic term
        stochastic_term = alpha * (u**X.shape[1]) * W_d

        # PDE residual (stochastic Burgers' equation)
        residual = u_t + convection_term - nu * div_p - stochastic_term

        # Compute the Laplacian (Δu)
        u_x = torch.autograd.grad(
            u, X, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(
            u_x, X, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_laplace = torch.sum(u_xx, dim=1, keepdim=True)
        dgm_residual = u_t + convection_term - nu * u_laplace - stochastic_term

        return residual, dgm_residual

    def forward(self, u, p, t, W_d, X, nu, alpha):
        """
        Parameters:
        - u: torch.Tensor, the predicted solution by the neural network
        - p: torch.Tensor, the predicted gradient of u
        - t: torch.Tensor, time variable
        - W_d: torch.Tensor, space-time white noise
        - X: torch.Tensor, spatial variable in d dimensions
        - nu: torch.Tensor, viscosity coefficient
        - alpha: torch.Tensor, noise coefficient
        Returns:
        - residual_error: PDE residual error
        - consistency_error: Error between predicted and computed gradients
        """
        # PDE residual error
        residual, dgm_residual = self.raw_residual(u, p, t, W_d, X, nu, alpha)
        residual_error = torch.mean(residual**2)
        dgm_residual_error = torch.mean(dgm_residual**2)

        # Consistency between u and p (p should equal ∇u)
        grad_u = torch.autograd.grad(
            u, X, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        consistency_error = torch.mean((grad_u - p)**2)

        return dgm_residual_error, residual_error, consistency_error


class BurgerDGMLoss(nn.Module):
    def __init__(self, lambda1: float, lambda2: float, lambda3: float):
        super().__init__()
        self.lambda1 = lambda1  # Penalty for initial condition
        self.lambda2 = lambda2  # Penalty for periodic boundary condition
        self.lambda3 = lambda3  # Penalty for periodic boundary derivatives

    def raw_residual(self, u, t, W_d, X, nu, alpha):
        """
        Parameters:
        - u: torch.Tensor, the predicted solution by the neural network
        - t: torch.Tensor, time variable
        - W_d: torch.Tensor, space-time white noise
        - X: torch.Tensor, spatial variable in d dimensions
        - nu: torch.Tensor, viscosity coefficient
        - alpha: torch.Tensor, noise coefficient
        Returns:
        - residual: torch.Tensor, the raw PDE residual
        """
        # Compute time derivative u_t = du/dt
        u_t = torch.autograd.grad(
            u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]

        # Compute spatial derivatives: first (u_x) and second derivatives (u_xx) using autograd
        u_x = torch.autograd.grad(
            u, X, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(
            u_x, X, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

        # Compute the Laplacian (Δu)
        u_laplace = torch.sum(u_xx, dim=1, keepdim=True)

        # Compute convection term (u·∇u)
        convection_term = torch.sum(u * u_x, dim=1, keepdim=True)

        # Compute stochastic term (σ(u) = αu^d where d is the number of spatial dimensions)
        stochastic_term = alpha * torch.pow(u, X.shape[1]) * W_d

        # PDE residual (stochastic Burgers' equation)
        residual = u_t + convection_term - nu * u_laplace - stochastic_term

        return residual

    def forward(self, u, u0, t, W_d, X, boundary_mask, nu, alpha):
        # PDE residual error
        residual = self.raw_residual(u, t, W_d, X, nu, alpha)
        residual_error = torch.mean(residual**2)

        # Initial condition error (only for points at t=0)
        t_zero_mask = t.abs() < 1e-6
        if t_zero_mask.any():
            initial_condition = torch.prod(
                torch.sin(X[t_zero_mask] * torch.pi), dim=1, keepdim=True)
            initial_error = self.lambda1 * \
                torch.mean((u[t_zero_mask] - initial_condition)**2)
        else:
            initial_error = torch.tensor(0.0, device=u.device)

        # Periodic boundary condition errors
        boundary_points = X[boundary_mask]
        u_boundary = u[boundary_mask]
        boundary_error = 0

        if len(boundary_points) > 0:  # Only proceed if we have boundary points
            for i in range(X.shape[1]):
                # Find points at x_i = 0 and x_i = 1
                mask_0 = boundary_points[:, i] == 0
                mask_1 = boundary_points[:, i] == 1

                if mask_0.any() and mask_1.any():
                    # Get the gradient for all boundary points at once
                    grad_u = torch.autograd.grad(u_boundary, X,
                                                 grad_outputs=torch.ones_like(
                                                     u_boundary),
                                                 create_graph=True)[0][boundary_mask]

                    points_0 = boundary_points[mask_0]
                    points_1 = boundary_points[mask_1]
                    u_0 = u_boundary[mask_0]
                    u_1 = u_boundary[mask_1]
                    grad_0 = grad_u[mask_0]
                    grad_1 = grad_u[mask_1]

                    # Create matrices for broadcasting comparison
                    points_0_expanded = points_0.unsqueeze(1)  # [N0, 1, d]
                    points_1_expanded = points_1.unsqueeze(0)  # [1, N1, d]

                    # Find matching pairs (ignore i-th dimension)
                    mask_dims = torch.ones(X.shape[1], dtype=torch.bool)
                    mask_dims[i] = False
                    matches = torch.all(
                        points_0_expanded[:, :,
                                          mask_dims] == points_1_expanded[:, :, mask_dims],
                        dim=-1
                    )  # [N0, N1]

                    # Compute errors for matching pairs
                    if matches.any():
                        match_indices = torch.where(matches)
                        u_diffs = u_0[match_indices[0]] - u_1[match_indices[1]]
                        grad_diffs = grad_0[match_indices[0]
                                            ] - grad_1[match_indices[1]]

                        boundary_error += self.lambda2 * torch.mean(u_diffs**2)
                        boundary_error += self.lambda3 * \
                            torch.mean(grad_diffs**2)

        return residual_error, initial_error, boundary_error
