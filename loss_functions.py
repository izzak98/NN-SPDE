import torch
from torch import nn

class HeatDgmLoss(nn.Module):
    def __init__(self, lambda1: float, lambda2: float, lambda3: float = 1.0):
        super().__init__()
        self.lambda1 = lambda1  # Penalty for initial condition
        self.lambda2 = lambda2  # Penalty for boundary condition
        self.lambda3 = lambda3  # Penalty for zero solutions

    def raw_residual(self, u, t, W_d, X):
        """
        Parameters:
        - u: torch.Tensor, the predicted solution by the neural network
        - t: torch.Tensor, time variable
        - W_d: torch.Tensor, space-time white noise
        - X: torch.Tensor, spatial variable in d dimensions
        Returns:
        - residual: torch.Tensor, the raw PDE residual
        """
        # Compute time derivative u_t = du/dt
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        
        # Compute spatial derivatives
        u_x = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, X, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        
        # Compute the Laplacian
        u_laplace = torch.sum(u_xx, dim=1)
        residual = u_t - u_laplace - u * W_d
        return residual

    def forward(self, u, u0, t, W_d, X, boundary_mask):
        # PDE residual
        residual = self.raw_residual(u, t, W_d, X)
        residual_error = torch.mean(residual**2)

        # Initial condition error
        initial_error = u0 - torch.prod(torch.sin(X * torch.pi), dim=1)
        initial_error = torch.mean(self.lambda1 * initial_error**2)

        # Boundary condition error
        u_boundary = u[boundary_mask]
        boundary_grad = torch.autograd.grad(u_boundary, X, grad_outputs=torch.ones_like(u_boundary), create_graph=True)[0]
        boundary_error = torch.mean(self.lambda2 * boundary_grad**2)

        # Zero-solution penalties
        magnitude = torch.mean(torch.abs(u))
        zero_penalty = self.lambda3 * (1.0 / (magnitude + 1e-6))  # Penalize small magnitudes
        
        # Add variance penalty to encourage non-constant solutions
        variance = torch.var(u)
        variance_penalty = self.lambda3 * (1.0 / (variance + 1e-6))  # Penalize low variance
        
        return residual_error, initial_error, boundary_error, zero_penalty, variance_penalty
    

class HeatMIMLoss(nn.Module):
    def __init__(self, lambda1: float = 1.0):
        super().__init__()
        self.lambda1 = lambda1  # Penalty for zero solutions

    def raw_resdiual(self, u, p, t, W_d, X):
        """
        Parameters:
        - u: torch.Tensor, the predicted solution by the neural network
        - p: torch.Tensor, the predicted gradient of u
        - t: torch.Tensor, time variable
        - W_d: torch.Tensor, space-time white noise
        - X: torch.Tensor, spatial variable in d dimensions
        Returns:
        - residual: torch.Tensor, the raw PDE residual
        """
        # Compute time derivative u_t = du/dt
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]

        # Compute the divergence of p (which should equal Laplacian of u)
        div_p = torch.zeros_like(u)
        for i in range(X.shape[1]):
            p_x = torch.autograd.grad(p[:, i], X, grad_outputs=torch.ones_like(p[:, i]), create_graph=True)[0]
            div_p += p_x[:, i].unsqueeze(1)
        
        # PDE residual (stochastic heat equation)
        mim_residual = u_t - div_p - u * W_d

        u_x = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, X, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_laplace = torch.sum(u_xx, dim=1, keepdim=True)
        dgm_residual = u_t - u_laplace - u * W_d

        return mim_residual, dgm_residual

    def forward(self, u, p, t, W_d, X):
        """
        Parameters:
        - u: torch.Tensor, the predicted solution by the neural network
        - p: torch.Tensor, the predicted gradient of u
        - t: torch.Tensor, time variable
        - W_d: torch.Tensor, space-time white noise
        - X: torch.Tensor, spatial variable in d dimensions
        Returns:
        - total_loss: scalar, total loss including PDE residual and consistency between u and p
        """


        # PDE residual (stochastic heat equation)
        residual, dgm_residual = self.raw_resdiual(u, p, t, W_d, X)
        residual_error = torch.mean(residual**2)
        dgm_residual_error = torch.mean(dgm_residual**2)

        # Consistency between u and p
        grad_u = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        consistency_error = torch.mean((grad_u - p)**2)

        # Total loss
     
        # Zero-solution penalties
        magnitude = torch.mean(torch.abs(u))
        zero_penalty = self.lambda1 * (1.0 / (magnitude + 1e-6))  # Penalize small magnitudes
        
        # Add variance penalty to encourage non-constant solutions
        variance = torch.var(u)
        variance_penalty = self.lambda1 * (1.0 / (variance + 1e-6))  # Penalize low variance

        return dgm_residual_error, residual_error, consistency_error, zero_penalty, variance_penalty

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
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]

        # Compute the divergence of p (which should equal Laplacian of u)
        div_p = torch.zeros_like(u)
        for i in range(X.shape[1]):  # Loop over spatial dimensions
            p_x = torch.autograd.grad(p[:, i], X, grad_outputs=torch.ones_like(p[:, i]), create_graph=True)[0]
            div_p += p_x[:, i].unsqueeze(1)

        # Compute convection term (u·p instead of u·∇u since p represents ∇u)
        convection_term = torch.sum(u * p, dim=1, keepdim=True)

        # Compute stochastic term
        stochastic_term = alpha * (u**X.shape[1]) * W_d
        
        # PDE residual (stochastic Burgers' equation)
        residual = u_t + convection_term - nu * div_p - stochastic_term

        # Compute the Laplacian (Δu)
        u_x = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, X, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
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
        grad_u = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u), create_graph=True)[0]
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
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
       
        # Compute spatial derivatives: first (u_x) and second derivatives (u_xx) using autograd
        u_x = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, X, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
       
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
            initial_condition = torch.prod(torch.sin(X[t_zero_mask] * torch.pi), dim=1, keepdim=True)
            initial_error = self.lambda1 * torch.mean((u[t_zero_mask] - initial_condition)**2)
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
                                            grad_outputs=torch.ones_like(u_boundary),
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
                        points_0_expanded[:, :, mask_dims] == points_1_expanded[:, :, mask_dims],
                        dim=-1
                    )  # [N0, N1]

                    # Compute errors for matching pairs
                    if matches.any():
                        match_indices = torch.where(matches)
                        u_diffs = u_0[match_indices[0]] - u_1[match_indices[1]]
                        grad_diffs = grad_0[match_indices[0]] - grad_1[match_indices[1]]
                        
                        boundary_error += self.lambda2 * torch.mean(u_diffs**2)
                        boundary_error += self.lambda3 * torch.mean(grad_diffs**2)

        return residual_error, initial_error, boundary_error


