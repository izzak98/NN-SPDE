import torch
from torch import nn

class HeatDgmLoss(nn.Module):
    def __init__(self, lambda1: float, lambda2: float):
        super().__init__()
        self.lambda1 = lambda1  # Penalty for initial condition
        self.lambda2 = lambda2  # Penalty for boundary condition

    def raw_resdiual(self, u, t, W_d, X):
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

        # Compute spatial derivatives: first (u_x) and second derivatives (u_xx) using autograd
        u_x = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, X, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

        # Compute the Laplacian (Δu)
        u_laplace = torch.sum(u_xx, dim=1)
        resdiual = u_t - u_laplace - u * W_d
        return resdiual

    def forward(self, u, u0, t, W_d, X, boundary_mask):
        """
        Parameters:
        - u: torch.Tensor, the predicted solution by the neural network
        - u0: torch.Tensor, the predicted initial condition solution
        - t: torch.Tensor, time variable
        - W_d: torch.Tensor, space-time white noise
        - X: torch.Tensor, spatial variable in d dimensions
        - boundary_mask: torch.Tensor, boolean mask to indicate points on the boundary

        Returns:
        - total_loss: scalar, total loss including PDE residual, initial condition error, and boundary condition error
        """
        residual = self.raw_resdiual(u, t, W_d, X)
        residual_error = torch.mean(residual**2)

        # Initial condition error
        initial_error = u0 - torch.prod(torch.sin(X * torch.pi), dim=1)
        initial_error = torch.mean(self.lambda1 * initial_error**2)

        # Boundary condition error (Neumann boundary condition)
        u_boundary = u[boundary_mask]  # Only select points on the boundary
        boundary_grad = torch.autograd.grad(u_boundary, X, grad_outputs=torch.ones_like(u_boundary), create_graph=True)[0]
        boundary_error = torch.mean(self.lambda2 * boundary_grad**2)

        # Total loss

        return residual_error, initial_error, boundary_error
    


class HeatMIMLoss(nn.Module):
    def __init__(self):
        super().__init__()

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
        residual = u_t - div_p - u * W_d
        return residual

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
        residual = self.raw_resdiual(u, p, t, W_d, X)
        residual_error = torch.mean(residual**2)

        # Consistency between u and p
        grad_u = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        consistency_error = torch.mean((grad_u - p)**2)

        # Total loss
     

        return residual_error, consistency_error

if __name__ == "__main__":
    from models import DGM, HeatMIM
    torch.manual_seed(0)

    # Dimensions for dummy data
    N = 64  # Number of grid points (spatial)
    d = 2   # Number of spatial dimensions
    T = 1.0  # Total time duration
    dt = 0.01  # Time step
    time_steps = int(T / dt)  # Number of time steps

    # Generate random spatial coordinates (X) and time (t) with requires_grad=True
    X = torch.rand((N, d), dtype=torch.float32, requires_grad=True)  # Random spatial coordinates in [0, 1]^d
    t = torch.rand((N, 1), dtype=torch.float32, requires_grad=True)  # Random time in [0, 1]
    t_0 = torch.zeros((N, 1), dtype=torch.float32)  # Initial time

    # Initialize the neural network
    input_dim = d
    # Generate space-time white noise 'W_d' for testing purposes (size matching 'u')
    W_d = torch.randn((N, 1), dtype=torch.float32)  # Random space-time white noise
    # Create a boundary mask for Neumann boundary conditions (for simplicity, assume half are boundary points)
    boundary_mask = torch.zeros(N, dtype=torch.bool)
    boundary_mask[:N // 2] = 1  # Mark first half as boundary points

    # Initialize the DGM loss function with lambda1 and lambda2
    lambda1 = 1.0
    lambda2 = 1.0
    dgm_heat_loss = HeatDgmLoss(lambda1, lambda2)
    mim_heat_loss = HeatMIMLoss()

    dgm = DGM(input_dim, [128, 128], 'relu')
    # Forward pass through the neural network (u is now a function of both X and t)
    dgm_u = dgm(X, t)
    dgm_u0 = dgm(X, t_0)

    mim = HeatMIM(input_dim, [128, 128], [128, 128], 'relu')
    # Forward pass through the neural network (u and p are now functions of both X and t)
    mim_u, mim_p = mim(X, t)

    # Test the loss function
    dgm_loss = dgm_heat_loss(u=dgm_u, u0=dgm_u0, t=t, W_d=W_d, X=X, boundary_mask=boundary_mask)
    mim_loss = mim_heat_loss(u=mim_u, p=mim_p, t=t, W_d=W_d, X=X)

    # Print the result
    print("DGM Loss:", dgm_loss.item())
    print("MIM Loss:", mim_loss.item())
