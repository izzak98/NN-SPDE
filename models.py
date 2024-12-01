import torch
from torch import nn

from utils.model_utils import create_fc_layers


class DGM(nn.Module):
    def __init__(self,
                 input_dims: int,
                 hidden_dims: list,
                 dgm_dims: int,
                 n_dgm_layers: int,
                 hidden_activation: str,
                 output_activation: str):
        super(DGM, self).__init__()
        input_dims += 1
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.dgm_dims = dgm_dims
        self.n_dgm_layers = n_dgm_layers
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        layers = create_fc_layers(
            input_dims, hidden_dims, hidden_activation, dgm_dims,
            n_dgm_layers, output_activation)
        self.input_layer, self.hidden_layers, self.dgm_layers, self.output_layer = layers

    def forward(self, t, *args):  # -> Any:
        # shapes:
        # t: (batch_size, 1)
        # x: (batch_size, dims)
        x = torch.cat(args, dim=1)
        inps = torch.cat([t, x], dim=1)
        input_x = self.input_layer(inps)

        if self.hidden_layers:
            for layer in self.hidden_layers:
                input_x = layer(input_x)

        if self.dgm_layers:
            S1 = input_x
            S = input_x
            for layer in self.dgm_layers:
                S = layer(input_x, S, S1)
            input_x = S

        return self.output_layer(input_x)


class HeatMIM(nn.Module):
    def __init__(self,
                 input_dims: int,
                 hidden_dims: list,
                 dgm_dims: int,
                 n_dgm_layers: int,
                 hidden_activation: str,
                 output_activation: str):
        super(HeatMIM, self).__init__()
        input_dims += 1  # Add time dimension
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.dgm_dims = dgm_dims
        self.n_dgm_layers = n_dgm_layers
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        # Create network layers
        layers = create_fc_layers(
            input_dims, hidden_dims, hidden_activation, dgm_dims,
            n_dgm_layers, output_activation, output_dim=2)
        self.input_layer, self.hidden_layers, self.dgm_layers, self.output_layer = layers

    def initial_conditions(self, x):
        """Compute initial condition: prod(cos(πx_i))"""
        return torch.prod(torch.cos(torch.pi * x), dim=1, keepdim=True)

    def enforce_neumann_boundary(self, p_theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Enforce zero Neumann boundary conditions on the unit hypercube
        Args:
            p_theta: Network output for derivatives (batch_size, dims)
            x: Spatial coordinates (batch_size, dims)
        """
        # Distance to boundary for unit hypercube
        d_left = x
        d_right = 1 - x
        L_N = torch.min(torch.min(d_left, d_right), dim=1, keepdim=True)[0]

        # Compute gradient of L_N
        x.requires_grad_(True)
        L_N_grad = torch.autograd.grad(L_N.sum(), x, create_graph=True)[0]

        # For zero Neumann condition, G_N = 0
        G_N = torch.zeros_like(p_theta)

        # Normal vectors (simpler for hypercube)
        # Will be unit vectors pointing outward at boundaries
        nu = torch.where(x < 0.5, -1.0, 1.0)

        # Compute boundary modification terms
        N_star_dot_nu = torch.sum(p_theta * nu, dim=1, keepdim=True)
        L_N_grad_dot_nu = torch.sum(L_N_grad * nu, dim=1, keepdim=True)

        # Compute modification factor
        F_N = (-N_star_dot_nu) / (L_N_grad_dot_nu + 1e-8)

        # Return modified p_theta
        return p_theta + F_N * L_N_grad

    def forward(self, t, x):
        # Combine inputs
        inps = torch.cat([t, x], dim=1)

        # Process through network
        hidden = self.input_layer(inps)
        if self.hidden_layers:
            for layer in self.hidden_layers:
                hidden = layer(hidden)

        if self.dgm_layers:
            S1 = hidden
            S = hidden
            for layer in self.dgm_layers:
                S = layer(hidden, S, S1)
            hidden = S

        # Get base outputs
        output = self.output_layer(hidden)
        u_base, p_base = output[:, 0], output[:, 1]

        # Apply conditions
        u = t * u_base + self.initial_conditions(x)
        p = self.enforce_neumann_boundary(p_base, x)

        return u, p
