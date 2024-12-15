from typing import Any, Callable
import torch
from torch import nn
import numpy as np
from utils.model_utils import create_fc_layers, MIM, create_x_circ


class HeatDGM(nn.Module):
    def __init__(self,
                 input_dims: int,
                 hidden_dims: list,
                 dgm_dims: int,
                 n_dgm_layers: int,
                 hidden_activation: str,
                 output_activation: str):
        super(HeatDGM, self).__init__()
        input_dims += 2  # Add time and nu dimensions
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.dgm_dims = dgm_dims
        self.n_dgm_layers = n_dgm_layers
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.name = "Heat DGM"

        layers = create_fc_layers(
            input_dims, hidden_dims, hidden_activation, dgm_dims,
            n_dgm_layers, output_activation)
        self.input_layer, self.hidden_layers, self.dgm_layers, self.output_layer = layers

    def forward(self, t, nu, *args):  # -> Any:
        # shapes:
        # t: (batch_size, 1)
        # x: (batch_size, dims)
        x = torch.cat(args, dim=1)
        inps = torch.cat([t, nu, x], dim=1)
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


class BurgerDGM(nn.Module):
    def __init__(self,
                 input_dims: int,
                 hidden_dims: list,
                 dgm_dims: int,
                 n_dgm_layers: int,
                 hidden_activation: str,
                 output_activation: str):
        super(BurgerDGM, self).__init__()
        input_dims += 3  # Add time and nu dimensions
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.dgm_dims = dgm_dims
        self.n_dgm_layers = n_dgm_layers
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.name = "Heat DGM"

        layers = create_fc_layers(
            input_dims, hidden_dims, hidden_activation, dgm_dims,
            n_dgm_layers, output_activation)
        self.input_layer, self.hidden_layers, self.dgm_layers, self.output_layer = layers

    def forward(self, t, nu, alpha, *args):  # -> Any:
        # shapes:
        # t: (batch_size, 1)
        # x: (batch_size, dims)
        x = torch.cat(args, dim=1)
        inps = torch.cat([t, nu, alpha, x], dim=1)
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


class HeatMIM(MIM):
    def __init__(self,
                 input_dims: int,
                 hidden_dims: list,
                 dgm_dims: int,
                 n_dgm_layers: int,
                 hidden_activation: str,
                 output_activation: str,
                 initial_conditions: Callable):
        super(HeatMIM, self).__init__(
            input_dims, hidden_dims, dgm_dims, n_dgm_layers,
            hidden_activation, output_activation, initial_conditions)
        self.name = "HeatMIM"

    def enforce_neumann_boundary(self, p_theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Enforce zero Neumann boundary conditions by making derivatives zero at boundaries
        Args:
            p_theta: Network output for derivatives (batch_size, dims)
            x: Spatial coordinates (batch_size, dims)
        """
        # Create boundary mask that smoothly goes to zero at boundaries
        boundary_mask = torch.prod(x * (1-x), dim=1, keepdim=True)
        return boundary_mask * p_theta

    def forward(self, t, nu, *args):
        # Combine inputs
        x = torch.cat(args, dim=1)
        inps = torch.cat([t, nu, x], dim=1)

        # Compute u
        u_base = self.u_input_layer(inps)
        if self.u_hidden_layers:
            for layer in self.u_hidden_layers:
                u_base = layer(u_base)
        if self.u_dgm_layers:
            S1 = u_base
            S = u_base
            for layer in self.u_dgm_layers:
                S = layer(u_base, S, S1)
            u_base = S
        u_base = self.u_output_layer(u_base)

        # Compute p
        p_base = self.p_input_layer(inps)
        if self.p_hidden_layers:
            for layer in self.p_hidden_layers:
                p_base = layer(p_base)
        if self.p_dgm_layers:
            S1 = p_base
            S = p_base
            for layer in self.p_dgm_layers:
                S = layer(p_base, S, S1)
            p_base = S
        p_base = self.p_output_layer(p_base)

        # Apply conditions
        dims = self.input_dims - 2
        u = t * u_base + self.initial_conditions(*args) * np.sqrt(dims) * np.log10(np.exp(dims))
        p = self.enforce_neumann_boundary(p_base, x)

        return u, p
