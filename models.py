from typing import Any, Callable
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
        self.name = "DGM"

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
                 output_activation: str,
                 initial_conditions: Callable):
        super(HeatMIM, self).__init__()
        input_dims += 1  # Add time dimension
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.dgm_dims = dgm_dims
        self.n_dgm_layers = n_dgm_layers
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.initial_conditions = initial_conditions
        self.name = "MIM"

        # Create network layers
        u_layers = create_fc_layers(
            input_dims, hidden_dims, hidden_activation, dgm_dims,
            n_dgm_layers, output_activation, output_dim=1)
        self.u_input_layer, self.u_hidden_layers, self.u_dgm_layers, self.u_output_layer = u_layers

        p_layers = create_fc_layers(
            input_dims, hidden_dims, hidden_activation, dgm_dims,
            n_dgm_layers, output_activation, output_dim=input_dims-1)
        self.p_input_layer, self.p_hidden_layers, self.p_dgm_layers, self.p_output_layer = p_layers

    def enforce_neumann_boundary(self, p_theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Enforce zero Neumann boundary conditions on the unit hypercube
        Args:
            p_theta: Network output for derivatives (batch_size, dims)
            x: Spatial coordinates (batch_size, dims)
        """
        return x * (1-x) * p_theta

    def forward(self, t, *args):
        # Combine inputs
        x = torch.cat(args, dim=1)
        inps = torch.cat([t, x], dim=1)

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
        u = t * u_base + self.initial_conditions(*args)
        p = self.enforce_neumann_boundary(p_base, x)

        return u, p
