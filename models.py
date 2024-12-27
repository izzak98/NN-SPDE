from typing import Any, Callable
import torch
from torch import nn
import numpy as np
from utils.model_utils import create_fc_layers, MIM, create_x_circ


class DGM(nn.Module):
    def __init__(self,
                 spatial_dims: int,
                 add_dims: int,
                 hidden_dims: list,
                 dgm_dims: int,
                 n_dgm_layers: int,
                 hidden_activation: str,
                 output_activation: str,
                 name: str = "DGM"):
        super(DGM, self).__init__()
        self.spatial_dims = spatial_dims
        self.add_dims = add_dims
        input_dims = spatial_dims + add_dims
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.dgm_dims = dgm_dims
        self.n_dgm_layers = n_dgm_layers
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.name = name

        layers = create_fc_layers(
            input_dims, hidden_dims, hidden_activation, dgm_dims,
            n_dgm_layers, output_activation)
        self.input_layer, self.hidden_layers, self.dgm_layers, self.output_layer = layers

    def forward(self, t, *args):  # -> Any:
        # shapes:
        # t: (batch_size, 1)
        # x: (batch_size, dims)
        add_args = args[:-self.spatial_dims]
        x = torch.cat(args[-self.spatial_dims:], dim=1).to(t.device)
        inps = torch.cat([t, *add_args, x], dim=1)
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
                 spatial_dims: int,
                 add_dims: int,
                 u_hidden_dims: list,
                 u_dgm_dims: int,
                 u_n_dgm_layers: int,
                 u_hidden_activation: str,
                 u_output_activation: str,
                 p_hidden_dims: list,
                 p_dgm_dims: int,
                 p_n_dgm_layers: int,
                 p_hidden_activation: str,
                 p_output_activation: str,
                 initial_conditions: Callable
                 ):
        super(HeatMIM, self).__init__(
            spatial_dims=spatial_dims,
            add_dims=add_dims,
            p_output_dims=spatial_dims,
            u_hidden_dims=u_hidden_dims,
            u_dgm_dims=u_dgm_dims,
            u_n_dgm_layers=u_n_dgm_layers,
            u_hidden_activation=u_hidden_activation,
            u_output_activation=u_output_activation,
            p_hidden_dims=p_hidden_dims,
            p_dgm_dims=p_dgm_dims,
            p_n_dgm_layers=p_n_dgm_layers,
            p_hidden_activation=p_hidden_activation,
            p_output_activation=p_output_activation,
            initial_conditions=initial_conditions
        )
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
        u = t * u_base + self.initial_conditions(*args)
        p = self.enforce_neumann_boundary(p_base, x)

        return u, p


class BurgerMIM(MIM):
    def __init__(self,
                 spatial_dims: int,
                 add_dims: int,
                 u_hidden_dims: list,
                 u_dgm_dims: int,
                 u_n_dgm_layers: int,
                 u_hidden_activation: str,
                 u_output_activation: str,
                 p_hidden_dims: list,
                 p_dgm_dims: int,
                 p_n_dgm_layers: int,
                 p_hidden_activation: str,
                 p_output_activation: str,
                 initial_conditions: Callable
                 ):
        super(BurgerMIM, self).__init__(
            spatial_dims=spatial_dims*8,
            p_output_dims=spatial_dims,
            add_dims=add_dims,
            u_hidden_dims=u_hidden_dims,
            u_dgm_dims=u_dgm_dims,
            u_n_dgm_layers=u_n_dgm_layers,
            u_hidden_activation=u_hidden_activation,
            u_output_activation=u_output_activation,
            p_hidden_dims=p_hidden_dims,
            p_dgm_dims=p_dgm_dims,
            p_n_dgm_layers=p_n_dgm_layers,
            p_hidden_activation=p_hidden_activation,
            p_output_activation=p_output_activation,
            initial_conditions=initial_conditions
        )

        self.name = "BurgerMIM"

    def forward(self, t, nu, alpha, *args):
        # Combine inputs
        x = torch.cat(args, dim=1)
        x_circ = create_x_circ(x)
        inps = torch.cat([t, nu, alpha, x_circ], dim=1)

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
        p = self.p_output_layer(p_base)

        # Apply conditions
        u = t * u_base + self.initial_conditions(*args)

        return u, p


class KPZMIM(MIM):
    def __init__(self,
                 spatial_dims: int,
                 add_dims: int,
                 u_hidden_dims: list,
                 u_dgm_dims: int,
                 u_n_dgm_layers: int,
                 u_hidden_activation: str,
                 u_output_activation: str,
                 p_hidden_dims: list,
                 p_dgm_dims: int,
                 p_n_dgm_layers: int,
                 p_hidden_activation: str,
                 p_output_activation: str,
                 initial_conditions: Callable
                 ):
        super(KPZMIM, self).__init__(
            spatial_dims=spatial_dims*8,
            p_output_dims=spatial_dims,
            add_dims=add_dims,
            u_hidden_dims=u_hidden_dims,
            u_dgm_dims=u_dgm_dims,
            u_n_dgm_layers=u_n_dgm_layers,
            u_hidden_activation=u_hidden_activation,
            u_output_activation=u_output_activation,
            p_hidden_dims=p_hidden_dims,
            p_dgm_dims=p_dgm_dims,
            p_n_dgm_layers=p_n_dgm_layers,
            p_hidden_activation=p_hidden_activation,
            p_output_activation=p_output_activation,
            initial_conditions=initial_conditions
        )
        self.name = "KPZMIM"

    def forward(self, t, nu, alpha, lambda_kpz, *args):
        # Combine inputs
        x = torch.cat(args, dim=1)
        x_circ = create_x_circ(x)
        inps = torch.cat([t, nu, alpha, lambda_kpz, x_circ], dim=1)

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
        p = self.p_output_layer(p_base)

        # Apply conditions
        u = t * u_base + self.initial_conditions(*args)

        return u, p
