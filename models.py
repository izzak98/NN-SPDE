import torch
from torch import nn
from torch.nn.functional import tanh, sigmoid


def get_activation(activation: str):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    else:
        return lambda x: x


class DGMLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super(DGMLayer, self).__init__()

        # Z gate
        self.U_z = nn.Linear(hidden_dim, hidden_dim)
        self.W_z = nn.Linear(hidden_dim, hidden_dim)
        self.B_z = nn.Linear(hidden_dim, hidden_dim)  # Bias term for Z gate

        # G gate
        self.U_g = nn.Linear(hidden_dim, hidden_dim)
        self.W_g = nn.Linear(hidden_dim, hidden_dim)
        self.B_g = nn.Linear(hidden_dim, hidden_dim)  # Bias term for G gate

        # R gate
        self.U_r = nn.Linear(hidden_dim, hidden_dim)
        self.W_r = nn.Linear(hidden_dim, hidden_dim)
        self.B_r = nn.Linear(hidden_dim, hidden_dim)  # Bias term for R gate

        # H transformation
        self.U_h = nn.Linear(hidden_dim, hidden_dim)
        self.W_h = nn.Linear(hidden_dim, hidden_dim)
        self.B_h = nn.Linear(hidden_dim, hidden_dim)  # Bias term for H transformation

    def forward(self, x, S, S1):
        # Calculate gates with biases and activation functions
        Z = sigmoid(self.U_z(x) + self.W_z(S) + self.B_z(S1))
        G = sigmoid(self.U_g(x) + self.W_g(S1) + self.B_g(S))
        R = sigmoid(self.U_r(x) + self.W_r(S) + self.B_r(S1))

        # Compute the H transformation
        H = tanh(self.U_h(x) + self.W_h(S * R) + self.B_h(S1))

        # Calculate new state with gating mechanism
        S_new = (1 - G) * H + Z * S

        return S_new


def create_dgm_layers(dgm_dims: int, n_dgm_layers: int):
    if not dgm_dims:
        return

    layers = nn.ModuleList()
    for _ in range(n_dgm_layers):
        layers.append(DGMLayer(dgm_dims))
    return layers


def create_fc_layers(input_dim: int,
                     hidden_dims: list[int],
                     dense_activation: str,
                     dgm_dims: int,
                     n_dgm_layers: int,
                     output_activation: str,
                     output_dim: int = 1):
    dense_layers_list = nn.ModuleList()
    output_layers = nn.ModuleList()
    dgm_layers_list = None
    output_layer_in_params = 1
    input_layer = nn.Linear(input_dim, output_dim)
    if hidden_dims:
        input_layer = nn.Linear(input_dim, hidden_dims[0])
        if dense_activation:
            dense_layers_list.append(get_activation(dense_activation))
        for i in range(1, len(hidden_dims)):
            dense_layers_list.append(
                nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            if dense_activation:
                dense_layers_list.append(get_activation(dense_activation))
        output_layer_in_params = hidden_dims[-1]
        if dgm_dims and n_dgm_layers:
            dense_layers_list.append(nn.Linear(hidden_dims[-1], dgm_dims))

    if dgm_dims and n_dgm_layers:
        if not dense_layers_list:
            input_layer = nn.Linear(input_dim, dgm_dims)
        dgm_layers_list = create_dgm_layers(
            dgm_dims, n_dgm_layers)
        output_layer_in_params = dgm_dims

    output_layers.append(nn.Linear(output_layer_in_params, output_dim))
    if output_activation:
        output_layers.append(get_activation(output_activation))
    output_layer = nn.Sequential(*output_layers)

    if not dense_layers_list and not dgm_layers_list:
        input_layer = nn.Sequential(
            input_layer, get_activation(dense_activation))

    return input_layer, dense_layers_list, dgm_layers_list, output_layer


class HeatDGM(nn.Module):
    def __init__(self,
                 input_dims: int,
                 hidden_dims: list,
                 dgm_dims: int,
                 n_dgm_layers: int,
                 hidden_activation: str,
                 output_activation: str):
        super(HeatDGM, self).__init__()
        input_dims += 1  # Add time dimension
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

    def forward(self, X, t):
        inps = torch.cat([X, t], dim=1)
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
                 u_hidden_dims: list[int],
                 u_activation: str,
                 u_dgm_dims: int,
                 u_n_dgm_layers: int,
                 u_out_activation: str,
                 p_hidden_dims: list[int],
                 p_activation: str,
                 p_dgm_dims: int,
                 p_n_dgm_layers: int,
                 p_out_activation: str
                 ):
        super(HeatMIM, self).__init__()
        input_dims += 1  # Add time dimension

        self.input_dims = input_dims
        self.u_hidden_dims = u_hidden_dims
        self.u_activation = u_activation
        self.u_dgm_dims = u_dgm_dims
        self.u_n_dgm_layers = u_n_dgm_layers
        self.u_out_activation = u_out_activation

        self.p_hidden_dims = p_hidden_dims
        self.p_activation = p_activation
        self.p_dgm_dims = p_dgm_dims
        self.p_n_dgm_layers = p_n_dgm_layers
        self.p_out_activation = p_out_activation

        self.u_layers = create_fc_layers(
            input_dims, u_hidden_dims, u_activation, u_dgm_dims,
            u_n_dgm_layers, u_out_activation)
        self.u_input_layer, self.u_hidden_layers, self.u_dgm_layers, self.u_output_layer = self.u_layers

        self.p_layers = create_fc_layers(
            input_dims, p_hidden_dims, p_activation, p_dgm_dims,
            p_n_dgm_layers, p_out_activation, output_dim=input_dims-1)
        self.p_input_layer, self.p_hidden_layers, self.p_dgm_layers, self.p_output_layer = self.p_layers

    def initial_condition(self, X):
        # Implement the initial condition: u(0, x) = prod(sin(pi * x_i))
        with torch.no_grad():
            initial_condition = torch.prod(
                torch.sin(torch.pi * X), dim=1, keepdim=True)
        return initial_condition

    def boundary_condition(self, X):
        # Implement the boundary condition: p(t, x) = x(1-x) * p(t, x)
        with torch.no_grad():
            boundary_condition = X * (1-X)
        return boundary_condition

    def forward(self, X, t):
        """
        X: spatial coordinates (N, d)
        t: time coordinate (N, 1)
        """

        input_tensor = torch.cat([X, t], dim=1)

        u_input = self.u_input_layer(input_tensor)
        if self.u_hidden_layers:
            for layer in self.u_hidden_layers:
                u_input = layer(u_input)

        if self.u_dgm_layers:
            S1 = u_input
            S = u_input

            for layer in self.u_dgm_layers:
                S = layer(u_input, S, S1)
            u_input = S

        u_homogeneous = self.u_output_layer(u_input)
        # u = u_homogeneous * t + self.initial_condition(X)
        X_copy = X.clone().detach().requires_grad_(False)
        t_copy = t.clone().detach().requires_grad_(False)
        u = u_homogeneous * t_copy + self.initial_condition(X_copy)

        p_input = self.p_input_layer(input_tensor)
        if self.p_hidden_layers:
            for layer in self.p_hidden_layers:
                p_input = layer(p_input)

        if self.p_dgm_layers:
            S1 = p_input
            S = p_input

            for layer in self.p_dgm_layers:
                S = layer(p_input, S, S1)
            p_input = S

        p_homogeneous = self.p_output_layer(p_input)
        p = self.boundary_condition(X_copy) * p_homogeneous

        return u, p


class PeriodicTransform(nn.Module):
    def __init__(self, periods, harmonics=2):
        """
        Initialize the periodic transformation layer.

        Args:
        - periods (list or tensor): Periods for each dimension of the input.
        - harmonics (int): Number of harmonics (frequency terms) to include for each dimension.
        """
        super(PeriodicTransform, self).__init__()
        self.periods = torch.tensor(periods)
        self.harmonics = harmonics

    def forward(self, x):
        """
        Transform the input to satisfy periodic boundary conditions.

        Args:
        - x (tensor): Input tensor of shape (batch_size, d), where d is the spatial dimension.

        Returns:
        - Transformed tensor of shape (batch_size, d * harmonics * 2).
        """
        batch_size, d = x.shape
        transformed = []

        for i in range(d):
            for j in range(1, self.harmonics + 1):
                freq = 2 * torch.pi * j / self.periods[i]
                # sine component
                transformed.append(torch.sin(freq * x[:, i:i+1]))
                # cosine component
                transformed.append(torch.cos(freq * x[:, i:i+1]))

        # Concatenate all frequency components along the last dimension
        return torch.cat(transformed, dim=1)


class BurgersMIM(nn.Module):
    def __init__(
        self,
        input_dims: int,
        u_hidden_dims: list[int],
        u_activation: str,
        u_dgm_dims: int,
        u_n_dgm_layers: int,
        u_dgm_activation: str,
        u_out_activation: str,
        p_hidden_dims: list[int],
        p_activation: str,
        p_dgm_dims: int,
        p_n_dgm_layers: int,
        p_dgm_activation: str,
        p_out_activation: str
    ):
        super(HeatMIM, self).__init__()
        self.periodic_transformer = PeriodicTransform([1]*input_dims, harmonics=2)

        input_dims  = 2 * 2 * input_dims + 1 + 2 # Add periodic boundary conditions

        self.input_dims = input_dims
        self.u_hidden_dims = u_hidden_dims
        self.u_activation = u_activation
        self.u_dgm_dims = u_dgm_dims
        self.u_n_dgm_layers = u_n_dgm_layers
        self.u_dgm_activation = u_dgm_activation
        self.u_out_activation = u_out_activation

        self.p_hidden_dims = p_hidden_dims
        self.p_activation = p_activation
        self.p_dgm_dims = p_dgm_dims
        self.p_n_dgm_layers = p_n_dgm_layers
        self.p_dgm_activation = p_dgm_activation
        self.p_out_activation = p_out_activation

        self.u_layers = create_fc_layers(
            input_dims, u_hidden_dims, u_activation, u_dgm_dims,
            u_n_dgm_layers, u_dgm_activation, u_out_activation)
        self.u_input_layer, self.u_hidden_layers, self.u_dgm_layers, self.u_output_layer = self.u_layers

        self.p_layers = create_fc_layers(
            input_dims, p_hidden_dims, p_activation, p_dgm_dims,
            p_n_dgm_layers, p_dgm_activation, p_out_activation, output_dim=input_dims-1)
        self.p_input_layer, self.p_hidden_layers, self.p_dgm_layers, self.p_output_layer = self.p_layers


    def initial_condition(self, X):
        # Implement the initial condition: u(0, x) = prod(sin(pi * x_i))
        with torch.no_grad():
            initial_condition = torch.prod(
                torch.sin(torch.pi * X), dim=1, keepdim=True)
        return initial_condition
    
    def forward(self, t, x, nu, alpha):
        """
        x: spatial coordinates (N, d)
        t: time coordinate (N, 1)
        nu: viscosity coefficient (N1 1)
        alpha: stochastic coefficient (N, 1)
        """

        periodic_x = self.periodic_transformer(x)
        input_tensor = torch.cat([periodic_x, t, nu, alpha], dim=1)

        u_input = self.u_input_layer(input_tensor)
        if self.u_hidden_layers:
            for layer in self.u_hidden_layers:
                u_input = layer(u_input)

        if self.u_dgm_layers:
            S1 = u_input
            S = u_input

            for layer in self.u_dgm_layers:
                S = layer(u_input, S, S1)
            u_input = S

        u_homogeneous = self.u_output_layer(u_input)
        t_copy = t.clone().detach().requires_grad_(False)
        x_copy = x.clone().detach().requires_grad_(False)
        u = u_homogeneous * t_copy + self.initial_condition(x_copy)

        p_input = self.p_input_layer(input_tensor)
        if self.p_hidden_layers:
            for layer in self.p_hidden_layers:
                p_input = layer(p_input)

        if self.p_dgm_layers:
            S1 = p_input
            S = p_input

            for layer in self.p_dgm_layers:
                S = layer(p_input, S, S1)
            p_input = S

        p = self.p_output_layer(p_input)        

        return u, p
        