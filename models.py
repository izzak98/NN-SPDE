import torch
from torch import nn


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
    def __init__(self, hidden_dim: int, activation: str = 'tanh'):  # Note: removed input_dim
        super(DGMLayer, self).__init__()
        self.activation = get_activation(activation)

        # Z gate
        self.U_z = nn.Linear(hidden_dim, hidden_dim)
        self.W_z = nn.Linear(hidden_dim, hidden_dim)

        # G gate
        self.U_g = nn.Linear(hidden_dim, hidden_dim)
        self.W_g = nn.Linear(hidden_dim, hidden_dim)

        # R gate
        self.U_r = nn.Linear(hidden_dim, hidden_dim)
        self.W_r = nn.Linear(hidden_dim, hidden_dim)

        # H transformation
        self.U_h = nn.Linear(hidden_dim, hidden_dim)
        self.W_h = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, S, S1):
        # All inputs should already be in hidden_dim space
        Z = self.activation(self.U_z(x) + self.W_z(S))
        G = self.activation(self.U_g(x) + self.W_g(S1))
        R = self.activation(self.U_r(x) + self.W_r(S))

        # Calculate hidden state
        H = self.activation(self.U_h(x) + self.W_h(S * R))

        # Calculate new state
        S_new = (1 - G) * H + Z * S

        return S_new


def create_dgm_layers(dgm_dims: int, n_dgm_layers:int, activation: str):
    if not dgm_dims:
        return

    layers = nn.ModuleList()
    for _ in range(n_dgm_layers):
        layers.append(DGMLayer(dgm_dims, activation))
    return layers


def create_fc_layers(input_dim: int,
                     hidden_dims: list[int],
                     dense_activation: str,
                     dgm_dims: int,
                     n_dgm_layers: int,
                     dgm_activation: str,
                     output_activation: str,
                     output_dim: int = 1):
    dense_layers_list = nn.ModuleList()
    output_layers = nn.ModuleList()
    dgm_layers_list = None
    output_layer_in_params = input_dim
    if hidden_dims:
        input_layer = nn.Linear(input_dim, hidden_dims[0])
        if dense_activation:
            dense_layers_list.append(get_activation(dense_activation))
        for i in range(1, len(hidden_dims)):
            dense_layers_list.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            if dense_activation:
                dense_layers_list.append(get_activation(dense_activation))
        output_layer_in_params = hidden_dims[-1]
        if dgm_dims and n_dgm_layers:
            dense_layers_list.append(nn.Linear(hidden_dims[-1], dgm_dims))

    if dgm_dims and n_dgm_layers:
        if not dense_layers_list:
            input_layer = nn.Linear(input_dim, dgm_dims)
        dgm_layers_list = create_dgm_layers(dgm_dims, n_dgm_layers, dgm_activation)
        output_layer_in_params = dgm_dims

    output_layers.append(nn.Linear(output_layer_in_params, output_dim))
    if output_activation:
        output_layers.append(get_activation(output_activation))
    output_layer = nn.Sequential(*output_layers)

    return input_layer, dense_layers_list, dgm_layers_list, output_layer


class HeatDGM(nn.Module):
    def __init__(self,
                 input_dims: int,
                 hidden_dims: list,
                 dgm_dims: int,
                 n_dgm_layers: int,
                 hidden_activation: str,
                 dgm_activation: str,
                 output_activation: str):
        super(HeatDGM, self).__init__()
        input_dims += 1  # Add time dimension
        self.input_dims = input_dims 
        self.hidden_dims = hidden_dims
        self.dgm_dims = dgm_dims
        self.n_dgm_layers = n_dgm_layers
        self.hidden_activation = hidden_activation
        self.dgm_activation = dgm_activation
        self.output_activation = output_activation

        layers = create_fc_layers(
            input_dims, hidden_dims, hidden_activation, dgm_dims,
            n_dgm_layers, dgm_activation, output_activation)
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
        input_dims += 1  # Add time dimension

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
        u = u_homogeneous * t + self.initial_condition(X)

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
        p = p_homogeneous * self.boundary_distance(X)

        return u, p

    def initial_condition(self, X):
        # Implement the initial condition: u(0, x) = prod(sin(pi * x_i))
        return torch.prod(torch.sin(torch.pi * X), dim=1, keepdim=True)

    def boundary_distance(self, X):
        # Compute the distance to the boundary for each spatial point
        # This function should return 0 on the boundary and > 0 inside the domain
        return torch.prod(X * (1 - X), dim=1, keepdim=True)


class BurgersDGM(nn.Module):
    def __init__(self, input_dims, hidden_dims: list[int], activation: str):
        super(BurgersDGM, self).__init__()
        self.layer_list = create_fc_layers(
            input_dims+3, hidden_dims, activation)
        self.layers = nn.Sequential(*self.layer_list)

    def forward(self, X, t, nu, alpha):
        """
        Parameters:
        X: spatial coordinates (N, d)
        t: time coordinate (N, 1)
        nu: viscosity coefficient (N, 1)
        alpha: noise coefficient (N, 1)
        """
        input_tensor = torch.cat([X, t, nu, alpha], dim=1)
        x = input_tensor
        x = self.layers(x)
        return x

class BurgersMIM(nn.Module):
    def __init__(self, input_dims: int, u_hidden_dims: list[int], p_hidden_dims: list[int], activation: str):
        super(BurgersMIM, self).__init__()
        self.input_dims = input_dims

        # 3 extra dimensions for nu, alpha, and t
        self.u_layer_list = create_fc_layers(
            input_dims+3, u_hidden_dims, activation)
        self.u_layers = nn.Sequential(*self.u_layer_list)

        self.p_layer_list = create_fc_layers(
            input_dims+3, p_hidden_dims, activation, output_dim=input_dims)
        self.p_layers = nn.Sequential(*self.p_layer_list)

    def forward(self, X, t, nu, alpha):
        """
        Parameters:
        X: spatial coordinates (N, d)
        t: time coordinate (N, 1)
        nu: viscosity coefficient (N, 1)
        alpha: noise coefficient (N, 1)
        """
        # Combine all inputs
        input_tensor = torch.cat([X, t, nu, alpha], dim=1)

        # Create periodic features for spatial coordinates
        X_periodic = self.make_periodic_features(X)

        # Enforce initial condition
        u_homogeneous = self.u_layers(input_tensor)
        u = u_homogeneous * t + self.initial_condition(X)

        # Enforce periodic boundary condition through periodic features
        p_homogeneous = self.p_layers(input_tensor)
        p = p_homogeneous * X_periodic

        return u, p

    def make_periodic_features(self, X):
        """
        Transform spatial coordinates to ensure periodic boundary conditions.
        Uses sine and cosine transformations for periodicity.
        """
        # Create periodic features using sin(2πx) and cos(2πx)
        features = torch.empty_like(X)
        for i in range(self.input_dims):
            features[:, i] = torch.sin(2 * torch.pi * X[:, i])
        return features

    def initial_condition(self, X):
        """
        Implement the initial condition: u(0, x) = prod(sin(pi * x_i))
        """
        return torch.prod(torch.sin(torch.pi * X), dim=1, keepdim=True)


if __name__ == "__main__":
    # Only dense layers
    dgm_model1 = HeatDGM(
        input_dims=2,
        hidden_dims=[50, 50, 50, 50],
        dgm_dims=0,
        n_dgm_layers=0,
        hidden_activation='tanh',
        dgm_activation='tanh',
        output_activation=None
    )

    mim_model1 = HeatMIM(
        input_dims=2,
        u_hidden_dims=[50, 50, 50],
        u_activation='tanh',
        u_dgm_dims=0,
        u_n_dgm_layers=0,
        u_dgm_activation='tanh',
        u_out_activation=None,
        p_hidden_dims=[50, 50],
        p_activation='tanh',
        p_dgm_dims=0,
        p_n_dgm_layers=0,
        p_dgm_activation='tanh',
        p_out_activation=None
    )

    # Only DGM layers
    dgm_model2 = HeatDGM(
        input_dims=2,
        hidden_dims=[],
        dgm_dims=50,
        n_dgm_layers=2,
        hidden_activation='tanh',
        dgm_activation='tanh',
        output_activation=None
    )

    mim_model2 = HeatMIM(
        input_dims=2,
        u_hidden_dims=[],
        u_activation='tanh',
        u_dgm_dims=50,
        u_n_dgm_layers=2,
        u_dgm_activation='tanh',
        u_out_activation=None,
        p_hidden_dims=[],
        p_activation='tanh',
        p_dgm_dims=50,
        p_n_dgm_layers=2,
        p_dgm_activation='tanh',
        p_out_activation=None
    )

    # Both dense and DGM layers
    dgm_model3 = HeatDGM(
        input_dims=2,
        hidden_dims=[50, 50],
        dgm_dims=50,
        n_dgm_layers=2,
        hidden_activation='tanh',
        dgm_activation='tanh',
        output_activation=None
    )

    mim_model3 = HeatMIM(
        input_dims=2,
        u_hidden_dims=[50, 50],
        u_activation='tanh',
        u_dgm_dims=50,
        u_n_dgm_layers=2,
        u_dgm_activation='tanh',
        u_out_activation=None,
        p_hidden_dims=[50, 50],
        p_activation='tanh',
        p_dgm_dims=50,
        p_n_dgm_layers=2,
        p_dgm_activation='tanh',
        p_out_activation=None
    )

    print(dgm_model1)
    print(dgm_model2)
    print(dgm_model3)

    sample_X = torch.randn(10, 2)
    sample_t = torch.randn(10, 1)
    print(dgm_model1(sample_X, sample_t))
    print(dgm_model2(sample_X, sample_t))
    print(dgm_model3(sample_X, sample_t))

    print(mim_model1)
    print(mim_model2)
    print(mim_model3)

    print(mim_model1(sample_X, sample_t))
    print(mim_model2(sample_X, sample_t))
    print(mim_model3(sample_X, sample_t))
