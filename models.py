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


def create_fc_layers(input_dim: int,
                     hidden_dims: list[int],
                     activation: str,
                     output_activation: str,
                     output_dim: int = 1):
    layers = nn.ModuleList()
    for i in range(len(hidden_dims)):
        if i == 0:
            layers.append(nn.Linear(input_dim, hidden_dims[i]))
        else:
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
        if activation:
            layers.append(get_activation(activation))
    layers.append(nn.Linear(hidden_dims[-1], output_dim))
    if output_activation:
        layers.append(get_activation(output_activation))
    return layers


class HeatDGM(nn.Module):
    def __init__(self,
                 input_dims: int,
                 hidden_dims: list[int],
                 activation: str,
                 output_activation: str):
        super(HeatDGM, self).__init__()
        self.layer_list = create_fc_layers(
            input_dims+1, hidden_dims, activation, output_activation)
        self.layers = nn.Sequential(*self.layer_list)

    def forward(self, X, t):
        """
        X: spatial coordinates (N, d)
        t: time coordinate (N, 1)
        """
        input_tensor = torch.cat([X, t], dim=1)
        x = input_tensor
        x = self.layers(x)
        return x


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


class HeatMIM(nn.Module):
    def __init__(self,
                 input_dims: int,
                 u_hidden_dims: list[int],
                 p_hidden_dims: list[int],
                 u_activation: str,
                 u_out_activation: str,
                 p_activation: str,
                 p_out_activation: str):
        super(HeatMIM, self).__init__()
        self.input_dims = input_dims
        self.u_layer_list = create_fc_layers(
            input_dims+1, u_hidden_dims, u_activation, u_out_activation)
        self.u_layers = nn.Sequential(*self.u_layer_list)
        self.p_layer_list = create_fc_layers(
            input_dims+1, p_hidden_dims, p_activation, p_out_activation, output_dim=input_dims)
        self.p_layers = nn.Sequential(*self.p_layer_list)

    def forward(self, X, t):
        """
        X: spatial coordinates (N, d)
        t: time coordinate (N, 1)
        """
        input_tensor = torch.cat([X, t], dim=1)

        # Enforce initial condition
        u_homogeneous = self.u_layers(input_tensor)
        u = u_homogeneous * t + self.initial_condition(X)

        # Enforce Neumann boundary condition
        p_homogeneous = self.p_layers(input_tensor)
        p = p_homogeneous * self.boundary_distance(X)

        return u, p

    def initial_condition(self, X):
        # Implement the initial condition: u(0, x) = prod(sin(pi * x_i))
        return torch.prod(torch.sin(torch.pi * X), dim=1, keepdim=True)

    def boundary_distance(self, X):
        # Compute the distance to the boundary for each spatial point
        # This function should return 0 on the boundary and > 0 inside the domain
        return torch.prod(X * (1 - X), dim=1, keepdim=True)


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
    dgm = HeatDGM(2, [128, 128], 'tanh', '')
    mim = HeatMIM(2, [128, 128], [128, 128], 'relu', 'tanh', 'relu', 'tanh')
    print(dgm)
    print(mim)
    print(dgm(torch.rand(10, 2), torch.rand(10, 1)))
    print(mim(torch.rand(10, 2), torch.rand(10, 1),))
