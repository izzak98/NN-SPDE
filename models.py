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


def create_fc_layers(input_dim: int, hidden_dims: list[int], activation: str, output_dim: int = 1):
    layers = nn.ModuleList()
    for i in range(len(hidden_dims) - 1):
        if i == 0:
            layers.append(nn.Linear(input_dim, hidden_dims[i]))
        else:
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
        layers.append(get_activation(activation))
    layers.append(nn.Linear(hidden_dims[-1], output_dim))
    return layers


class DGM(nn.Module):
    def __init__(self, input_dims, hidden_dims: list[int], activation: str):
        super(DGM, self).__init__()
        self.layer_list = create_fc_layers(
            input_dims+1, hidden_dims, activation)
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


class HeatMIM(nn.Module):
    def __init__(self, input_dims: int, u_hidden_dims: list[int], p_hidden_dims: list[int], activation: str):
        super(HeatMIM, self).__init__()
        self.input_dims = input_dims
        self.u_layer_list = create_fc_layers(input_dims+1, u_hidden_dims, activation)
        self.u_layers = nn.Sequential(*self.u_layer_list)
        self.p_layer_list = create_fc_layers(input_dims+1, p_hidden_dims, activation, output_dim=input_dims)
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

if __name__ == "__main__":
    dgm = DGM(2, [128, 128, 1], 'relu')
    mim = HeatMIM(2, [128, 128, 1], [128, 128, 1], 'relu')
    print(dgm)
    print(mim)
    print(dgm(torch.rand(10, 2), torch.rand(10, 1)))
    print(mim(torch.rand(10, 2), torch.rand(10, 1)))