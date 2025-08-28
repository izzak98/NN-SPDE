"""Feedforward Neural Network (FNN) for regression tasks."""

import torch.nn as nn


class FNN(nn.Module):
    """
    A simple feedforward neural network (FNN) for regression tasks.
    This network consists of multiple fully connected layers with Tanh activation.
    The number of layers and hidden dimensions can be configured.
    """
    __name__ = "FNN"

    def __init__(self, input_dim=4, hidden_dim=128, num_layers=5):  # d + t + nu
        """
        Initialize the FNN.
        :param input_dim: Number of input features (default: 4 for t, x1, x2, nu)
        :param hidden_dim: Number of neurons in each hidden layer (default: 128)
        :param num_layers: Total number of layers in the network (default: 5)
        """
        super(FNN, self).__init__()

        # Build network layers
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 1))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.network(x)
