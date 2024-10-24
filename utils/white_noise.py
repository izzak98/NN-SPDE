import torch

def simulate_dot_W_spectral(points, modes=10):
    """
    Simulates space-time white noise via continuous spectral decomposition
    at arbitrary points in both space and time.
    
    Parameters:
    - points: A tensor of shape (N, d+1) where N is the number of points 
              and d is the spatial dimension. Each row represents (t, x_1, x_2, ..., x_d).
    - modes: The number of Fourier modes to use for spectral decomposition (default is 10).
    
    Returns:
    - noise_values: A tensor of simulated white noise values at the given points.
    """
    points = torch.tensor(points)
    N, d = points.shape
    d -= 1  # Subtract 1 to account for the time dimension

    # Create mode numbers
    k = torch.arange(1, modes + 1).unsqueeze(0).unsqueeze(1)  # Shape: (1, 1, modes)
    
    # Extract spatial and time coordinates
    t = points[:, 0].unsqueeze(1)  # Shape: (N, 1)
    x = points[:, 1:].unsqueeze(2)  # Shape: (N, d, 1)

    # Compute Fourier basis for space
    fourier_basis = torch.sin(k * torch.pi * x)  # Shape: (N, d, modes)

    # Compute product of basis functions across spatial dimensions
    basis_product = torch.prod(fourier_basis, dim=1)  # Shape: (N, modes)

    # Simulate a continuous-time stochastic process for each mode
    # In this case, Brownian motion is evaluated at each arbitrary time t
    W_t = torch.normal(0, 1, size=(N, modes)) * torch.sqrt(t)  # Brownian motion scaling with time

    # Compute noise values
    noise_values = torch.sum(W_t * basis_product, dim=1)

    return noise_values

# Example usage:
if __name__ == "__main__":
    points = torch.tensor([
        [0.5, 0.3, 0.8, 0.3],
        [0.7, 0.5, 0.01, 0.2],
        [0.2, 0.8, 0.6, 0.4]
    ])
    noise_values = simulate_dot_W_continuous(points)
    print(noise_values)
