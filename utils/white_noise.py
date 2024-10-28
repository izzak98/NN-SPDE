import torch

def simulate_dot_W_spectral(points, modes=10, seed=None):
    """Optimized version of white noise simulation"""
    if seed is not None:
        generator = torch.Generator(device=points.device)
        generator.manual_seed(seed)
    else:
        generator = None
    
    N, d = points.shape
    d -= 1  # Subtract 1 for time dimension
    
    # Vectorized operations
    t = points[:, 0].unsqueeze(1)  # Shape: (N, 1)
    x = points[:, 1:].unsqueeze(2)  # Shape: (N, d, 1)
    k = torch.arange(1, modes + 1, device=points.device).unsqueeze(0).unsqueeze(1)
    
    # Compute all Fourier bases at once
    fourier_basis = torch.sin(k * torch.pi * x)  # Shape: (N, d, modes)
    basis_product = torch.prod(fourier_basis, dim=1)  # Shape: (N, modes)
    
    # Generate noise on GPU directly
    W_t = torch.randn(N, modes, device=points.device, generator=generator) * torch.sqrt(t)
    
    return torch.sum(W_t * basis_product, dim=1)


# Example usage:
if __name__ == "__main__":
    # Test reproducibility
    points = torch.tensor([
        [0.5, 0.3, 0.8, 0.3],
        [0.7, 0.5, 0.01, 0.2],
        [0.2, 0.8, 0.6, 0.4]
    ])
    
    # Generate noise with same seed
    seed = 128
    noise1 = simulate_dot_W_spectral(points, seed=seed)
    noise2 = simulate_dot_W_spectral(points, seed=seed)
    print("Same seed produces same noise:")
    print(f"Noise 1: {noise1}")
    print(f"Noise 2: {noise2}")
    print(f"Maximum difference: {torch.max(torch.abs(noise1 - noise2))}")
    
    # Generate noise with different seeds
    noise3 = simulate_dot_W_spectral(points, seed=None)
    noise4 = simulate_dot_W_spectral(points, seed=None)
    print("\nDifferent seeds produce different noise:")
    print(f"Noise 3: {noise3}")
    print(f"Noise 4: {noise4}")
    print(f"Maximum difference: {torch.max(torch.abs(noise3 - noise4))}")