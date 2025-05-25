import torch


def shifted_gaussian_white_noise(t: torch.Tensor, mu: float, sigma: float, seed: int = None):
    if seed is not None:
        torch.manual_seed(seed)
    noise = torch.randn_like(t) * sigma + mu
    return noise.unsqueeze(1)  # Add a dimension to match the shape of t


def time_dependant_gaussian_white_noise(t: torch.Tensor, mu: float, sigma: float, seed: int = None):
    if seed is not None:
        torch.manual_seed(seed)
    noise = torch.randn_like(t) * t * sigma + mu
    return noise.unsqueeze(1)  # Add a dimension to match the shape of t


if __name__ == "__main__":
    # validate mean
    t = torch.linspace(0, 1, 1000)
    mu = 0.5
    sigma = 0.1
    noise = shifted_gaussian_white_noise(t, mu, sigma)
    print(f"Mean: {noise.mean().item()}, Std: {noise.std().item()}")
