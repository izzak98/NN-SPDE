import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional
import time
from itertools import product

class BrownianSheet:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize Brownian Sheet simulator
        Args:
            device: torch device for computations
        """
        self.device = device

    @staticmethod
    @torch.jit.script
    def _compute_covariance_batch_fast(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Ultra-fast vectorized covariance computation using cumulative products
        Args:
            x1: tensor of shape (batch_size1, n_dims)
            x2: tensor of shape (batch_size2, n_dims)
        Returns:
            Covariance tensor of shape (batch_size1, batch_size2)
        """
        n_dims = x1.size(1)
        x1 = x1.unsqueeze(1)  # (batch1, 1, dims)
        x2 = x2.unsqueeze(0)  # (1, batch2, dims)
        
        # Scale coordinates to prevent numerical issues in high dimensions
        scale_factor = torch.exp(torch.log(torch.tensor(0.5, device=x1.device)) / n_dims)
        x1 = x1 * scale_factor
        x2 = x2 * scale_factor
        
        mins = torch.minimum(x1, x2)  # (batch1, batch2, dims)
        log_mins = torch.log(mins + 1e-10)
        log_prod = log_mins.sum(dim=2)
        return torch.exp(log_prod)

    @staticmethod
    @torch.jit.script
    def _cholesky_simulate_batch_fast(L: torch.Tensor, n_samples: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """
        Optimized batched Cholesky simulation
        """
        z = torch.randn(n_samples, L.shape[0], device=L.device, generator=generator)
        
        # Optimize matrix multiplication based on size
        if L.shape[0] <= 512:
            return torch.mm(z, L.T)
        return (L @ z.T).T

    def simulate(self, points: torch.Tensor, n_samples: int = 1,
                method: str = 'auto', chunk_size: int = 2048,
                seed: Optional[int] = None) -> torch.Tensor:
        """
        Simulate Brownian sheet at given points with dimensional scaling
        Args:
            points: tensor of shape (n_points, n_dims)
            n_samples: number of samples to generate
            method: 'auto' or 'cholesky'
            chunk_size: size of chunks for memory-efficient computation
            seed: Optional seed for random number generation
        Returns:
            Simulated values of shape (n_samples, n_points)
        """
        points = points.to(self.device)
        n_points, n_dims = points.shape
        
        # Set up generator
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device)  # CPU generator is more reliable
            generator.manual_seed(seed)
        
        # Pre-allocate output tensor
        result = torch.empty(n_samples, n_points, device=self.device)
        
        # Process in strided chunks
        for i in range(0, n_points, chunk_size):
            end_idx = min(i + chunk_size, n_points)
            chunk_points = points[i:end_idx]
            
            # Compute covariance matrix efficiently
            cov = self._compute_covariance_batch_fast(chunk_points, chunk_points)
            
            # Numerical stability with efficient indexing
            diag_indices = torch.arange(cov.shape[0], device=cov.device)
            cov.diagonal().add_(1e-7)
            
            try:
                L = torch.linalg.cholesky(cov)
            except RuntimeError:
                # First fallback: symmetrize and try again
                cov = (cov + cov.T) * 0.5
                cov.diagonal().add_(1e-6)
                try:
                    L = torch.linalg.cholesky(cov)
                except RuntimeError:
                    # Second fallback: larger diagonal term
                    cov.diagonal().add_(1e-5)
                    L = torch.linalg.cholesky(cov)
            
            # Generate samples
            chunk_result = self._cholesky_simulate_batch_fast(L, n_samples, generator)
            
            # Scale based on dimension to maintain variance
            dim_scale = torch.sqrt(torch.tensor(n_dims, device=chunk_result.device))
            chunk_result.mul_(dim_scale)
            
            # Store in pre-allocated tensor
            result[:, i:end_idx] = chunk_result
        
        return result

    @staticmethod
    def theoretical_variance(points: torch.Tensor) -> torch.Tensor:
        """
        Compute theoretical variance at each point
        """
        return points.prod(dim=1)

    def validate_scaling(self, points: torch.Tensor, n_samples: int = 1000) -> dict:
        """
        Validate the scaling properties of the simulation
        """
        simulated = self.simulate(points, n_samples)
        empirical_var = simulated.var(dim=0)
        theoretical_var = self.theoretical_variance(points)
        
        return {
            'empirical_mean': simulated.mean(dim=0).mean().item(),
            'empirical_var_mean': empirical_var.mean().item(),
            'theoretical_var_mean': theoretical_var.mean().item(),
            'relative_error': (empirical_var - theoretical_var).abs().mean().item(),
            'dimension': points.size(1)
        }
    
if __name__ == "__main__":
    bs = BrownianSheet(device='cuda')

    # Test scaling with increasing dimensions
    dims = [2, 4, 8, 16]
    n_points = 5000

    for dim in dims:
        points = torch.rand(n_points, dim, device='cuda')
        stats = bs.validate_scaling(points)
        print(f"\nDimension {dim}:")
        print(f"Empirical Mean: {stats['empirical_mean']:.6f}")
        print(f"Empirical Var: {stats['empirical_var_mean']:.6f}")
        print(f"Theoretical Var: {stats['theoretical_var_mean']:.6f}")
        print(f"Relative Error: {stats['relative_error']:.6f}")