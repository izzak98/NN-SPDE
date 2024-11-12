import time
import torch
import numpy as np
from tqdm import tqdm
from utils.white_noise import simulate_dot_W_spectral

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HeatDataGenerator:
    def __init__(self, d, modes=10, base_seed=42, n_points=5000):
        self.d = d
        self.modes = modes
        self.base_seed = base_seed
        self.n_points = n_points
        self.current_epoch = 0
        self.DEVICE = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Pre-allocate generators
        self.base_gen = torch.Generator(device=self.DEVICE)
        self.bound_gen = torch.Generator(device=self.DEVICE)
        self.init_gen = torch.Generator(device=self.DEVICE)

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def get_epoch_seed(self):
        return self.base_seed + self.current_epoch

    def generate_all_points(self, p_base, p_bound, p_init):
        """Generate all points in a single batch for maximum efficiency"""
        epoch_seed = self.get_epoch_seed()

        # Set seeds for all generators
        self.base_gen.manual_seed(epoch_seed)
        self.bound_gen.manual_seed(epoch_seed + 1)
        self.init_gen.manual_seed(epoch_seed + 2)

        # Calculate sizes
        n_base = int(self.n_points * p_base)
        n_bound = int(self.n_points * p_bound)
        n_init = self.n_points - n_base - n_bound

        points_list = []
        masks = []

        # Generate base points
        if n_base > 0:
            t_base = torch.rand(
                (n_base, 1), generator=self.base_gen, device=self.DEVICE, requires_grad=True)
            X_base = torch.rand(
                (n_base, self.d), generator=self.base_gen, device=self.DEVICE, requires_grad=True)
            points_list.append(torch.cat([t_base, X_base], dim=1))
            masks.append(torch.zeros(
                n_base, dtype=torch.long, device=self.DEVICE))

        # Generate boundary points - Modified to avoid in-place operations
        if n_bound > 0:
            t_bound = torch.rand(
                (n_bound, 1), generator=self.bound_gen, device=self.DEVICE, requires_grad=True)
            # No requires_grad yet
            X_bound = torch.rand(
                (n_bound, self.d), generator=self.bound_gen, device=self.DEVICE)

            # Generate boundary conditions
            bound_dims = torch.randint(
                0, self.d, (n_bound,), generator=self.bound_gen, device=self.DEVICE)
            bound_values = torch.randint(
                0, 2, (n_bound,), generator=self.bound_gen, device=self.DEVICE).float()

            # Create mask for boundary dimensions
            mask = torch.zeros_like(X_bound)
            indices = torch.arange(n_bound, device=self.DEVICE)
            mask[indices, bound_dims] = 1

            # Apply boundary conditions without in-place operations
            X_bound = X_bound * (1 - mask) + bound_values.unsqueeze(1) * mask
            # Enable gradients after modifications
            X_bound.requires_grad_(True)

            points_list.append(torch.cat([t_bound, X_bound], dim=1))
            masks.append(torch.ones(
                n_bound, dtype=torch.long, device=self.DEVICE))

        # Generate initial points
        if n_init > 0:
            t_init = torch.zeros(
                (n_init, 1), device=self.DEVICE, requires_grad=True)
            X_init = torch.rand(
                (n_init, self.d), generator=self.init_gen, device=self.DEVICE, requires_grad=True)
            points_list.append(torch.cat([t_init, X_init], dim=1))
            masks.append(
                2 * torch.ones(n_init, dtype=torch.long, device=self.DEVICE))

        # Concatenate all points and masks
        all_points = torch.cat(points_list, dim=0)

        # Generate noise for all points at once
        W_d = simulate_dot_W_spectral(
            all_points, self.modes, epoch_seed).unsqueeze(1)

        # Split results back into categories
        results = []
        start_idx = 0
        for n in [n_base, n_bound, n_init]:
            if n > 0:
                end_idx = start_idx + n
                points = all_points[start_idx:end_idx]
                results.append((
                    points[:, 0].unsqueeze(1),  # t
                    points[:, 1:],              # X
                    W_d[start_idx:end_idx]      # W_d
                ))
                start_idx = end_idx
            else:
                results.append(
                    (torch.empty(0), torch.empty(0), torch.empty(0)))

        return results

    def __call__(self, p_base=0.6, p_bound=0.2, p_init=0.2):
        """Main call method to generate all points"""
        assert abs(p_base + p_bound + p_init -
                   1.0) < 1e-6, "Proportions must sum to 1"
        return self.generate_all_points(p_base, p_bound, p_init)


def test_heat():
    # Initialize the generator
    generator = HeatDataGenerator(d=2)  # for 2D spatial domain

    # Generate points
    import time
    start_time = time.time()
    base_points, boundary_points, initial_points = generator(
        p_base=0.34,
        p_bound=0.33,
        p_init=0.33
    )
    print(f"Time taken: {time.time() - start_time:.2f} seconds")

    # Each returned tuple contains (t, X, W_d) for that category
    t_base, X_base, W_d_base = base_points
    t_bound, X_bound, W_d_bound = boundary_points
    t_init, X_init, W_d_init = initial_points

    print(
        f"Base points: t{t_base.shape}, X{X_base.shape}, W_d{W_d_base.shape}")
    print(
        f"Boundary points: t{t_bound.shape}, X{X_bound.shape}, W_d{W_d_bound.shape}")
    print(
        f"Initial points: t{t_init.shape}, X{X_init.shape}, W_d{W_d_init.shape}")

class BurgerDataGenerator:
    def __init__(self, d, modes=10, base_seed=42, n_points=5000):
        self.d = d
        self.modes = modes
        self.base_seed = base_seed
        self.n_points = n_points
        self.current_epoch = 0
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Pre-allocate generators
        self.base_gen = torch.Generator(device=self.DEVICE)
        self.bound_gen = torch.Generator(device=self.DEVICE)
        self.init_gen = torch.Generator(device=self.DEVICE)
        
    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def get_epoch_seed(self):
        return self.base_seed + self.current_epoch
        
    def generate_log_uniform(self, size, generator):
        """Generate samples from log-uniform distribution between ln(0.001) and ln(1)"""
        log_min = np.log(0.001)
        log_max = np.log(1.0)
        uniform = torch.rand(size, generator=generator, device=self.DEVICE)
        return torch.exp(log_min + (log_max - log_min) * uniform)

    def generate_paired_points(self, n_pairs, generator, t_val=None, is_boundary=False):
        """
        Generate pairs of points where one point has an inverted dimension
        Args:
            n_pairs: number of pairs to generate
            generator: random number generator
            t_val: if provided, use this t value instead of random
            is_boundary: if True, set dimension to 1 before inverting
        """
        # Generate time values (or use provided)
        if t_val is None:
            t = torch.rand((n_pairs, 1), generator=generator, device=self.DEVICE, requires_grad=True)
        else:
            t = t_val * torch.ones((n_pairs, 1), device=self.DEVICE, requires_grad=True)
        
        # Generate spatial coordinates
        X = torch.rand((n_pairs, self.d), generator=generator, device=self.DEVICE)
        
        # Generate random dimensions to invert
        dims = torch.randint(0, self.d, (n_pairs,), generator=generator, device=self.DEVICE)
        
        # Create paired points
        X_pairs = X.clone()
        
        # Create masks for the selected dimensions
        dim_mask = torch.zeros_like(X)
        indices = torch.arange(n_pairs, device=self.DEVICE)
        dim_mask[indices, dims] = 1
        
        if is_boundary:
            # For boundary points, set selected dimension to 1 first
            X = X * (1 - dim_mask) + dim_mask
            X_pairs = X * (1 - dim_mask)  # inverse will be 0
        else:
            # For non-boundary points, just take inverse of selected dimension
            X_pairs = X * (1 - dim_mask) + (1 - X * dim_mask) * dim_mask
        
        # Duplicate time values for pairs
        t_full = t.repeat_interleave(2, dim=0)
        
        # Stack points in consecutive order (pair1_point1, pair1_point2, pair2_point1, pair2_point2, ...)
        X_full = torch.zeros((2 * n_pairs, self.d), device=self.DEVICE)
        X_full[0::2] = X
        X_full[1::2] = X_pairs
        
        # Generate parameters for all points
        nu = self.generate_log_uniform((2 * n_pairs, 1), generator)
        alpha = self.generate_log_uniform((2 * n_pairs, 1), generator)
        
        return t_full, X_full, nu, alpha

    def generate_all_points(self, p_base, p_bound, p_init):
        """Generate all points in a single batch for maximum efficiency"""
        epoch_seed = self.get_epoch_seed()

        # Set seeds for all generators
        self.base_gen.manual_seed(epoch_seed)
        self.bound_gen.manual_seed(epoch_seed + 1)
        self.init_gen.manual_seed(epoch_seed + 2)

        # Calculate number of pairs for each category
        n_base_pairs = int(self.n_points * p_base) // 2
        n_bound_pairs = int(self.n_points * p_bound) // 2
        n_init_pairs = int(self.n_points * p_init) // 2

        points_list = []
        params_list = []
        masks = []

        # Generate base points
        if n_base_pairs > 0:
            t_base, X_base, nu_base, alpha_base = self.generate_paired_points(
                n_base_pairs, self.base_gen)
            points_list.append(torch.cat([t_base, X_base], dim=1))
            params_list.append(torch.cat([nu_base, alpha_base], dim=1))
            masks.append(torch.zeros(2 * n_base_pairs, dtype=torch.long, device=self.DEVICE))

        # Generate boundary points
        if n_bound_pairs > 0:
            t_bound, X_bound, nu_bound, alpha_bound = self.generate_paired_points(
                n_bound_pairs, self.bound_gen, is_boundary=True)
            points_list.append(torch.cat([t_bound, X_bound], dim=1))
            params_list.append(torch.cat([nu_bound, alpha_bound], dim=1))
            masks.append(torch.ones(2 * n_bound_pairs, dtype=torch.long, device=self.DEVICE))

        # Generate initial points
        if n_init_pairs > 0:
            t_init, X_init, nu_init, alpha_init = self.generate_paired_points(
                n_init_pairs, self.init_gen, t_val=0.0)
            points_list.append(torch.cat([t_init, X_init], dim=1))
            params_list.append(torch.cat([nu_init, alpha_init], dim=1))
            masks.append(2 * torch.ones(2 * n_init_pairs, dtype=torch.long, device=self.DEVICE))

        # Concatenate all points and masks
        all_points = torch.cat(points_list, dim=0)
        all_params = torch.cat(params_list, dim=0)
        all_masks = torch.cat(masks, dim=0)

        # Generate noise for all points at once
        W_d = simulate_dot_W_spectral(all_points, self.modes, epoch_seed).unsqueeze(1)

        # Split results back into categories
        results = []
        start_idx = 0
        for n in [2 * n_base_pairs, 2 * n_bound_pairs, 2 * n_init_pairs]:
            if n > 0:
                end_idx = start_idx + n
                points = all_points[start_idx:end_idx]
                params = all_params[start_idx:end_idx]
                results.append((
                    points[:, 0].unsqueeze(1),  # t
                    points[:, 1:],              # X
                    W_d[start_idx:end_idx],     # W_d
                    params[:, 0].unsqueeze(1),  # nu
                    params[:, 1].unsqueeze(1)   # alpha
                ))
                start_idx = end_idx
            else:
                results.append((torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0)))

        return results

    def __call__(self, p_base=0.6, p_bound=0.2, p_init=0.2):
        """Main call method to generate all points"""
        assert abs(p_base + p_bound + p_init - 1.0) < 1e-6, "Proportions must sum to 1"
        return self.generate_all_points(p_base, p_bound, p_init)

def test_burgers():
    # Initialize the generator
    generator = BurgerDataGenerator(d=3)  # for 2D spatial domain
    
    # Generate points
    import time
    start_time = time.time()
    base_points, boundary_points, initial_points = generator(
        p_base=0.7,
        p_bound=0.2,
        p_init=0.1
    )
    print(f"Time taken: {time.time() - start_time:.2f} seconds")

    # Each returned tuple contains (t, X, W_d, nu, alpha) for that category
    t_base, X_base, W_d_base, nu_base, alpha_base = base_points
    t_bound, X_bound, W_d_bound, nu_bound, alpha_bound = boundary_points
    t_init, X_init, W_d_init, nu_init, alpha_init = initial_points

    print("\nBase points:")
    print(f"  t: {t_base.shape}")
    print(f"  X: {X_base.shape}")
    print(f"  W_d: {W_d_base.shape}")
    print(f"  nu: {nu_base.shape}")
    print(f"  alpha: {alpha_base.shape}")
    
    # Check first pair for base points
    if X_base.shape[0] > 0:
        print("\nExample base pair:")
        print(f"  t values: {t_base[0].item():.3f}, {t_base[1].item():.3f}")
        print(f"  Point 1: {X_base[0]}")
        print(f"  Point 2: {X_base[1]}")
        diff = torch.abs(X_base[0] - X_base[1])
        inv_dim = torch.where(diff > 0)[0]
        print(f"  Inverted dimension: {inv_dim.item()}")

    print("\nBoundary points:")
    print(f"  t: {t_bound.shape}")
    print(f"  X: {X_bound.shape}")
    print(f"  W_d: {W_d_bound.shape}")
    print(f"  nu: {nu_bound.shape}")
    print(f"  alpha: {alpha_bound.shape}")

    # Check first pair for boundary points
    if X_bound.shape[0] > 0:
        print("\nExample boundary pair:")
        print(f"  t values: {t_bound[0].item():.3f}, {t_bound[1].item():.3f}")
        print(f"  Point 1: {X_bound[0]}")
        print(f"  Point 2: {X_bound[1]}")
        diff = torch.abs(X_bound[0] - X_bound[1])
        periodic_dim = torch.where(diff > 0.9)[0]
        print(f"  Periodic dimension: {periodic_dim.item()}")
        print(f"  Values in periodic dimension: {X_bound[0][periodic_dim].item():.3f}, {X_bound[1][periodic_dim].item():.3f}")

    print("\nInitial points:")
    print(f"  t: {t_init.shape}")
    print(f"  X: {X_init.shape}")
    print(f"  W_d: {W_d_init.shape}")
    print(f"  nu: {nu_init.shape}")
    print(f"  alpha: {alpha_init.shape}")

    # Check first pair for initial points
    if X_init.shape[0] > 0:
        print("\nExample initial pair:")
        print(f"  t values: {t_init[0].item():.3f}, {t_init[1].item():.3f}")
        print(f"  Point 1: {X_init[0]}")
        print(f"  Point 2: {X_init[1]}")
        diff = torch.abs(X_init[0] - X_init[1])
        inv_dim = torch.where(diff > 0)[0]
        print(f"  Inverted dimension: {inv_dim.item()}")

    # Verify parameter ranges
    print("\nParameter ranges:")
    print(f"nu range: [{nu_base.min():.3e}, {nu_base.max():.3e}]")
    print(f"alpha range: [{alpha_base.min():.3e}, {alpha_base.max():.3e}]")

if __name__ == "__main__":
    test_heat()
    # test_burgers()
