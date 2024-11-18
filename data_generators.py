import time
import torch
import numpy as np
from tqdm import tqdm
from utils.white_noise import BrownianSheet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BS = BrownianSheet(device=DEVICE)

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

    def generate_all_points(self, p_base, p_bound, p_init, white_noise_seed=None):
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
        if white_noise_seed is None:
            white_noise_seed = epoch_seed
        W_d = BS.simulate(
            all_points, seed=white_noise_seed).unsqueeze(1)

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

    def __call__(self, p_base=0.6, p_bound=0.2, p_init=0.2, white_noise_seed=None):
        """Main call method to generate all points"""
        assert abs(p_base + p_bound + p_init -
                   1.0) < 1e-6, "Proportions must sum to 1"
        return self.generate_all_points(p_base, p_bound, p_init, white_noise_seed)


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
    print(f"base_points mean: {X_base.mean().item()}, std: {X_base.std().item()}")
    print(
        f"Boundary points: t{t_bound.shape}, X{X_bound.shape}, W_d{W_d_bound.shape}")
    print(
        f"Initial points: t{t_init.shape}, X{X_init.shape}, W_d{W_d_init.shape}")

if __name__ == "__main__":
    test_heat()
    # test_burgers()
