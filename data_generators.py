import time
import torch
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
        m=5000,
        p_base=0.7,
        p_bound=0.2,
        p_init=0.1
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


class BurgersDataGenerator:
    def __init__(self, N_total, d, nu_range=(0.001, 1), alpha_range=(0.001, 1), p_interior=0.8, p_boundary=0.2, modes=10):
        """
        N_total: Total number of points to generate
        d: Number of spatial dimensions
        nu_range: Tuple of (min_nu, max_nu) for viscosity sampling
        p_interior: Probability of generating an interior point
        p_boundary: Probability of generating a boundary point
        Note: For periodic BCs, we don't need exterior points
        """
        self.N_total = N_total
        self.d = d
        self.nu_min, self.nu_max = nu_range
        self.log_nu_min = torch.log(torch.tensor(self.nu_min))
        self.log_nu_max = torch.log(torch.tensor(self.nu_max))
        self.alpha_min, self.alpha_max = alpha_range
        self.log_alpha_min = torch.log(torch.tensor(self.alpha_min))
        self.log_alpha_max = torch.log(torch.tensor(self.alpha_max))
        self.p_interior = p_interior
        self.p_boundary = p_boundary
        assert abs(p_interior + p_boundary -
                   1.0) < 1e-6, "Probabilities must sum to 1"
        self.modes = modes

    def __call__(self):
        # Generate random categorization for each point
        categories = torch.rand(self.N_total)
        boundary_mask = categories >= self.p_interior

        # Generate time points
        t = torch.rand((self.N_total, 1), requires_grad=True).to(DEVICE)

        # Generate spatial points in [0,1]^d
        X = torch.rand((self.N_total, self.d), requires_grad=True).to(DEVICE)

        # Sample viscosity values (one for each point)
        log_nu = torch.rand(self.N_total, 1) * \
            (self.log_nu_max - self.log_nu_min) + self.log_nu_min
        nu = torch.exp(log_nu).to(DEVICE)

        log_alpha = torch.rand(
            self.N_total, 1) * (self.log_alpha_max - self.log_alpha_min) + self.log_alpha_min
        alpha = torch.exp(log_alpha).to(DEVICE)

        # Adjust boundary points for periodic boundary conditions
        boundary_indices = torch.where(boundary_mask)[0]
        if len(boundary_indices) > 0:
            # Randomly choose which dimension will be at the boundary
            boundary_dim = torch.randint(
                0, self.d, (len(boundary_indices),)).to(DEVICE)
            # Randomly set to either 0 or 1 for periodic boundary
            X[boundary_indices, boundary_dim] = torch.randint(
                0, 2, (len(boundary_indices),)).float().to(DEVICE)

            # For each boundary point at x_i = 1, create a corresponding point at x_i = 0
            # and vice versa to enforce periodic boundary conditions
            periodic_pairs = X[boundary_indices].clone()
            periodic_pairs[periodic_pairs == 0] = 1
            periodic_pairs[periodic_pairs == 1] = 0

            X = torch.cat([X, periodic_pairs], dim=0)
            t = torch.cat([t, t[boundary_indices]], dim=0)
            # Use same nu for periodic pairs
            nu = torch.cat([nu, nu[boundary_indices]], dim=0)
            alpha = torch.cat([alpha, alpha[boundary_indices]], dim=0)
            boundary_mask = torch.cat(
                [boundary_mask, boundary_mask[boundary_indices]], dim=0)

        # Simulate space-time white noise for all points
        W_d = simulate_dot_W_spectral(
            [(t_i.item(), *X_i) for t_i, X_i in zip(t, X)], modes=self.modes)
        W_d = W_d.reshape(-1, 1).to(DEVICE)

        return t, X, W_d, boundary_mask.to(DEVICE), nu, alpha


def test_burgers():
    data_generator = BurgersDataGenerator(
        N_total=256, d=16, nu_range=(0.001, 1.5))

    start_time = time.time()
    t, X, W_d, boundary_mask, nu, alpha = data_generator()
    print(f"Time taken: {time.time() - start_time:.2f} seconds")

    print(f"Shapes: t{t.shape}, X{X.shape}, W_d{W_d.shape}, boundary_mask{boundary_mask.shape}, nu{nu.shape}, alpha{alpha.shape}")
    print(f"\nViscosity range: [{nu.min().item():.3f}, {nu.max().item():.3f}]")

    # Count points in each category
    interior_count = (~boundary_mask).sum().item()
    boundary_count = boundary_mask.sum().item()

    print(f"\nInterior points: {interior_count}")
    print(f"Boundary points: {boundary_count}")

    # Verify periodic boundary conditions
    boundary_points = X[boundary_mask]
    print("\nSample boundary points and their viscosities:")
    for i in range(min(5, len(boundary_points))):
        print(
            f"Point {i}: {boundary_points[i]}, nu: {nu[boundary_mask][i].item():.3f}")


if __name__ == "__main__":
    test_heat()
