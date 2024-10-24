import time
import torch
from utils.white_noise import simulate_dot_W_spectral

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HeatDataGenerator:
    def __init__(self, N_total, d, p_interior=0.7, p_boundary=0.2, modes=10):
        """
        N_total: Total number of points to generate
        d: Number of spatial dimensions
        p_interior: Probability of generating an interior point
        p_boundary: Probability of generating a boundary point
        Note: p_exterior = 1 - p_interior - p_boundary
        """
        self.N_total = N_total
        self.d = d
        self.p_interior = p_interior
        self.p_boundary = p_boundary
        self.p_exterior = 1 - p_interior - p_boundary
        assert 0 <= self.p_exterior <= 1, "Invalid probabilities"
        self.modes = modes

    def __call__(self):
        # Generate random categorization for each point
        categories = torch.rand(self.N_total)
        boundary_mask = (self.p_interior <= categories) & (categories < self.p_interior + self.p_boundary)
        exterior_mask = categories >= self.p_interior + self.p_boundary

        # Generate time points
        t = torch.rand((self.N_total, 1), requires_grad=True).to(DEVICE)

        # Generate spatial points
        X = torch.rand((self.N_total, self.d), requires_grad=True).to(DEVICE)

        # Adjust boundary points
        boundary_indices = torch.where(boundary_mask)[0]
        boundary_dim = torch.randint(0, self.d, (len(boundary_indices),)).to(DEVICE)
        X[boundary_indices, boundary_dim] = torch.randint(0, 2, (len(boundary_indices),)).float().to(DEVICE)

        # Adjust exterior points
        exterior_indices = torch.where(exterior_mask)[0]
        shift = (torch.rand((len(exterior_indices), self.d)) * 0.2 + 0.1).to(DEVICE)
        signs = (torch.randint(0, 2, (len(exterior_indices), self.d)) * 2 - 1).to(DEVICE)
        X[exterior_indices] = X[exterior_indices] + signs * shift

        # Simulate space-time white noise for all points
        W_d = simulate_dot_W_spectral([(t_i.item(), *X_i) for t_i, X_i in zip(t, X)], modes=self.modes)
        W_d = W_d.reshape(-1, 1).to(DEVICE)

        dgm_boundry_mask = ((X <= 0) | (X >= 1)).any(dim=1).to(DEVICE)
        return t, X, W_d, dgm_boundry_mask

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
        assert abs(p_interior + p_boundary - 1.0) < 1e-6, "Probabilities must sum to 1"
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
        log_nu = torch.rand(self.N_total, 1) * (self.log_nu_max - self.log_nu_min) + self.log_nu_min
        nu = torch.exp(log_nu).to(DEVICE)
        
        log_alpha = torch.rand(self.N_total, 1) * (self.log_alpha_max - self.log_alpha_min) + self.log_alpha_min
        alpha = torch.exp(log_alpha).to(DEVICE)

        # Adjust boundary points for periodic boundary conditions
        boundary_indices = torch.where(boundary_mask)[0]
        if len(boundary_indices) > 0:
            # Randomly choose which dimension will be at the boundary
            boundary_dim = torch.randint(0, self.d, (len(boundary_indices),)).to(DEVICE)
            # Randomly set to either 0 or 1 for periodic boundary
            X[boundary_indices, boundary_dim] = torch.randint(0, 2, (len(boundary_indices),)).float().to(DEVICE)
            
            # For each boundary point at x_i = 1, create a corresponding point at x_i = 0
            # and vice versa to enforce periodic boundary conditions
            periodic_pairs = X[boundary_indices].clone()
            periodic_pairs[periodic_pairs == 0] = 1
            periodic_pairs[periodic_pairs == 1] = 0
            
            X = torch.cat([X, periodic_pairs], dim=0)
            t = torch.cat([t, t[boundary_indices]], dim=0)
            nu = torch.cat([nu, nu[boundary_indices]], dim=0)  # Use same nu for periodic pairs
            alpha = torch.cat([alpha, alpha[boundary_indices]], dim=0)
            boundary_mask = torch.cat([boundary_mask, boundary_mask[boundary_indices]], dim=0)

        # Simulate space-time white noise for all points
        W_d = simulate_dot_W_spectral([(t_i.item(), *X_i) for t_i, X_i in zip(t, X)], modes=self.modes)
        W_d = W_d.reshape(-1, 1).to(DEVICE)

        return t, X, W_d, boundary_mask.to(DEVICE), nu, alpha

def test_heat():
    data_generator = HeatDataGenerator(N_total=256, d=16)

    start_time = time.time()
    t, X, W_d, boundary_mask = data_generator()
    print(f"Time taken: {time.time() - start_time:.2f} seconds")

    print(f"Shapes: t{t.shape}, X{X.shape}, W_d{W_d.shape}, boundary_mask{boundary_mask.shape}")

    # Count points in each category
    interior_count = (~boundary_mask).sum().item()
    boundary_count = boundary_mask.sum().item()

    print(f"\nInterior points: {interior_count}")
    print(f"Boundary points: {boundary_count}")

    print("\nSample boundary points:")
    for i in range(min(5, boundary_count)):
        print(f"Point {i}: {X[boundary_mask][i]}")


def test_burgers():
    data_generator = BurgersDataGenerator(N_total=256, d=16, nu_range=(0.001, 1.5))
    
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
        print(f"Point {i}: {boundary_points[i]}, nu: {nu[boundary_mask][i].item():.3f}")

if __name__ == "__main__":
    test_heat()
    