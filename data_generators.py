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

if __name__ == "__main__":
    import time
    data_generator = HeatDataGenerator(N_total=256, d=16, p_interior=0.7, p_boundary=0.2)
    
    start_time = time.time()
    t, X, W_d = data_generator()
    print(f"Time taken: {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    t, X, W_d = data_generator()
    print(f"Time taken: {time.time() - start_time:.2f} seconds")
    print(t.shape, X.shape, W_d.shape)
    
    # Count points in each category
    interior_count = ((0 <= X) & (X <= 1)).all(dim=1).sum().item()
    boundary_count = ((X == 0) | (X == 1)).any(dim=1).sum().item() - ((X < 0) | (X > 1)).any(dim=1).sum().item()
    exterior_count = ((X < 0) | (X > 1)).any(dim=1).sum().item()
    
    print(f"Interior points: {interior_count}")
    print(f"Boundary points: {boundary_count}")
    print(f"Exterior points: {exterior_count}")