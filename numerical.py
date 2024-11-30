import numpy as np
import matplotlib.pyplot as plt


def initialize_domain(Lx, Ly, Nx, Ny):
    """Initialize spatial grid and initial condition."""
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y)
    u = np.cos(np.pi * X) * np.cos(np.pi * Y)
    return x, y, u, np.zeros_like(u)


def apply_neumann_bc(u):
    """Apply Neumann boundary conditions (zero flux)."""
    u[0, :] = u[1, :]          # dT/dx = 0 at x=0
    u[-1, :] = u[-2, :]        # dT/dx = 0 at x=Lx
    u[:, 0] = u[:, 1]          # dT/dy = 0 at y=0
    u[:, -1] = u[:, -2]        # dT/dy = 0 at y=Ly


def finite_difference_step(u, u_new, alpha, dt, dx, dy):
    """Perform a single time step using the finite difference method."""
    Nx, Ny = u.shape
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            u_new[i, j] = u[i, j] + alpha * dt * (
                (u[i+1, j] - 2*u[i, j] + u[i-1, j]) / dx**2 +
                (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / dy**2
            )
    apply_neumann_bc(u_new)


def time_step_loop(u, u_new, alpha, dt, dx, dy, steps):
    """Iterate through time steps and collect snapshots."""
    u_snapshots = [u.copy()]
    step = 0
    while step < max(steps):
        finite_difference_step(u, u_new, alpha, dt, dx, dy)
        u, u_new = u_new, u  # Swap references
        step += 1
        if step in steps:
            u_snapshots.append(u.copy())
    return u_snapshots


def compute_analytical_solution(x, y, t_values, alpha):
    """Compute the analytical solution for given times."""
    X, Y = np.meshgrid(x, y)
    analytical_solutions = []
    for t in t_values:
        u_analytical = np.cos(np.pi * X) * np.cos(np.pi * Y) * np.exp(-2 * np.pi**2 * alpha * t)
        analytical_solutions.append(u_analytical)
    return analytical_solutions


def plot_comparison(u_snapshots, analytical_solutions, t_values, Lx, Ly):
    """Plot numerical and analytical solutions side by side."""
    fig, axes = plt.subplots(2, len(t_values), figsize=(15, 8))
    for idx, t in enumerate(t_values):
        # Numerical solution
        im_num = axes[0, idx].imshow(u_snapshots[idx], extent=[0, Lx, 0, Ly],
                                     origin='lower', cmap='hot', vmin=-1, vmax=1)
        axes[0, idx].set_title(f"Numerical (t={t:.2f})")
        plt.colorbar(im_num, ax=axes[0, idx])

        # Analytical solution
        im_ana = axes[1, idx].imshow(analytical_solutions[idx], extent=[0, Lx, 0, Ly],
                                     origin='lower', cmap='hot', vmin=-1, vmax=1)
        axes[1, idx].set_title(f"Analytical (t={t:.2f})")
        plt.colorbar(im_ana, ax=axes[1, idx])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Parameters
    Lx, Ly = 1.0, 1.0       # Domain size
    Nx, Ny = 50, 50         # Number of grid points
    dx, dy = Lx/(Nx-1), Ly/(Ny-1)  # Grid spacing
    alpha = 0.1             # Diffusion coefficient
    dt = 0.0005             # Time step
    t_values = [0.1, 0.25, 0.5, 0.9]  # Time points for plotting
    steps = [int(t/dt) for t in t_values]   # Corresponding time steps

    # Main execution
    x, y, u, u_new = initialize_domain(Lx, Ly, Nx, Ny)
    u_snapshots = time_step_loop(u, u_new, alpha, dt, dx, dy, steps)
    analytical_solutions = compute_analytical_solution(x, y, t_values, alpha)
    plot_comparison(u_snapshots, analytical_solutions, t_values, Lx, Ly)
