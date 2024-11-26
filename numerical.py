import numpy as np
from fenics import *
import torch
from utils.white_noise import BrownianSheet


def solve_she(t_final, nx=32, ny=32, dt=0.001, n_samples=50, seed=42, sigma=1.0):
    """
    Solve stochastic heat equation with rescaled noise:
    du = Δu dt + σu ∘ dW
    with ∂u/∂n = 0 on boundary
    """
    torch.manual_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    brownian = BrownianSheet(device=device)

    # FEniCS setup
    mesh = UnitSquareMesh(nx, ny)
    V = FunctionSpace(mesh, 'P', 1)
    coords = torch.tensor(mesh.coordinates(), device=device, dtype=torch.float32)

    # Weak form for heat equation
    u = TrialFunction(V)
    v = TestFunction(V)

    # Reduced diffusion coefficient to 0.1
    diffusion_coeff = 0.1
    a = u*v*dx + dt*diffusion_coeff*dot(grad(u), grad(v))*dx
    A = assemble(a)

    solutions = []
    for sample in range(n_samples):
        u_n = Function(V)
        u_n.interpolate(Expression('cos(pi*x[0])*cos(pi*x[1])', degree=2, pi=np.pi))

        t = 0
        n_steps = int(t_final/dt)

        for step in range(n_steps):
            # Heat equation part
            b = assemble(u_n*v*dx)
            u_heat = Function(V)
            solve(A, u_heat.vector(), b)

            # Generate noise
            dW = brownian.simulate(coords, n_samples=1,
                                   seed=seed + sample*n_steps + step).squeeze().cpu().numpy()

            # Combine with adjusted noise scaling
            u_values = u_heat.vector().get_local()
            noise_term = sigma * u_values * dW
            u_new = Function(V)
            u_new.vector()[:] = u_values + noise_term

            u_n.assign(u_new)
            t += dt

        solutions.append(u_n.vector().get_local())

    mean_solution = np.mean(solutions, axis=0)

    # Output processing
    x = np.linspace(0, 1, nx+1)
    y = np.linspace(0, 1, ny+1)
    solution_grid = np.zeros((nx+1, ny+1))

    mean_func = Function(V)
    mean_func.vector()[:] = mean_solution

    for i in range(nx+1):
        for j in range(ny+1):
            try:
                solution_grid[j, i] = mean_func(Point(x[i], y[j]))
            except:
                solution_grid[j, i] = 0

    return solution_grid, x, y


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    times = [0, 0.05, 0.25, 0.5, 0.9, 1]

    params = {
        'dt': 0.01,
        'n_samples': 100,
        'nx': 32,
        'ny': 32,
        'sigma': 1.0  # Noise intensity parameter
    }

    for ax, t in zip(axes.flatten(), times):
        solution, x, y = solve_she(t, **params)
        mean = np.mean(solution)
        std = np.std(solution)

        c = ax.pcolormesh(x, y, solution, shading='auto', cmap='hot')
        fig.colorbar(c, ax=ax)
        ax.set_title(f'SHE solution at t={t}\nMean: {mean:.2e}, Std: {std:.2e}')

    plt.tight_layout()
    plt.show()
