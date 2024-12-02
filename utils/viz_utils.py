import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Dict, List, Tuple

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gen_heat_snapshots(model, grid_size=100, time_steps=[0.1, 0.25, 0.5, 0.9], name=None) -> Tuple[plt.Figure, Dict]:
    if name is None:
        name = model.__name__

    # Store statistics for each timestep
    stats = {t: {} for t in time_steps}

    # Define the grid size and spatial coordinates
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.vstack([xx.ravel(), yy.ravel()]).T

    # Convert spatial grid to torch tensor
    grid_tensor = torch.tensor(grid_points).float().to(DEVICE)

    # Precompute time tensors for the specified time steps
    time_tensors = [torch.full((grid_tensor.shape[0], 1), t, device=DEVICE) for t in time_steps]

    # Create subplots for the snapshots in a grid layout
    fig, axes = plt.subplots(1, len(time_steps), figsize=(20, 10), constrained_layout=True)

    for i, (ax, time_tensor, t) in enumerate(zip(axes, time_tensors, time_steps)):
        with torch.no_grad():
            output = model(time_tensor, grid_tensor[:, 0:1], grid_tensor[:, 1:2])
            if isinstance(output, tuple):
                output = output[0]
        solution = output.cpu().numpy().reshape(grid_size, grid_size)

        # Check for symmetries
        temp_t = torch.tensor([[t]]).float().to(DEVICE)
        c1 = model(temp_t, torch.tensor([[0.99]]).to(DEVICE), torch.tensor([[0.01]]).to(DEVICE))
        c2 = model(temp_t, torch.tensor([[0.01]]).to(DEVICE), torch.tensor([[0.99]]).to(DEVICE))

        c3 = model(temp_t, torch.tensor([[0.99]]).to(DEVICE), torch.tensor([[0.99]]).to(DEVICE))
        c4 = model(temp_t, torch.tensor([[0.01]]).to(DEVICE), torch.tensor([[0.01]]).to(DEVICE))

        if isinstance(c1, tuple):
            c1 = c1[0]
            c2 = c2[0]
            c3 = c3[0]
            c4 = c4[0]

        diagonal_symmetry = abs(c1 - c2).item()
        corner_asymmetry = abs(c3 - c4).item()

        # Store statistics
        stats[t] = {
            'mean': float(np.mean(solution)),
            'diagonal_diff': float(diagonal_symmetry),
            'corner_diff': float(corner_asymmetry),
            'min': float(np.min(solution)),
            'max': float(np.max(solution)),
            'std': float(np.std(solution))
        }

        # Plot the solution
        im = ax.imshow(solution, extent=(0, 1, 0, 1), origin="lower", cmap="hot", vmin=-1 *
                       np.sqrt(2) * np.log10(np.exp(2)), vmax=1*np.sqrt(2) * np.log10(np.exp(2)))
        ax.set_title(
            f"t = {t:.2f}\nMean: {stats[t]['mean']:.2e}\n[0.01, 0.99] - [0.99, 0.01]: {stats[t]['diagonal_diff']:.2e}\n[0.99, 0.99] [0.01, 0.01]: {stats[t]['corner_diff']:.2e}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    # Add a single shared color bar for all subplots
    cbar = fig.colorbar(im, ax=axes, location='right', shrink=0.75, label="Temperature")

    plt.savefig(f"{name}_heat_snapshots.png")
    plt.suptitle("Stochastic Heat Equation Snapshots", fontsize=16)

    # Generate summary string
    summary = "Heat Equation Solution Statistics:\n\n"
    for t in time_steps:
        summary += f"Time t={t:.2f}:\n"
        summary += f"  Mean: {stats[t]['mean']:.2e}\n"
        summary += f"  [0.01, 0.99] - [0.99, 0.01]: {stats[t]['diagonal_diff']:.2e}\n"
        summary += f"  [0.99, 0.99] - [0.01, 0.01]: {stats[t]['corner_diff']:.2e}\n"
        summary += f"  Min: {stats[t]['min']:.2e}\n"
        summary += f"  Max: {stats[t]['max']:.2e}\n"
        summary += f"  Std: {stats[t]['std']:.2e}\n"

    print("\nStatistics Summary:")
    print(summary)

    plt.show()
    return fig, stats


if __name__ == "__main__":
    def analytical_solution(t, x, y):
        t = t[0]
        # x, y = x[:, 0], x[:, 1]
        return torch.cos(np.pi*x)*torch.cos(np.pi*y)*torch.exp(-(np.pi**2)*t*0.1)

    gen_heat_snapshots(analytical_solution, grid_size=100, time_steps=[
                       0.1, 0.25, 0.5, 0.9], name="Analytical Solution")
