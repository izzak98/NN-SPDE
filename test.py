import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from utils.white_noise import BrownianSheet
from models import DGM
from utils.viz_utils import gen_heat_snapshots

# Set up device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Set up TensorBoard writer
log_dir = Path("runs") / datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir)

sheet = BrownianSheet(device=DEVICE)


def heat_residual_loss_nd(model, t, spatial_coords, alpha, w, use_stochastic=False):
    """
    Compute residual loss for n-dimensional heat equation
    spatial_coords: tensor of shape (batch_size, n_dims)
    """
    t.requires_grad = True
    spatial_coords.requires_grad = True

    u = model(t, spatial_coords)

    # Compute time derivative
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    # Compute spatial derivatives (Laplacian)
    laplacian = 0
    for dim in range(spatial_coords.shape[1]):
        u_x = torch.autograd.grad(u, spatial_coords, grad_outputs=torch.ones_like(
            u), create_graph=True)[0][:, dim:dim+1]
        u_xx = torch.autograd.grad(u_x, spatial_coords, grad_outputs=torch.ones_like(
            u_x), create_graph=True)[0][:, dim:dim+1]
        laplacian += u_xx

    if use_stochastic:
        residual = u_t - alpha * laplacian + (u * w)
    else:
        residual = u_t - alpha * laplacian

    return torch.mean(residual**2)


def initial_condition_loss_nd(model, spatial_coords, u0_func):
    """
    Compute initial condition loss for n-dimensional case
    """
    t0 = torch.zeros(spatial_coords.shape[0], 1, device=spatial_coords.device)
    u0_pred = model(t0, spatial_coords)
    u0_true = u0_func(spatial_coords)
    return torch.mean((u0_pred - u0_true)**2)


def neumann_boundary_condition_loss_nd(model, t, spatial_coords, boundaries):
    """
    Compute Neumann boundary condition loss for n-dimensional case
    boundaries: list of tensors [(min_coords, max_coords)] for each dimension
    """
    total_loss = 0
    n_dims = spatial_coords.shape[1]
    batch_size = spatial_coords.shape[0]

    for dim in range(n_dims):
        # Create boundary points for current dimension
        for boundary_type in ['min', 'max']:
            # Copy spatial coordinates
            boundary_points = spatial_coords.clone()

            # Set the current dimension to its boundary value
            if boundary_type == 'min':
                boundary_points[:, dim] = boundaries[dim][0]
            else:
                boundary_points[:, dim] = boundaries[dim][1]

            boundary_points.requires_grad = True

            # Compute derivative at boundary
            u_boundary = model(t, boundary_points)
            du_dx = torch.autograd.grad(
                u_boundary,
                boundary_points,
                grad_outputs=torch.ones_like(u_boundary),
                create_graph=True
            )[0][:, dim:dim+1]

            total_loss += torch.mean(du_dx**2)

    return total_loss


def compute_losses(model, batch, n_dims):
    """
    Compute all losses for n-dimensional case
    """
    t, spatial_coords, boundaries = [b.to(DEVICE) for b in batch]

    # Define initial condition function for n dimensions
    def u0_func(coords):
        # Example: product of cosines for each dimension
        result = torch.ones(coords.shape[0], 1, device=coords.device)
        for dim in range(coords.shape[1]):
            result *= torch.cos(torch.pi * coords[:, dim:dim+1])
        return result

    loss_initial = initial_condition_loss_nd(model, spatial_coords, u0_func)
    loss_boundary = neumann_boundary_condition_loss_nd(model, t, spatial_coords, boundaries)

    return {
        "initial_loss": loss_initial,
        "boundary_loss": loss_boundary,
    }


def train_heat_equation_nd(model, optimizer, alpha, epochs, batch_size, delta_t, n_dims, num_samples=5):
    model = model.to(DEVICE)
    scaler = torch.amp.GradScaler()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=100, factor=0.5)
    best_loss = float("inf")
    pbar = tqdm(range(epochs), desc="Training")

    for epoch in pbar:
        model.train()

        # Generate batch data for n dimensions
        t = torch.rand((batch_size, 1), device=DEVICE)
        spatial_coords = torch.rand((batch_size, n_dims), device=DEVICE)

        # Generate boundaries for each dimension
        boundaries = [(torch.zeros(1, device=DEVICE), torch.ones(1, device=DEVICE))
                      for _ in range(n_dims)]

        total_residual_loss = torch.tensor(0, device=DEVICE, dtype=torch.float32)

        # Generate stochastic terms
        ws = [sheet.simulate(torch.cat([t, spatial_coords], dim=1)) for _ in range(num_samples)]

        for i in range(num_samples):
            w = ws[i]
            residual_loss = heat_residual_loss_nd(
                model, t, spatial_coords, alpha, w, use_stochastic=True)
            total_residual_loss += residual_loss

        avg_residual_loss = total_residual_loss / num_samples

        batch = (t, spatial_coords, boundaries)
        losses = compute_losses(model, batch, n_dims)
        losses["Avg. Residual Loss"] = avg_residual_loss
        losses["total_loss"] = avg_residual_loss + losses["initial_loss"] + losses["boundary_loss"]

        loss = losses["total_loss"]

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        scheduler.step(loss)
        pbar.set_postfix({key: f"{value.item():.2e}" for key, value in losses.items()})

        if loss < best_loss:
            best_loss = loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            }, log_dir / "best_model.pt")

    writer.close()


if __name__ == "__main__":
    # Parameters
    n_dims = 3  # Change this to the desired number of dimensions
    L = 1.0
    T = 1.0
    alpha = 0.1
    delta_t = 0.01
    torch.autograd.set_detect_anomaly(True)

    # Initialize model and optimizer
    model = DGM(
        input_dims=n_dims,  # Changed to match n_dims
        hidden_dims=[128, 128, 64],
        dgm_dims=0,
        n_dgm_layers=3,
        hidden_activation="tanh",
        output_activation=None,
    )

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # Train the model
    train_heat_equation_nd(
        model,
        optimizer,
        alpha=alpha,
        epochs=1000,
        batch_size=2048,
        delta_t=delta_t,
        n_dims=n_dims,
    )

    # Note: Visualization needs to be modified for n>3 dimensions
    if n_dims <= 3:
        gen_heat_snapshots(
            model,
            grid_size=100,
            time_steps=[0.1, 0.25, 0.5, 0.9],
            name=f"DGM_Stochastic_{n_dims}D",
        )
