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


def heat_residual_loss_nd(model, t, x, alpha, w, use_stochastic=False):
    """
    Compute residual loss for n-dimensional heat equation
    Args:
        t: time tensor (batch_size, 1)
        x: spatial coordinates tensor (batch_size, n_dims)
        alpha: diffusion coefficient
        w: noise tensor
        use_stochastic: whether to include stochastic term
    """
    t.requires_grad = True
    x.requires_grad = True

    u = model(t, x)

    # Compute time derivative
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    # Compute spatial derivatives for each dimension
    laplacian_u = torch.zeros_like(u)
    for dim in range(x.shape[1]):
        u_i = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                  create_graph=True)[0][:, dim:dim+1]
        u_ii = torch.autograd.grad(u_i, x, grad_outputs=torch.ones_like(
            u_i), create_graph=True)[0][:, dim:dim+1]
        laplacian_u += u_ii

    if use_stochastic:
        residual = u_t - alpha * laplacian_u + (u * w)
    else:
        residual = u_t - alpha * laplacian_u

    return torch.mean(residual**2)


def initial_condition_loss_nd(model, x, u0_func):
    """
    Compute initial condition loss for n-dimensional case
    Args:
        x: spatial coordinates tensor (batch_size, n_dims)
        u0_func: initial condition function that takes x as input
    """
    t0 = torch.zeros(x.shape[0], 1, device=x.device)
    u0_pred = model(t0, x)
    u0_true = u0_func(x)
    return torch.mean((u0_pred - u0_true)**2)


def neumann_boundary_condition_loss_nd(model, t, x, n_dims):
    """
    Compute Neumann boundary condition loss for n-dimensional case
    Args:
        t: time tensor (batch_size, 1)
        x: spatial coordinates tensor (batch_size, n_dims)
        n_dims: number of spatial dimensions
    """
    total_loss = 0
    batch_size = x.shape[0]

    for dim in range(n_dims):
        # Create boundary points for current dimension
        x_boundary = x.clone()
        x_boundary.requires_grad = True

        # Evaluate at both boundaries (0 and 1) for current dimension
        for boundary_val in [0.0, 1.0]:
            x_boundary[:, dim] = boundary_val
            u_boundary = model(t, x_boundary)

            # Compute derivative with respect to current dimension
            du_dx = torch.autograd.grad(
                u_boundary, x_boundary,
                grad_outputs=torch.ones_like(u_boundary),
                create_graph=True
            )[0][:, dim:dim+1]

            total_loss += torch.mean(du_dx**2)

    return total_loss


def compute_losses(model, batch, n_dims, u0_func):
    t, x = [b.to(DEVICE) for b in batch]
    loss_initial = initial_condition_loss_nd(model, x, u0_func)
    loss_boundary = neumann_boundary_condition_loss_nd(model, t, x, n_dims)

    return {
        "initial_loss": loss_initial,
        "boundary_loss": loss_boundary,
    }


def generate_batch_data(batch_size, n_dims, device):
    """Generate batch data for n-dimensional case"""
    t = torch.rand((batch_size, 1), device=device)
    x = torch.rand((batch_size, n_dims), device=device)
    return t, x


def train_heat_equation_nd(model, optimizer, alpha, epochs, batch_size, n_dims, u0_func, delta_t, num_samples=5):
    model = model.to(DEVICE)
    scaler = torch.amp.GradScaler()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=100, factor=0.5)
    best_loss = float("inf")
    pbar = tqdm(range(epochs), desc="Training")

    for epoch in pbar:
        model.train()

        # Generate batch data
        t, x = generate_batch_data(batch_size, n_dims, DEVICE)
        total_residual_loss = torch.tensor(0, device=DEVICE, dtype=torch.float32)

        # Generate multiple noise samples
        ws = [sheet.simulate(torch.cat([t, x], dim=1)) for _ in range(num_samples)]
        for i in range(num_samples):
            w = ws[i]
            residual_loss = heat_residual_loss_nd(model, t, x, alpha, w, use_stochastic=True)
            total_residual_loss += residual_loss

        avg_residual_loss = total_residual_loss / num_samples

        batch = (t, x)
        losses = compute_losses(model, batch, n_dims, u0_func)
        losses["Avg. Residual Loss"] = avg_residual_loss
        losses["total_loss"] = avg_residual_loss + losses["initial_loss"] + losses["boundary_loss"]

        loss = losses["total_loss"]

        # Optimization step
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
    L = 1.0
    T = 1.0
    alpha = 0.1
    delta_t = 0.01
    n_dims = 3  # Change this to the desired number of dimensions
    torch.autograd.set_detect_anomaly(True)

    # Define n-dimensional initial condition
    def u0_func(x):
        return torch.cos(torch.pi * x).prod(dim=1, keepdim=True)

    # Initialize model and optimizer
    model = DGM(
        input_dims=n_dims,  # Updated to handle n dimensions
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
        n_dims=n_dims,
        u0_func=u0_func,
        delta_t=delta_t,
    )

    # Generate visualization
    gen_heat_snapshots(
        model,
        grid_size=100,
        time_steps=[0.1, 0.25, 0.5, 0.9],
        name="DGM_Stochastic",
    )
