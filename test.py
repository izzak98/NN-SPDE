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


def heat_residual_loss_2d(model, t, x, y, alpha, w, use_stochastic=False):
    t.requires_grad = True
    x.requires_grad = True
    y.requires_grad = True

    u = model(t, x, y)
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]

    laplacian_u = u_xx + u_yy

    if use_stochastic:
        residual = u_t - alpha * laplacian_u + (u * w)
    else:
        residual = u_t - alpha * laplacian_u

    return torch.mean(residual**2)

# Loss for the initial condition


def initial_condition_loss_2d(model, x, y, u0_func):
    t0 = torch.zeros_like(x)
    u0_pred = model(t0, x, y)
    u0_true = u0_func(x, y)
    return torch.mean((u0_pred - u0_true)**2)

# Loss for Neumann boundary conditions


def neumann_boundary_condition_loss_2d(model, t, x_min, x_max, y_min, y_max):
    # Derivative with respect to x at all boundary points
    x_min.requires_grad = True
    x_max.requires_grad = True

    # Evaluate at both y_min and y_max for x boundaries
    u_x_min_ymin = model(t, x_min, y_min)
    u_x_min_ymax = model(t, x_min, y_max)
    u_x_max_ymin = model(t, x_max, y_min)
    u_x_max_ymax = model(t, x_max, y_max)

    du_dx_min_ymin = torch.autograd.grad(
        u_x_min_ymin, x_min, grad_outputs=torch.ones_like(u_x_min_ymin), create_graph=True)[0]
    du_dx_min_ymax = torch.autograd.grad(
        u_x_min_ymax, x_min, grad_outputs=torch.ones_like(u_x_min_ymax), create_graph=True)[0]
    du_dx_max_ymin = torch.autograd.grad(
        u_x_max_ymin, x_max, grad_outputs=torch.ones_like(u_x_max_ymin), create_graph=True)[0]
    du_dx_max_ymax = torch.autograd.grad(
        u_x_max_ymax, x_max, grad_outputs=torch.ones_like(u_x_max_ymax), create_graph=True)[0]

    # Derivative with respect to y at all boundary points
    y_min.requires_grad = True
    y_max.requires_grad = True

    # Evaluate at both x_min and x_max for y boundaries
    u_y_min_xmin = model(t, x_min, y_min)
    u_y_min_xmax = model(t, x_max, y_min)
    u_y_max_xmin = model(t, x_min, y_max)
    u_y_max_xmax = model(t, x_max, y_max)

    du_dy_min_xmin = torch.autograd.grad(
        u_y_min_xmin, y_min, grad_outputs=torch.ones_like(u_y_min_xmin), create_graph=True)[0]
    du_dy_min_xmax = torch.autograd.grad(
        u_y_min_xmax, y_min, grad_outputs=torch.ones_like(u_y_min_xmax), create_graph=True)[0]
    du_dy_max_xmin = torch.autograd.grad(
        u_y_max_xmin, y_max, grad_outputs=torch.ones_like(u_y_max_xmin), create_graph=True)[0]
    du_dy_max_xmax = torch.autograd.grad(
        u_y_max_xmax, y_max, grad_outputs=torch.ones_like(u_y_max_xmax), create_graph=True)[0]

    return (
        torch.mean(du_dx_min_ymin**2) + torch.mean(du_dx_min_ymax**2) +
        torch.mean(du_dx_max_ymin**2) + torch.mean(du_dx_max_ymax**2) +
        torch.mean(du_dy_min_xmin**2) + torch.mean(du_dy_min_xmax**2) +
        torch.mean(du_dy_max_xmin**2) + torch.mean(du_dy_max_xmax**2)
    )


def compute_losses(model, batch):
    t, x, y, x_min, x_max, y_min, y_max = [b.to(DEVICE) for b in batch]
    loss_initial = initial_condition_loss_2d(
        model, x, y, lambda x, y: torch.cos(torch.pi * x) * torch.cos(torch.pi * y)
    )
    loss_boundary = neumann_boundary_condition_loss_2d(
        model, t, x_min, x_max, y_min, y_max
    )

    return {
        "initial_loss": loss_initial,
        "boundary_loss": loss_boundary,
    }


def train_heat_equation_2d_with_neumann(model, optimizer, alpha, epochs, batch_size, delta_t, num_samples=5):
    model = model.to(DEVICE)
    scaler = torch.amp.GradScaler()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=100, factor=0.5)
    best_loss = float("inf")
    pbar = tqdm(range(epochs), desc="Training")

    for epoch in pbar:
        model.train()

        # Generate batch data
        t = torch.rand((batch_size, 1), device=DEVICE)
        x = torch.rand((batch_size, 1), device=DEVICE)
        y = torch.rand((batch_size, 1), device=DEVICE)
        x_min = torch.zeros((batch_size, 1), device=DEVICE)
        x_max = torch.ones((batch_size, 1), device=DEVICE)
        y_min = torch.zeros((batch_size, 1), device=DEVICE)
        y_max = torch.ones((batch_size, 1), device=DEVICE)

        total_residual_loss = torch.tensor(0, device=DEVICE, dtype=torch.float32)

        ws = [sheet.simulate(torch.cat([t, x, y], dim=1)) for _ in range(num_samples)]
        for i in range(num_samples):
            # Simulate Brownian sheet increments
            w = ws[i]

            residual_loss = heat_residual_loss_2d(
                model, t, x, y, alpha, w, use_stochastic=True)
            total_residual_loss += residual_loss

        avg_residual_loss = total_residual_loss / num_samples

        batch = (t, x, y, x_min, x_max, y_min, y_max)
        losses = compute_losses(model, batch)
        losses["Avg. Residual Loss"] = avg_residual_loss
        losses["total_loss"] = avg_residual_loss + losses["initial_loss"] + losses["boundary_loss"]

        # Combine the average residual loss with other losses
        loss = avg_residual_loss + losses["initial_loss"] + losses["boundary_loss"]

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
    torch.autograd.set_detect_anomaly(True)

    # Initialize model and optimizer
    model = DGM(
        input_dims=2,
        hidden_dims=[128, 128, 64],
        dgm_dims=0,
        n_dgm_layers=3,
        hidden_activation="tanh",
        output_activation=None,
    )

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # Train the model
    train_heat_equation_2d_with_neumann(
        model,
        optimizer,
        alpha=alpha,
        epochs=1000,
        batch_size=2048,
        delta_t=delta_t,
    )

    # Generate visualization
    gen_heat_snapshots(
        model,
        grid_size=100,
        time_steps=[0.1, 0.25, 0.5, 0.9],
        name="DGM_Stochastic",
    )
