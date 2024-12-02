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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

log_dir = Path("runs") / datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir)

sheet = BrownianSheet(device=DEVICE)


def heat_residual_loss_nd(model, *coords, alpha, w, use_stochastic=False):
    for coord in coords:
        coord.requires_grad = True

    u = model(*coords)
    u_t = torch.autograd.grad(u, coords[0], grad_outputs=torch.ones_like(u), create_graph=True)[0]

    laplacian_u = 0
    for coord in coords[1:]:
        u_x = torch.autograd.grad(u, coord, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(
            u_x, coord, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        laplacian_u += u_xx

    if use_stochastic:
        residual = u_t - alpha * laplacian_u + (u * w)
    else:
        residual = u_t - alpha * laplacian_u

    return torch.mean(residual**2)


def initial_condition_loss_nd(model, coords, u0_func):
    t0 = torch.zeros_like(coords[0])
    u0_pred = model(t0, *coords)
    u0_true = u0_func(*coords)
    return torch.mean((u0_pred - u0_true)**2)


def neumann_boundary_condition_loss_nd(model, t, boundaries):
    total_loss = 0
    batch_size = t.shape[0]

    for dim, (min_val, max_val) in enumerate(boundaries):
        coords_min = [torch.rand((batch_size, 1), device=DEVICE) for _ in range(len(boundaries))]
        coords_max = [x.clone() for x in coords_min]

        coords_min[dim] = min_val * torch.ones_like(coords_min[dim])
        coords_max[dim] = max_val * torch.ones_like(coords_max[dim])

        for coords in [coords_min, coords_max]:
            coord = coords[dim]
            coord.requires_grad = True

            u = model(t, *coords)
            du_dx = torch.autograd.grad(
                u, coord, grad_outputs=torch.ones_like(u), create_graph=True)[0]
            total_loss += torch.mean(du_dx**2)

    return total_loss


def adjusted_initial_condition(*coords):
    n_dims = len(coords)
    cos_terms = [torch.cos(torch.pi * xi) for xi in coords]
    prod_cos = torch.prod(torch.stack(cos_terms), dim=0)

    # Compute scaling factor
    scaling_factor = np.sqrt(n_dims) * np.log10(np.exp(n_dims))
    return prod_cos * scaling_factor


# Use in compute_losses


def compute_losses(model, batch, boundaries):
    t, *coords = batch
    loss_initial = initial_condition_loss_nd(
        model, coords, adjusted_initial_condition
    )
    loss_boundary = neumann_boundary_condition_loss_nd(model, t, boundaries)
    return {
        "initial_loss": loss_initial,
        "boundary_loss": loss_boundary,
    }


def train_heat_equation_nd(model, optimizer, alpha, epochs, batch_size, boundaries, num_samples=5):
    model = model.to(DEVICE)
    scaler = torch.amp.GradScaler()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=100, factor=0.5)
    best_loss = float("inf")
    pbar = tqdm(range(epochs), desc="Training")

    for epoch in pbar:
        model.train()

        t = torch.rand((batch_size, 1), device=DEVICE)
        coords = [torch.rand((batch_size, 1), device=DEVICE) for _ in range(len(boundaries))]
        batch = (t, *coords)

        total_residual_loss = torch.tensor(0, device=DEVICE, dtype=torch.float32)

        points = torch.cat([t] + coords, dim=1)
        ws = [sheet.simulate(points) for _ in range(num_samples)]

        for w in ws:
            residual_loss = heat_residual_loss_nd(
                model, t, *coords, alpha=alpha, w=w, use_stochastic=False)
            total_residual_loss += residual_loss

        avg_residual_loss = total_residual_loss / num_samples
        losses = compute_losses(model, batch, boundaries)
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
    alpha = 0.1
    torch.autograd.set_detect_anomaly(True)

    n_dims = 16  # Change this for different dimensions
    boundaries = [(0, 1) for _ in range(n_dims)]  # Boundaries for each dimension

    model = DGM(
        input_dims=n_dims,
        hidden_dims=[128, 128, 64],
        dgm_dims=0,
        n_dgm_layers=3,
        hidden_activation="tanh",
        output_activation=None,
    )

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    train_heat_equation_nd(
        model,
        optimizer,
        alpha=alpha,
        epochs=1000,
        batch_size=2048,
        boundaries=boundaries,
        num_samples=5,
    )

    gen_heat_snapshots(model, grid_size=100, time_steps=[0.1, 0.25, 0.5, 0.9], name="DGM")
