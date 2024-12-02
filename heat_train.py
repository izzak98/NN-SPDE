import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils.white_noise import BrownianSheet
from pathlib import Path
from datetime import datetime

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


def initial_condition_loss_nd(model, coords, u0_func):
    t0 = torch.zeros_like(coords[0])
    u0_pred = model(t0, *coords)
    u0_true = u0_func(*coords)
    return torch.mean((u0_pred - u0_true)**2)


def adjusted_initial_condition(*coords):
    n_dims = len(coords)
    cos_terms = [torch.cos(torch.pi * xi) for xi in coords]
    prod_cos = torch.prod(torch.stack(cos_terms), dim=0)

    # Compute scaling factor
    scaling_factor = np.sqrt(n_dims) * np.log10(np.exp(n_dims))
    return prod_cos * scaling_factor


class TrainDGM():
    def __init__(self, lambda1=1, lambda2=1, use_stochastic=False, alpha=0.1):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.use_stochastic = use_stochastic
        self.alpha = alpha
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        assert self.lambda1 >= 1 and self.lambda2 >= 1, "Lambda values must be greater than or equal to 1"

    def heat_residual_loss_nd(self, model, *coords, w):
        for coord in coords:
            coord.requires_grad = True

        u = model(*coords)
        u_t = torch.autograd.grad(
            u, coords[0], grad_outputs=torch.ones_like(u), create_graph=True)[0]

        laplacian_u = 0
        for coord in coords[1:]:
            u_x = torch.autograd.grad(
                u, coord, grad_outputs=torch.ones_like(u), create_graph=True)[0]
            u_xx = torch.autograd.grad(
                u_x, coord, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
            laplacian_u += u_xx

        if self.use_stochastic:
            residual = u_t - self.alpha * laplacian_u + (u * w)
        else:
            residual = u_t - self.alpha * laplacian_u

        return torch.mean(residual**2)

    def neumann_boundary_condition_loss_nd(self, model, t, boundaries):
        total_loss = 0
        batch_size = t.shape[0]

        for dim, (min_val, max_val) in enumerate(boundaries):
            coords_min = [torch.rand((batch_size, 1), device=self.device)
                          for _ in range(len(boundaries))]
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

    def compute_losses(self, model, batch, boundaries):
        t, *coords = batch
        loss_initial = initial_condition_loss_nd(
            model, coords, adjusted_initial_condition
        )
        loss_boundary = self.neumann_boundary_condition_loss_nd(model, t, boundaries)
        return {
            "initial_loss": loss_initial,
            "boundary_loss": loss_boundary,
        }

    def forward(self, model, t, coords, ws, boundaries):
        total_residual_loss = torch.tensor(0, device=self.device, dtype=torch.float32)
        batch = (t, *coords)
        for w in ws:
            residual_loss = self.heat_residual_loss_nd(model, t, *coords, w=w)
            total_residual_loss += residual_loss

        avg_residual_loss = total_residual_loss / len(ws)
        losses = self.compute_losses(model, batch, boundaries)
        losses["Avg. Residual Loss"] = avg_residual_loss
        total_loss = avg_residual_loss + self.lambda1 * \
            losses["initial_loss"] + self.lambda2 * losses["boundary_loss"]
        losses["total_loss"] = total_loss
        with torch.no_grad():
            losses["unadjusted_total_loss"] = avg_residual_loss + \
                losses["initial_loss"] + losses["boundary_loss"]
        return losses


class TrainMIM():
    def __init__(self, lambda1=1, use_stochastic=False, alpha=0.1):
        self.lambda1 = lambda1
        self.use_stochastic = use_stochastic
        self.alpha = alpha
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert self.lambda1 >= 1, "Lambda value must be greater than or equal to 1"

    def heat_residual_loss_nd(self, model, *coords, w):
        for coord in coords:
            coord.requires_grad = True

        u, p = model(*coords)
        u_t = torch.autograd.grad(
            u, coords[0], grad_outputs=torch.ones_like(u), create_graph=True)[0]

        laplacian_u = 0
        for coord in coords[1:]:
            p_x = torch.autograd.grad(
                p, coord, grad_outputs=torch.ones_like(p), create_graph=True)[0]

            laplacian_u += p_x

        if self.use_stochastic:
            residual = u_t - self.alpha * laplacian_u + (u * w)
        else:
            residual = u_t - self.alpha * laplacian_u

        return torch.mean(residual**2)

    def gradient_loss(self, model, t, coords):
        # Forward pass to compute u and p
        u, p = model(t, *coords)

        # Ensure gradients can be computed with respect to coordinates
        for coord in coords:
            coord.requires_grad = True

        # Compute the difference between p and ∇u
        total_diff = 0
        for i, coord in enumerate(coords):
            # Compute gradient ∂u/∂coord
            u_grad = torch.autograd.grad(
                u, coord, grad_outputs=torch.ones_like(u), create_graph=True)[0]

            # Compute squared difference between p and ∇u for this dimension
            total_diff += torch.mean((p[:, i] - u_grad) ** 2)

        return total_diff

    def forward(self, model, t, coords, ws, boundaries):
        total_residual_loss = torch.tensor(0, device=self.device, dtype=torch.float32)
        for w in ws:
            residual_loss = self.heat_residual_loss_nd(model, t, *coords, w=w)
            total_residual_loss += residual_loss

        avg_residual_loss = total_residual_loss / len(ws)
        gradient_loss = self.gradient_loss(model, t, coords)
        total_loss = avg_residual_loss + self.lambda1 * gradient_loss
        losses = {
            "Avg. Residual Loss": avg_residual_loss,
            "Gradient Loss": gradient_loss,
            "total_loss": total_loss,
        }
        with torch.no_grad():
            losses["unadjusted_total_loss"] = avg_residual_loss + gradient_loss
        return losses


def train_heat(model, optimizer, epochs, batch_size, boundaries, loss_calculator, num_samples=5):
    run_name = f"{model.name}-{model.input_dims-1}D-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log_dir = Path("runs") / run_name
    writer = SummaryWriter(log_dir)

    sheet = BrownianSheet(device=DEVICE)

    model = model.to(DEVICE)
    scaler = torch.amp.GradScaler()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=100, factor=0.5)
    best_loss = float("inf")
    pbar = tqdm(range(epochs), desc="Training")

    for epoch in pbar:
        model.train()

        t = torch.rand((batch_size, 1), device=DEVICE)
        coords = [torch.rand((batch_size, 1), device=DEVICE) for _ in range(len(boundaries))]
        points = torch.cat([t] + coords, dim=1)
        ws = [sheet.simulate(points) for _ in range(num_samples)]
        losses = loss_calculator.forward(model, t, coords, ws, boundaries)
        loss = losses["total_loss"]
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        scheduler.step(loss)
        pbar.set_postfix({key: f"{value.item():.2e}" for key, value in losses.items()})
        for key, value in losses.items():
            writer.add_scalar(f"Loss/{key}", value.item(), epoch)

        if loss < best_loss:
            best_loss = losses["unadjusted_total_loss"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            }, log_dir / "best_model.pt")

    writer.close()
