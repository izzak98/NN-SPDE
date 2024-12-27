import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils.white_noise import BrownianSheet
from pathlib import Path
from datetime import datetime
from accelerate import Accelerator


def initial_condition_loss_nd(model, nu, coords, u0_func):
    t0 = torch.zeros_like(coords[0])
    u0_pred = model(t0, nu, *coords)
    u0_true = u0_func(*coords)
    return torch.mean((u0_pred - u0_true)**2)


def heat_initial_condition(*coords):
    n_dims = len(coords)
    cos_terms = [torch.cos(torch.pi * xi) for xi in coords]
    prod_cos = torch.prod(torch.stack(cos_terms), dim=0)

    # Compute scaling factor
    scaling_factor = np.sqrt(n_dims) * np.log10(np.exp(n_dims))
    return prod_cos * scaling_factor


class HeatTrainDGM():
    def __init__(self, lambda1=1, lambda2=1, use_stochastic=False):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.use_stochastic = use_stochastic
        assert self.lambda1 >= 1 and self.lambda2 >= 1, "Lambda values must be greater than or equal to 1"

    def heat_residual_loss_nd(self, model, t, nu, *coords, w):
        for coord in coords:
            coord.requires_grad = True
        t.requires_grad = True

        u = model(t, nu, *coords)
        u_t = torch.autograd.grad(
            u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]

        laplacian_u = 0
        for coord in coords:
            u_x = torch.autograd.grad(
                u, coord, grad_outputs=torch.ones_like(u), create_graph=True)[0]
            u_xx = torch.autograd.grad(
                u_x, coord, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
            laplacian_u += u_xx

        if self.use_stochastic:
            residual = u_t - nu * laplacian_u + (u * w)
        else:
            residual = u_t - nu * laplacian_u

        return torch.mean(residual**2)

    def neumann_boundary_condition_loss_nd(self, model, t, nu, boundaries):
        total_loss = 0
        batch_size = t.shape[0]

        for dim, (min_val, max_val) in enumerate(boundaries):
            coords_min = [torch.rand((batch_size, 1)) for _ in range(len(boundaries))]
            coords_max = [x.clone() for x in coords_min]

            coords_min[dim] = min_val * torch.ones_like(coords_min[dim])
            coords_max[dim] = max_val * torch.ones_like(coords_max[dim])

            for coords in [coords_min, coords_max]:
                coord = coords[dim]
                coord.requires_grad = True

                u = model(t, nu, *coords)
                du_dx = torch.autograd.grad(
                    u, coord, grad_outputs=torch.ones_like(u), create_graph=True)[0]
                total_loss += torch.mean(du_dx**2)

        return total_loss

    def compute_losses(self, model, batch, boundaries):
        t, nu, *coords = batch
        loss_initial = initial_condition_loss_nd(
            model, nu, coords, heat_initial_condition
        )
        loss_boundary = self.neumann_boundary_condition_loss_nd(model, t, nu, boundaries)
        return {
            "initial_loss": loss_initial,
            "boundary_loss": loss_boundary,
        }

    def forward(self, model, t, nu, coords, ws, boundaries):
        total_residual_loss = torch.tensor(0., device=t.device, dtype=torch.float32)
        batch = (t, nu, *coords)
        for w in ws:
            residual_loss = self.heat_residual_loss_nd(model, t, nu, *coords, w=w)
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


class HeatTrainMIM():
    def __init__(self, lambda1=1, use_stochastic=False):
        self.lambda1 = lambda1  # Consider increasing this significantly
        self.use_stochastic = use_stochastic
        assert self.lambda1 >= 1, "Lambda value must be greater than or equal to 1"

    def heat_residual_loss_nd(self, model, t, nu, *coords, w):
        for coord in coords:
            coord.requires_grad = True
        t.requires_grad = True

        u, p = model(t, nu, *coords)

        # Compute ut through autograd
        u_t = torch.autograd.grad(
            u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]

        # Use learned p for Laplacian
        laplacian_u = torch.zeros_like(u)
        for i, coord in enumerate(coords):
            # Get divergence of p
            p_i = p[:, [i]]
            p_i_grad = torch.autograd.grad(
                p_i, coord, grad_outputs=torch.ones_like(p_i), create_graph=True)[0]
            laplacian_u += p_i_grad

        if self.use_stochastic:
            residual = u_t - nu * laplacian_u + (u * w)
        else:
            residual = u_t - nu * laplacian_u

        return torch.mean(residual**2)

    def gradient_loss(self, model, t, nu, coords):
        # Initialize total loss
        total_diff = 0

        # Ensure gradients enabled
        for coord in coords:
            coord.requires_grad = True

        # Forward pass to compute u and p
        u, p = model(t, nu, *coords)

        # For each dimension
        for i, coord in enumerate(coords):
            # Compute true gradient through autograd
            u_grad = torch.autograd.grad(
                u, coord, grad_outputs=torch.ones_like(u), create_graph=True)[0]

            # Compare with learned gradient (p)
            diff = (p[:, [i]] - u_grad) ** 2
            total_diff += torch.mean(diff)

        return total_diff

    def forward(self, model, t, nu, coords, ws, boundaries):
        total_residual_loss = torch.tensor(0., device=t.device, dtype=torch.float32)
        for w in ws:
            residual_loss = self.heat_residual_loss_nd(model, t, nu, *coords, w=w)
            total_residual_loss += residual_loss

        avg_residual_loss = total_residual_loss / len(ws)
        gradient_loss = self.gradient_loss(model, t, nu, coords)

        # The key change: significantly increase lambda1 to enforce gradient matching
        total_loss = avg_residual_loss + self.lambda1 * gradient_loss

        losses = {
            "Avg. Residual Loss": avg_residual_loss,
            "Gradient Loss": gradient_loss,
            "total_loss": total_loss,
        }

        with torch.no_grad():
            losses["unadjusted_total_loss"] = avg_residual_loss + gradient_loss

        return losses


def train_heat(model,
               optimizer,
               epochs,
               batch_size,
               boundaries,
               loss_calculator,
               scheduler=None,
               num_samples=5,
               trial_n=""):

    # Initialize accelerator
    accelerator = Accelerator(mixed_precision='fp16')

    if trial_n != "":
        trial_n = f"{trial_n}-"

    run_name = f"{trial_n}{model.name}-{model.spatial_dims}D-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log_dir = Path("runs") / run_name
    writer = SummaryWriter(log_dir)

    # Initialize BrownianSheet with accelerator's device
    sheet = BrownianSheet(device=accelerator.device)

    # Prepare model and optimizer
    model, optimizer = accelerator.prepare(model, optimizer)
    if scheduler is not None:
        scheduler = accelerator.prepare(scheduler)

    best_loss = float("inf")
    pbar = tqdm(range(epochs), desc="Training")

    for epoch in pbar:
        model.train()

        # Create all tensors on CPU first
        t = torch.rand((batch_size, 1))
        log_a, log_b = torch.log(torch.tensor(1e-10)), torch.log(torch.tensor(1.0))
        nu = torch.exp(torch.empty(batch_size, 1).uniform_(log_a, log_b))
        coords = [torch.rand((batch_size, 1)) for _ in range(len(boundaries))]

        # Move all tensors to the same device before concatenation
        t = t.to(accelerator.device)
        nu = nu.to(accelerator.device)
        coords = [c.to(accelerator.device) for c in coords]

        # Now all tensors are on the same device for concatenation
        points = torch.cat([t] + coords, dim=1)
        ws = [sheet.simulate(points) for _ in range(num_samples)]

        # Calculate losses
        losses = loss_calculator.forward(model, t, nu, coords, ws, boundaries)
        loss = losses["total_loss"]

        # Replace manual scaling with accelerator
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()

        if scheduler is not None:
            scheduler.step(loss)

        # Update progress bar and tensorboard
        pbar.set_postfix({key: f"{value.item():.2e}" for key, value in losses.items()})
        for key, value in losses.items():
            writer.add_scalar(f"Loss/{key}", value.item(), epoch)

        if losses["unadjusted_total_loss"] < best_loss:
            best_loss = losses["unadjusted_total_loss"]
            # Save using accelerator
            accelerator.save({
                "epoch": epoch,
                "model_state_dict": accelerator.unwrap_model(model).state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            }, log_dir / "best_model.pt")

        if any([l.item() < 1e-16 for l in losses.values()]):
            return float("inf")

    writer.close()
    return best_loss
