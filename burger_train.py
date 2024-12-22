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


def initial_condition_loss_nd(model, nu, alpha, coords, u0_func):
    t0 = torch.zeros_like(coords[0])
    u0_pred = model(t0, nu, alpha, *coords)
    u0_true = u0_func(*coords)
    return torch.mean((u0_pred - u0_true)**2)


def adjusted_initial_condition(*coords):
    n_dims = len(coords)
    sin_terms = [torch.sin(torch.pi * xi) for xi in coords]
    prod_sin = torch.prod(torch.stack(sin_terms), dim=0)

    # Compute scaling factor
    scaling_factor = np.sqrt(n_dims) * np.log10(np.exp(n_dims))
    return prod_sin * scaling_factor


class TrainDGM():
    def __init__(self, lambda1=1, lambda2=1, use_stochastic=False):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.use_stochastic = use_stochastic
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        assert self.lambda1 >= 1 and self.lambda2 >= 1, "Lambda values must be greater than or equal to 1"

    def burger_residual_loss_nd(self, model, t, nu, alpha, *coords, w):
        for coord in coords:
            coord.requires_grad = True
        t.requires_grad = True

        u = model(t, nu, alpha, *coords)
        u_t = torch.autograd.grad(
            u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]

        # Separate accumulation for Laplacian and convection terms
        laplacian_u = 0
        convection_term = 0
        for coord in coords:
            u_x = torch.autograd.grad(
                u, coord, grad_outputs=torch.ones_like(u), create_graph=True)[0]
            u_xx = torch.autograd.grad(
                u_x, coord, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
            # For each dimension, multiply u by its gradient and add to convection
            convection_term += u * u_x
            laplacian_u += u_xx

        if self.use_stochastic:
            residual = u_t + convection_term - nu * laplacian_u - \
                (alpha*torch.pow(u, len(coords)) * w)
        else:
            residual = u_t + convection_term - nu * laplacian_u

        return torch.mean(residual**2)

    def periodic_boundary_condition_loss_nd(self, model, t, nu, alpha, boundaries, *coords):
        total_loss = 0
        batch_size = t.shape[0]

        for dim, (min_val, max_val) in enumerate(boundaries):
            min_coords = list(coords)
            min_coords[dim] = torch.tensor(
                [[min_val]] * batch_size, device=self.device)
            max_coords = list(coords)
            max_coords[dim] = torch.tensor(
                [[max_val]] * batch_size, device=self.device)
            u_min = model(t, nu, alpha, *min_coords)
            u_max = model(t, nu, alpha, *max_coords)
            total_loss += torch.mean((u_min - u_max)**2)

        return total_loss

    def compute_losses(self, model, batch, boundaries):
        t, nu, alpha, *coords = batch
        loss_initial = initial_condition_loss_nd(
            model, nu, alpha, coords, adjusted_initial_condition
        )
        loss_boundary = self.periodic_boundary_condition_loss_nd(
            model, t, nu, alpha, boundaries, *coords)
        return {
            "initial_loss": loss_initial,
            "boundary_loss": loss_boundary,
        }

    def forward(self, model, t, nu, alpha, coords, ws, boundaries):
        total_residual_loss = torch.tensor(0, device=self.device, dtype=torch.float32)
        batch = (t, nu, alpha, *coords)
        for w in ws:
            residual_loss = self.burger_residual_loss_nd(model, t, nu, alpha, *coords, w=w)
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
    def __init__(self, lambda1=1, use_stochastic=False):
        self.lambda1 = lambda1  # Consider increasing this significantly
        self.use_stochastic = use_stochastic
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert self.lambda1 >= 1, "Lambda value must be greater than or equal to 1"

    def burger_residual_loss_nd(self, model, t, nu, alpha, *coords, w):
        for coord in coords:
            coord.requires_grad = True
        t.requires_grad = True

        u, p = model(t, nu, alpha, *coords)

        # Compute ut through autograd
        u_t = torch.autograd.grad(
            u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]

        # Use learned p for Laplacian
        laplacian_u = torch.zeros_like(u)
        convection_term = torch.zeros_like(u)
        for i, coord in enumerate(coords):
            # Get divergence of p
            p_i = p[:, [i]]
            p_i_grad = torch.autograd.grad(
                p_i, coord, grad_outputs=torch.ones_like(p_i), create_graph=True)[0]
            laplacian_u += p_i_grad
            convection_term += u * p_i

        if self.use_stochastic:
            residual = u_t + convection_term - nu * laplacian_u - \
                (alpha*torch.pow(u, len(coords)) * w)
        else:
            residual = u_t + convection_term - nu * laplacian_u

        return torch.mean(residual**2)

    def gradient_loss(self, model, t, nu, alpha, coords):
        # Initialize total loss
        total_diff = 0

        # Ensure gradients enabled
        for coord in coords:
            coord.requires_grad = True

        # Forward pass to compute u and p
        u, p = model(t, nu, alpha, *coords)

        # For each dimension
        for i, coord in enumerate(coords):
            # Compute true gradient through autograd
            u_grad = torch.autograd.grad(
                u, coord, grad_outputs=torch.ones_like(u), create_graph=True)[0]

            # Compare with learned gradient (p)
            diff = (p[:, [i]] - u_grad) ** 2
            total_diff += torch.mean(diff)

        return total_diff

    def forward(self, model, t, nu, alpha, coords, ws, boundaries):
        total_residual_loss = torch.tensor(0, device=self.device, dtype=torch.float32)
        for w in ws:
            residual_loss = self.burger_residual_loss_nd(model, t, nu, alpha, *coords, w=w)
            total_residual_loss += residual_loss

        avg_residual_loss = total_residual_loss / len(ws)
        gradient_loss = self.gradient_loss(model, t, nu, alpha, coords)

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


def train_burger(model, optimizer, epochs, batch_size, boundaries, loss_calculator, num_samples=5):
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
        # sample nu with log-uniform distribution
        log_a, log_b = torch.log(torch.tensor(1e-10)), torch.log(torch.tensor(1.0))
        nu = torch.exp(torch.empty(batch_size, 1).uniform_(log_a, log_b)).to(DEVICE)
        # sample alpha with log-uniform distribution
        alpha = torch.rand((batch_size, 1), device=DEVICE)
        coords = [torch.rand((batch_size, 1), device=DEVICE) for _ in range(len(boundaries))]
        points = torch.cat([t] + coords, dim=1)
        ws = [sheet.simulate(points) for _ in range(num_samples)]
        losses = loss_calculator.forward(model, t, nu, alpha, coords, ws, boundaries)
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
