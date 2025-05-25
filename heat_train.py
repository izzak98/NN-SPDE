from typing import Callable
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils.noise_utils import shifted_gaussian_white_noise
from pathlib import Path
from datetime import datetime
from accelerate import Accelerator

# Equation to try
# d_t u = nu * Laplacian(u) + sin(pi x) * E[\xi]
# ic = sin(pi x)
# bc = 0


class HeatTrainLEC:
    def __init__(self, model,
                 inital_condition: Callable,
                 boundary_condition: Callable,
                 forcing_term: Callable,
                 boundary_conversion_func: Callable,
                 lambda1: float = 1.0,
                 lambda2: float = 1.0,
                 stochastic: bool = True):
        self.initial_condition = inital_condition
        self.boundary_condition = boundary_condition
        self.forcing_term = forcing_term
        self.boundary_conversion_func = boundary_conversion_func
        self.model = model
        self.name = "HeatTrainLEC"
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.stochastic = stochastic

    def forward_pass(self, t, nu, noise_args, coords):
        # Concatenate all coordinates
        return self.model(t, nu, *noise_args, *coords)

    def heat_residual_loss_nd(self, t, nu, noise_args, coords, w):
        for coord in coords:
            coord.requires_grad = True
        t.requires_grad = True

        u = self.forward_pass(t, nu, noise_args, coords)
        u_t = torch.autograd.grad(
            u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]

        laplacian_u = 0
        for coord in coords:
            u_x = torch.autograd.grad(
                u, coord, grad_outputs=torch.ones_like(u), create_graph=True)[0]
            u_xx = torch.autograd.grad(
                u_x, coord, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
            laplacian_u += u_xx

        if self.stochastic:
            residual = u_t - nu * laplacian_u - self.forcing_term(coords) * w
        else:
            residual = u_t - nu * laplacian_u - self.forcing_term(coords)

        return torch.mean(residual**2)

    def inital_loss(self, nu, noise_args, coords):
        # Concatenate all coordinates
        t = torch.zeros_like(coords[0])
        u = self.forward_pass(t, nu, noise_args, coords)
        initial_condition = self.initial_condition(coords)
        return torch.mean((u - initial_condition)**2)

    def boundary_loss(self, t, nu, noise_args, coords):
        coords = [self.boundary_conversion_func(coord) for coord in coords]
        u = self.forward_pass(t, nu, noise_args, coords)
        boundary_condition = self.boundary_condition(coords)
        return torch.mean((u - boundary_condition)**2)

    def forward(self, t, nu, noise_args, coords, ws):
        # Calculate the residual loss
        total_residual_loss = 0
        for w in ws:
            # Calculate the residual loss for each sample
            total_residual_loss += self.heat_residual_loss_nd(t, nu, noise_args, coords, w=w)
        # Average the residual loss over the samples
        residual_loss = total_residual_loss / len(ws)
        # Calculate the initial condition loss
        initial_condition_loss = self.inital_loss(nu, noise_args, coords)

        # Calculate the boundary condition loss
        boundary_condition_loss = self.boundary_loss(t, nu, noise_args, coords)

        # Calculate the total loss
        total_loss = (residual_loss +
                      self.lambda1 * initial_condition_loss +
                      self.lambda2 * boundary_condition_loss)

        # unadjusted_total_loss
        unadjusted_total_loss = (residual_loss +
                                 initial_condition_loss +
                                 boundary_condition_loss)

        return {
            "total_loss": total_loss,
            "unadjusted_total_loss": unadjusted_total_loss,
            "residual_loss": residual_loss,
            "initial_condition_loss": initial_condition_loss,
            "boundary_condition_loss": boundary_condition_loss
        }


def train_heat(model,
               optimizer,
               analytic_solution: Callable,
               noise_func: Callable,
               loss_helper,
               n_dims: int,
               n_train_points: int,
               verbose: bool,
               name: str,
               domain: tuple = (0, 1),
               noise_dims: list[list[int]] = [[0, 1], [0, 1]],
               m_samples=5,
               ):
    accelerator = Accelerator()
    device = accelerator.device
    model, optimizer = accelerator.prepare(model, optimizer)
    name = name + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path("logs") / name
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    if verbose:
        p_bar = tqdm(range(n_train_points), desc="Training", unit="batch")
    else:
        p_bar = range(n_train_points)

    # sample validation points
    val_t = torch.rand(n_train_points, 1).to(device)
    val_coords = [torch.rand(n_train_points, 1).to(device) * (domain[1] -
                                                              domain[0]) + domain[0] for _ in range(n_dims)]
    val_noise_args = [torch.rand(n_train_points, 1).to(device) for _ in range(len(noise_dims))]
    val_nu = torch.rand(n_train_points, 1).to(device)
    val_solution = analytic_solution(val_t, val_nu, val_noise_args, val_coords)
    for e in p_bar:
        model.train()
        # sample training points
        t = torch.rand(n_train_points, 1).to(device)
        coords = [torch.rand(n_train_points, 1).to(device) * (domain[1] -
                                                              domain[0]) + domain[0] for _ in range(n_dims)]
        noise_args = [torch.rand(n_train_points, 1).to(device) for _ in range(len(noise_dims))]
        nu = torch.rand(n_train_points, 1).to(device)
        ws = [noise_func(t, *noise_args) for _ in range(m_samples)]

        losses = loss_helper.forward(t, nu, noise_args, coords, ws)
        total_loss = losses["total_loss"]
        optimizer.zero_grad()
        accelerator.backward(total_loss)
        optimizer.step()

        with torch.no_grad():
            val_guess = model(val_t, val_nu, *val_noise_args, *val_coords)
            val_error = (torch.norm(val_guess - val_solution) / torch.norm(val_solution)).item()
            # clear gradients of the model
            model.zero_grad()
        out_total_loss = losses["total_loss"].item()
        out_residual_loss = losses["residual_loss"].item()
        out_initial_condition_loss = losses["initial_condition_loss"].item()
        out_initial_condition_loss = losses["boundary_condition_loss"].item()

        writer.add_scalar(
            "Loss/total_loss", out_total_loss, e)
        writer.add_scalar(
            "Loss/residual_loss", out_residual_loss, e)
        writer.add_scalar(
            "Loss/initial_condition_loss", out_initial_condition_loss, e)
        writer.add_scalar(
            "Loss/boundary_condition_loss", out_initial_condition_loss, e)
        writer.add_scalar(
            "Loss/l2_error", val_error, e)

        if verbose and isinstance(p_bar, tqdm):
            p_bar.set_postfix(
                loss=round(out_total_loss, 4),
                # residual_loss=round(out_residual_loss, 4),
                # initial_condition_loss=round(out_initial_condition_loss, 4),
                # boundary_condition_loss=round(out_initial_condition_loss, 4),
                l2_error=round(val_error, 4),
            )
    return model
