import torch
import numpy as np
from tqdm import tqdm
from typing import Callable


def train_model(
        model: torch.nn.Module,
        optimizer,
        loss_fn: Callable,
        data_generator: Callable,
        epochs: int,
        step_fn: Callable,
        verbose: bool = True
):
    """Generic training loop for a model with custom step logic and dynamic data variables."""
    if verbose:
        p_bar = tqdm(range(epochs), desc="Training")
    else:
        p_bar = range(epochs)

    best_val_loss = np.inf
    best_weights = model.state_dict()
    val_losses = []

    for epoch in p_bar:
        model.train()

        # Unpack any number of variables from the data generator
        data = data_generator()
        optimizer.zero_grad()

        # Pass unpacked data to the step function
        loss, val_loss = step_fn(model, loss_fn, *data)

        loss.backward()
        optimizer.step()

        val_losses.append(val_loss)
        compare_loss = torch.mean((torch.tensor(val_losses[-10:])))
        if compare_loss < best_val_loss:
            best_val_loss = compare_loss
            best_weights = model.state_dict()

        if verbose:
            p_bar.set_postfix({
                "loss": loss.item(),
                "val_loss": val_loss,
                "rolling_val_loss": compare_loss.item(),
                "best_val_loss": best_val_loss.item()
            })

    model.load_state_dict(best_weights)
    return best_val_loss, val_losses, model


def mim_heat_step(model, loss_fn, t, X, w, boundary_mask, *args):
    """Step logic for MIM heat model with flexible data inputs."""
    u, p = model(X, t)
    dgm_resdiual_error, residual,  consistency = loss_fn(u, p, t, w, X)
    loss = residual + consistency
    return loss, dgm_resdiual_error.item()


def mim_burgers_step(model, loss_fn, t, X, w, boundary_mask, *args):
    """Step logic for MIM Burgers model with flexible data inputs."""
    u, p = model(X, t, *args)
    dgm_resdiual_error, residual, consistency,  = loss_fn(u, p, t, w, X, *args)
    loss = residual + consistency
    return loss, dgm_resdiual_error.item()


def dgm_heat_step(model, loss_fn, t, X, w, boundary_mask, *args):
    """Step logic for DGM heat model with flexible data inputs."""
    t_0 = torch.zeros_like(t)
    u = model(X, t)
    u0 = model(X, t_0)
    residual, initial, boundary = loss_fn(u, u0, t, w, X, boundary_mask)
    loss = residual + initial + boundary
    return loss, residual.item()


def dgm_burgers_step(model, loss_fn, t, X, w,  boundary_mask, *args):
    """Step logic for DGM Burgers model with flexible data inputs."""
    t_0 = torch.zeros_like(t)
    u = model(X, t, *args)
    u0 = model(X, t_0, *args)
    residual, initial, boundary = loss_fn(u, u0, t, w, X, boundary_mask, *args)
    loss = residual + initial + boundary
    return loss, residual.item()



def train_mim_heat(
        model: torch.nn.Module,
        optimizer,
        loss_fn: torch.nn.Module,
        data_generator: Callable,
        epochs: int,
        verbose: bool = True
):
    return train_model(model, optimizer, loss_fn, data_generator, epochs, mim_heat_step, verbose)

def train_mim_burgers(
        model: torch.nn.Module,
        optimizer,
        loss_fn: torch.nn.Module,
        data_generator: Callable,
        epochs: int,
        verbose: bool = True
):
    return train_model(model, optimizer, loss_fn, data_generator, epochs, mim_burgers_step, verbose)

def train_dgm_heat(
        model: torch.nn.Module,
        optimizer,
        loss_fn: torch.nn.Module,
        data_generator: Callable,
        epochs: int,
        verbose: bool = True
):
    return train_model(model, optimizer, loss_fn, data_generator, epochs, dgm_heat_step, verbose)

def train_dgm_burgers(
        model: torch.nn.Module,
        optimizer,
        loss_fn: torch.nn.Module,
        data_generator: Callable,
        epochs: int,
        verbose: bool = True
):
    return train_model(model, optimizer, loss_fn, data_generator, epochs, dgm_burgers_step, verbose)