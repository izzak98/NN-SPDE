import torch
import numpy as np
from tqdm import tqdm
from typing import Callable

def check_solution_quality(model, data):
    """Helper function to check solution quality"""
    with torch.no_grad():
        X, t = data[1], data[0]  # Assuming data format
        u = model(X, t)
        if isinstance(u, tuple):
            u = u[0]
        stats = {
            'mean': torch.mean(u).item(),
            'std': torch.std(u).item(),
            'max': torch.max(u).item(),
            'min': torch.min(u).item(),
            'abs_mean': torch.mean(torch.abs(u)).item()
        }
        
        # Define quality checks
        is_trivial = (
            stats['abs_mean'] < 1e-5 or  # Too small
            stats['std'] < 1e-5 or       # Almost constant
            stats['max'] - stats['min'] < 1e-5  # Almost constant
        )
        
        return is_trivial, stats

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
        
        # Check solution quality
        if epoch % 10 == 0:
            is_trivial, stats = check_solution_quality(model, data)
        
        # if is_trivial:
        #     if verbose:
        #         print(f"\nPruning due to trivial solution at epoch {epoch}")
        #         print(f"Solution stats: {stats}")
        #     return np.inf, [], model
        
        if val_loss == 0:
            if verbose:
                print("\nPruning due to zero loss")
            return np.inf, [], model
        
        loss.backward()
        optimizer.step()
        
        val_losses.append(val_loss)
        compare_loss = torch.mean((torch.tensor(val_losses[-10:]))).item() if len(val_losses) > 10 else np.inf
        
        if compare_loss < best_val_loss:
            best_val_loss = compare_loss
            best_weights = model.state_dict()
            
        if verbose:
            # Add solution stats to progress bar
            postfix = {
                "loss": f"{loss.item():.2e}",
                "val_loss": f"{val_loss:.2e}",
                "rolling_val_loss": f"{compare_loss:.2e}",
                "best_val_loss": f"{best_val_loss:.2e}",
                "mean": f"{stats['mean']:.2e}",
                "std": f"{stats['std']:.2e}"
            }
            p_bar.set_postfix(postfix)
        
    
    model.load_state_dict(best_weights)
    return best_val_loss, val_losses, model


def mim_heat_step(model, loss_fn, t, X, w, boundary_mask, *args):
    """Step logic for MIM heat model with flexible data inputs."""
    u, p = model(X, t)
    loss_calculations = loss_fn(u, p, t, w, X)
    dgm_residual_error = loss_calculations[0]
    loss = sum(loss_calculations[1:])
    return loss, dgm_residual_error.item()


def mim_burgers_step(model, loss_fn, t, X, w, boundary_mask, *args):
    """Step logic for MIM Burgers model with flexible data inputs."""
    u, p = model(X, t, *args)
    dgm_residual_error, residual, consistency,  = loss_fn(u, p, t, w, X, *args)
    loss = residual + consistency
    return loss, dgm_residual_error.item()


def dgm_heat_step(model, loss_fn, t, X, w, boundary_mask, *args):
    """Step logic for DGM heat model with flexible data inputs."""
    t_0 = torch.zeros_like(t)
    u = model(X, t)
    u0 = model(X, t_0)
    loss_calculations = loss_fn(u, u0, t, w, X, boundary_mask)
    residual = loss_calculations[0]
    loss = sum(loss_calculations)
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