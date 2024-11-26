import torch
from torch.optim import Optimizer
import numpy as np
from scipy.stats import entropy
from tqdm import tqdm
from typing import Callable, Tuple, NamedTuple, Optional


class BatchData(NamedTuple):
    x: torch.Tensor
    t: torch.Tensor
    w: torch.Tensor = None
    nu: torch.Tensor = None
    alpha: torch.Tensor = None


def get_batch_slice(start_idx: int, batch_size: int, proportion: float) -> slice:
    """Calculate batch indices based on proportion and batch size."""
    start = int(start_idx * proportion * batch_size)
    end = int((start_idx + 1) * proportion * batch_size)
    return slice(start, end)


def extract_batch(data: Tuple[torch.Tensor, ...], batch_slice: slice) -> BatchData:
    """
    Extract a batch of data from the full dataset, creating independent copies.
    """
    if len(data) == 5:  # Full data includes nu and alpha
        return BatchData(
            data[1][batch_slice].clone().detach().requires_grad_(True),
            data[0][batch_slice].clone().detach().requires_grad_(True),
            data[2][batch_slice].clone().detach().requires_grad_(True),
            data[3][batch_slice].clone().detach().requires_grad_(True),
            data[4][batch_slice].clone().detach().requires_grad_(True)
        )
    if len(data) == 3:  # Base data includes white noise
        return BatchData(
            data[1][batch_slice].clone().detach().requires_grad_(True),
            data[0][batch_slice].clone().detach().requires_grad_(True),
            data[2][batch_slice].clone().detach()  # No grad needed for noise
        )
    return BatchData(
        data[1][batch_slice].clone().detach().requires_grad_(True),
        data[0][batch_slice].clone().detach().requires_grad_(True)
    )


def process_batch(model, batch_data: BatchData) -> torch.Tensor:
    """Process a batch through the model."""
    return model(batch_data.x, batch_data.t)


def check_solution_quality(model, data):
    """Helper function to check solution quality"""
    with torch.no_grad():
        data = data[0]
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


def validate_solution(model, data, loss_fn, verbose=True):
    model.eval()
    base_data = data[0]
    residuals = []
    T = base_data[0]
    X = base_data[1]
    W = base_data[2]
    for i in range(0, len(base_data[0]), 1024):
        t = T[i:i+1024]
        x = X[i:i+1024]
        w = W[i:i+1024]
        u_pred = model(x, t)
        if isinstance(u_pred, tuple):
            u_pred = u_pred[0]
        loss = loss_fn.raw_residual(u_pred, t, w, x)
        residuals.extend(list(loss.detach().cpu().squeeze(1).numpy()))
    hist_counts = np.histogram(residuals, bins=1000)[0]
    hist_probs = hist_counts / np.sum(hist_counts)
    final_loss = np.mean(np.array(residuals)**2)
    ent = entropy(hist_probs, base=2)
    return final_loss

def train_model(
        model: torch.nn.Module,
        optimizer: Optimizer,
        loss_fn: Callable,
        data_generator,
        epochs: int,
        step_fn: Callable,
        train_params: dict,
        verbose: bool = True,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
):
    """
    Generic training loop with optional learning rate scheduler.

    Args:
        model: The neural network model
        optimizer: The optimizer
        loss_fn: Loss function
        data_generator: Data generation function
        epochs: Number of epochs to train
        step_fn: Function defining a single training step
        train_params: Dictionary of training parameters
        verbose: Whether to show progress bar
        scheduler: Optional learning rate scheduler
    """
    if verbose:
        p_bar = tqdm(range(epochs), desc="Training")
        n_params = sum(p.numel() for p in model.parameters())
        tqdm.write(f"Training model with {n_params:.2e} params for {epochs} epochs")
    else:
        p_bar = range(epochs)

    best_val_loss = np.inf
    best_weights = model.state_dict()
    best_stats = {
        "min": 100,
        "max": 100,
        "mean": 100,
        "std": 100,
    }
    val_losses = []
    data_generator.set_epoch(271198)
    val_data = data_generator(1,0,0, white_noise_seed=271198)
    value_splitter = epochs//30
    for epoch in p_bar:
        model.train()

        # Update the epoch in the data generator
        seed = epoch % value_splitter
        data_generator.set_epoch(seed)

        # Get data for this epoch
        data = data_generator(
            *train_params["bound_split"], white_noise_seed=epoch)
        optimizer.zero_grad()

        # Pass unpacked data to the step function
        loss, components = step_fn(model, loss_fn, data, optimizer,
                       train_params, data_generator.n_points)

        if epoch == 0:
            first_loss = loss
        if epoch == 100:
            diff = (loss-first_loss)/first_loss
            if verbose:
                tqdm.write(
                    f"First loss: {first_loss}, 100th loss: {loss}, diff: {loss-first_loss:.2e}, % diff: {diff:.2%}")
            if np.sign(diff) == 1 or abs(diff) < 0.5:
                if verbose:
                    tqdm.write(
                        f"\nPruning due to increasing loss at epoch {epoch}")
                return np.inf, [], model, best_stats
        if epoch % 10 == 0:
            val_loss = validate_solution(model, val_data, loss_fn, verbose)

        # Check solution quality
        if epoch % 10 == 0:
            is_trivial, stats = check_solution_quality(model, data)
            if is_trivial:
                if verbose:
                    tqdm.write(
                        f"\nPruning due to trivial solution at epoch {epoch}")
                    tqdm.write(f"Solution stats: {stats}")
                return np.inf, [], model, stats

        if np.isnan(loss):
            if verbose:
                tqdm.write(
                    f"\nPruning due to NaN loss at epoch {epoch}")
            return np.inf, [], model, stats
    
        val_losses.append(val_loss)

        # Update learning rate if scheduler is provided
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # Use rolling average for plateau scheduler
                scheduler.step(loss)
            else:
                scheduler.step()

        if val_loss < best_val_loss and epoch > 100:
            best_val_loss = val_loss
            best_weights = model.state_dict()
            best_stats = stats

        if verbose:
            postfix = {
                "loss": f"{loss:.2e}",
                "val_loss": f"{val_loss:.2e}",
                "best_val_loss": f"{best_val_loss:.2e}",
                "mean": f"{stats['mean']:.2e}",
                "std": f"{stats['std']:.2e}",
                # Added LR to progress bar
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
            }
            for i, val in enumerate(components):
                postfix[f"l{i}"] = f"{val:.2e}"
            p_bar.set_postfix(postfix)

    if best_stats["abs_mean"] - best_stats["mean"] < 1e-3:
        if verbose:
            tqdm.write(
                f"\nPruning due to trivial solution at end of training")
            tqdm.write(f"Solution stats: {best_stats}")
        return np.inf, [], model, best_stats
    model.load_state_dict(best_weights)
    return best_val_loss, val_losses, model, best_stats


def dgm_heat_step(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    data: Tuple[Tuple[torch.Tensor, ...], ...],
    optimizer: torch.optim.Optimizer,
    train_params: dict,
    data_len: int
) -> float:
    """
    Performs a single step of training with independent tensor copies for each batch.
    """
    model.train()

    batch_size = train_params["batch_size"]
    p_base, p_bound, p_init = train_params["bound_split"]
    base_data, bound_data, init_data = data

    num_batches = max(data_len // batch_size, 1)
    total_loss = 0

    for batch_idx in range(num_batches):
        # Clear gradients
        optimizer.zero_grad(set_to_none=True)

        # Get batch slices
        base_slice = get_batch_slice(batch_idx, batch_size, p_base)
        bound_slice = get_batch_slice(batch_idx, batch_size, p_bound)
        init_slice = get_batch_slice(batch_idx, batch_size, p_init)

        # Extract independent copies of batches
        base_batch = extract_batch(base_data, base_slice)
        bound_batch = extract_batch(bound_data, bound_slice)
        init_batch = extract_batch(init_data, init_slice)

        # Forward pass
        u_base = process_batch(model, base_batch)
        u_bound = process_batch(model, bound_batch)
        u_init = process_batch(model, init_batch)

        # Calculate losses
        loss_components = loss_fn(
            u_base, u_bound, u_init,
            base_batch.t, base_batch.w, base_batch.x, 
            bound_batch.x,
            init_batch.x
        )

        # Sum losses and backward pass
        batch_loss = sum(loss_components)
        batch_loss.backward()

        # Clip gradients if specified
        if train_params.get("clip_grad"):
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                train_params["clip_grad"]
            )

        optimizer.step()

        # Record losses
        total_loss += batch_loss.item()

    # Calculate means
    mean_loss = total_loss / num_batches

    return mean_loss, loss_components


def mim_heat_step(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    data: Tuple[Tuple[torch.Tensor, ...], ...],
    optimizer: torch.optim.Optimizer,
    train_params: dict,
    data_len: int
):

    model.train()

    batch_size = train_params["batch_size"]
    p_base, _, _ = train_params["bound_split"]
    base_data, _, _ = data

    num_batches = max(data_len // batch_size, 1)

    total_loss = 0

    for batch_idx in range(num_batches):
        optimizer.zero_grad(set_to_none=True)

        base_slice = get_batch_slice(batch_idx, batch_size, p_base)

        batch = extract_batch(base_data, base_slice)

        u, p = process_batch(model, batch)

        loss_components = loss_fn(
            u, p, batch.t, batch.w, batch.x,
        )

        batch_loss = sum(loss_components)
        batch_loss.backward()

        if train_params.get("clip_grad"):
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                train_params["clip_grad"]
            )

        optimizer.step()

        total_loss += batch_loss.item()

    mean_loss = total_loss / num_batches

    return mean_loss, loss_components


def train_dgm_heat(
        model: torch.nn.Module,
        optimizer: Optimizer,
        loss_fn: torch.nn.Module,
        data_generator: Callable,
        epochs: int,
        train_params: dict,
        verbose: bool = True,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
):
    """
    Training function for DGM heat equation with optional scheduler.
    """
    return train_model(
        model, optimizer, loss_fn, data_generator, epochs,
        dgm_heat_step, train_params, verbose, scheduler
    )


def train_mim_heat(
        model: torch.nn.Module,
        optimizer: Optimizer,
        loss_fn: torch.nn.Module,
        data_generator: Callable,
        epochs: int,
        train_params: dict,
        verbose: bool = True,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
):
    """
    Training function for MIM heat equation with optional scheduler.
    """
    return train_model(
        model, optimizer, loss_fn, data_generator, epochs,
        mim_heat_step, train_params, verbose, scheduler
    )
