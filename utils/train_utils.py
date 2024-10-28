import torch
from torch.optim import Optimizer
import numpy as np
from tqdm import tqdm
from typing import Callable, Tuple, NamedTuple, Optional


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
    else:
        p_bar = range(epochs)

    best_val_loss = np.inf
    best_weights = model.state_dict()
    best_stats = {}
    val_losses = []

    for epoch in p_bar:
        model.train()

        # Update the epoch in the data generator
        data_generator.set_epoch(epoch)

        # Get data for this epoch
        data = data_generator()
        optimizer.zero_grad()

        # Pass unpacked data to the step function
        set_output = step_fn(model, loss_fn, data, optimizer,
                             train_params, data_generator.n_points)

        loss = set_output.mean_loss
        val_loss = set_output.mean_residual

        # Check solution quality
        if epoch % 10 == 0:
            is_trivial, stats = check_solution_quality(model, data)
            if is_trivial:
                if verbose:
                    print(
                        f"\nPruning due to trivial solution at epoch {epoch}")
                    print(f"Solution stats: {stats}")
                return np.inf, [], model, stats

        if val_loss == 0:
            if verbose:
                print("\nPruning due to zero loss")
            return np.inf, [], model, stats

        val_losses.append(val_loss)
        compare_loss = torch.mean(
            torch.tensor(val_losses[-10:])).item() if len(val_losses) > 100 else np.inf

        # Update learning rate if scheduler is provided
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # Use rolling average for plateau scheduler
                scheduler.step(loss)
            else:
                scheduler.step()

        if compare_loss < best_val_loss:
            best_val_loss = compare_loss
            best_weights = model.state_dict()
            best_stats = stats

        if verbose:
            postfix = {
                "loss": f"{loss:.2e}",
                "val_loss": f"{val_loss:.2e}",
                "rolling_val_loss": f"{compare_loss:.2e}",
                "best_val_loss": f"{best_val_loss:.2e}",
                "mean": f"{stats['mean']:.2e}",
                "std": f"{stats['std']:.2e}",
                # Added LR to progress bar
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
            }
            p_bar.set_postfix(postfix)

    model.load_state_dict(best_weights)
    return best_val_loss, val_losses, model, best_stats


class BatchData(NamedTuple):
    x: torch.Tensor
    t: torch.Tensor
    w: torch.Tensor = None


def get_batch_slice(start_idx: int, batch_size: int, proportion: float) -> slice:
    """Calculate batch indices based on proportion and batch size."""
    start = int(start_idx * proportion * batch_size)
    end = int((start_idx + 1) * proportion * batch_size)
    return slice(start, end)


def extract_batch(data: Tuple[torch.Tensor, ...], batch_slice: slice) -> BatchData:
    """
    Extract a batch of data from the full dataset, creating independent copies.
    """
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


class StepOutput(NamedTuple):
    mean_loss: float
    mean_residual: float
    loss_components: list


def dgm_heat_step(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    data: Tuple[Tuple[torch.Tensor, ...], ...],
    optimizer: torch.optim.Optimizer,
    train_params: dict,
    data_len: int
) -> StepOutput:
    """
    Performs a single step of training with independent tensor copies for each batch.
    """
    model.train()

    batch_size = train_params["batch_size"]
    p_base, p_bound, p_init = train_params["bound_split"]
    base_data, bound_data, init_data = data

    num_batches = max(data_len // batch_size, 1)
    total_loss = 0
    total_residual = 0
    all_loss_components = []

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
        total_residual += loss_components[0].item()
        all_loss_components.append([comp.item() for comp in loss_components])

    # Calculate means
    mean_loss = total_loss / num_batches
    mean_residual = total_residual / num_batches
    mean_components = [sum(x)/num_batches for x in zip(*all_loss_components)]

    return StepOutput(
        mean_loss=mean_loss,
        mean_residual=mean_residual,
        loss_components=mean_components
    )


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
