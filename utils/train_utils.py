"""Module for training utilities."""
from typing import Callable
import numpy as np
from tqdm import tqdm
from torch import nn
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_mim_heat(
        model: nn.Module,
        optimzer,
        loss_fn: nn.Module,
        data_generator: Callable,
        epochs: int,
        verbose: bool = True
):
    if verbose:
        p_bar = tqdm(range(epochs), desc="Training")
    else:
        p_bar = range(epochs)
    best_val_loss = np.inf
    val_losses = []
    best_weights = model.state_dict()
    for epoch in p_bar:
        model.train()
        t, X, w, _ = data_generator()
        optimzer.zero_grad()
        u, p = model(X, t)
        residual, consitancy = loss_fn(u, p, t, w, X)
        loss = residual + consitancy
        loss.backward()
        optimzer.step()
        val_losses.append(residual.item())
        if residual.item() < best_val_loss:
            best_val_loss = residual.item()
            best_weights = model.state_dict()
        if verbose:
            p_bar.set_postfix({"loss": loss.item(),
                               "val_loss": residual.item(),
                               "best_val_loss": best_val_loss})

    model.load_state_dict(best_weights)
    return best_val_loss, val_losses, model


def train_dgm_heat(
        model: nn.Module,
        optimzer,
        loss_fn: nn.Module,
        data_generator: Callable,
        epochs: int,
        verbose: bool = True
):
    if verbose:
        p_bar = tqdm(range(epochs), desc="Training")
    else:
        p_bar = range(epochs)
    best_val_loss = np.inf
    best_weights = model.state_dict()
    val_losses = []
    for epoch in p_bar:
        model.train()
        t, X, w, boundry_mask = data_generator()
        t_0 = torch.zeros_like(t)
        optimzer.zero_grad()
        u = model(X, t)
        u0 = model(X, t_0)
        residual, initial, boundary = loss_fn(u, u0, t, w, X, boundry_mask)
        loss = residual + initial + boundary
        val_loss = residual.item()
        loss.backward()
        optimzer.step()
        val_losses.append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.state_dict()
        if verbose:
            p_bar.set_postfix({"loss": loss.item(),
                               "val_loss": residual.item(),
                               "best_val_loss": best_val_loss})
    model.load_state_dict(best_weights)
    return best_val_loss, val_losses, model
