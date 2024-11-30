import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from torch.nn.functional import tanh, sigmoid
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from utils.white_noise import BrownianSheet

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set up TensorBoard writer
log_dir = Path("runs") / datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir)

sheet = BrownianSheet(device=device)


def get_activation(activation: str):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    else:
        return lambda x: x


def create_dgm_layers(dgm_dims: int, n_dgm_layers: int):
    if not dgm_dims:
        return

    layers = nn.ModuleList()
    for _ in range(n_dgm_layers):
        layers.append(DGMLayer(dgm_dims))
    return layers


def create_fc_layers(input_dim: int,
                     hidden_dims: list[int],
                     dense_activation: str,
                     dgm_dims: int,
                     n_dgm_layers: int,
                     output_activation: str,
                     output_dim: int = 1):
    dense_layers_list = nn.ModuleList()
    output_layers = nn.ModuleList()
    dgm_layers_list = None
    output_layer_in_params = 1
    input_layer = nn.Linear(input_dim, output_dim)
    if hidden_dims:
        input_layer = nn.Linear(input_dim, hidden_dims[0])
        if dense_activation:
            dense_layers_list.append(get_activation(dense_activation))
        for i in range(1, len(hidden_dims)):
            dense_layers_list.append(
                nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            if dense_activation:
                dense_layers_list.append(get_activation(dense_activation))
        output_layer_in_params = hidden_dims[-1]
        if dgm_dims and n_dgm_layers:
            dense_layers_list.append(nn.Linear(hidden_dims[-1], dgm_dims))

    if dgm_dims and n_dgm_layers:
        if not dense_layers_list:
            input_layer = nn.Linear(input_dim, dgm_dims)
        dgm_layers_list = create_dgm_layers(
            dgm_dims, n_dgm_layers)
        output_layer_in_params = dgm_dims

    output_layers.append(nn.Linear(output_layer_in_params, output_dim))
    if output_activation:
        output_layers.append(get_activation(output_activation))
    output_layer = nn.Sequential(*output_layers)

    if not dense_layers_list and not dgm_layers_list:
        input_layer = nn.Sequential(
            input_layer, get_activation(dense_activation))

    return input_layer, dense_layers_list, dgm_layers_list, output_layer


class DGMLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super(DGMLayer, self).__init__()

        # Initialize all linear layers with Xavier/Glorot initialization
        self.U_z = nn.Linear(hidden_dim, hidden_dim)
        self.W_z = nn.Linear(hidden_dim, hidden_dim)
        self.B_z = nn.Linear(hidden_dim, hidden_dim)

        self.U_g = nn.Linear(hidden_dim, hidden_dim)
        self.W_g = nn.Linear(hidden_dim, hidden_dim)
        self.B_g = nn.Linear(hidden_dim, hidden_dim)

        self.U_r = nn.Linear(hidden_dim, hidden_dim)
        self.W_r = nn.Linear(hidden_dim, hidden_dim)
        self.B_r = nn.Linear(hidden_dim, hidden_dim)

        self.U_h = nn.Linear(hidden_dim, hidden_dim)
        self.W_h = nn.Linear(hidden_dim, hidden_dim)
        self.B_h = nn.Linear(hidden_dim, hidden_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, S, S1):
        Z = sigmoid(self.U_z(x) + self.W_z(S) + self.B_z(S1))
        G = sigmoid(self.U_g(x) + self.W_g(S1) + self.B_g(S))
        R = sigmoid(self.U_r(x) + self.W_r(S) + self.B_r(S1))
        H = tanh(self.U_h(x) + self.W_h(S * R) + self.B_h(S1))
        S_new = (1 - G) * H + Z * S
        return S_new

# Rest of the helper functions remain similar, just moved to device


class HeatDGM(nn.Module):
    def __init__(self,
                 input_dims: int,
                 hidden_dims: list,
                 dgm_dims: int,
                 n_dgm_layers: int,
                 hidden_activation: str,
                 output_activation: str):
        super(HeatDGM, self).__init__()
        input_dims += 1
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.dgm_dims = dgm_dims
        self.n_dgm_layers = n_dgm_layers
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        layers = create_fc_layers(
            input_dims, hidden_dims, hidden_activation, dgm_dims,
            n_dgm_layers, output_activation)
        self.input_layer, self.hidden_layers, self.dgm_layers, self.output_layer = layers

    def initial_condition(self, X):
        with torch.no_grad():
            initial_condition = torch.prod(
                torch.cos(torch.pi * X), dim=1, keepdim=True)
        return initial_condition

    def forward(self, t, x, y):
        inps = torch.cat([x, y, t], dim=1)
        input_x = self.input_layer(inps)

        if self.hidden_layers:
            for layer in self.hidden_layers:
                input_x = layer(input_x)

        if self.dgm_layers:
            S1 = input_x
            S = input_x
            for layer in self.dgm_layers:
                S = layer(input_x, S, S1)
            input_x = S

        return self.output_layer(input_x)


def heat_residual_loss_2d(model, t, x, y, alpha, w):
    t.requires_grad = True
    x.requires_grad = True
    y.requires_grad = True
    u = model(t, x, y)
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    residual = u_t - alpha * (u_xx + u_yy) + u * w
    return torch.mean(residual**2)

# Loss for the initial condition


def initial_condition_loss_2d(model, x, y, u0_func):
    t0 = torch.zeros_like(x)
    u0_pred = model(t0, x, y)
    u0_true = u0_func(x, y)
    return torch.mean((u0_pred - u0_true)**2)

# Loss for Neumann boundary conditions


def neumann_boundary_condition_loss_2d(model, t, x_min, x_max, y_min, y_max):
    # Derivative with respect to x at x=0 and x=L
    x_min.requires_grad = True
    x_max.requires_grad = True
    u_x_min = model(t, x_min, y_min)
    u_x_max = model(t, x_max, y_min)
    du_dx_min = torch.autograd.grad(
        u_x_min, x_min, grad_outputs=torch.ones_like(u_x_min), create_graph=True)[0]
    du_dx_max = torch.autograd.grad(
        u_x_max, x_max, grad_outputs=torch.ones_like(u_x_max), create_graph=True)[0]

    # Derivative with respect to y at y=0 and y=L
    y_min.requires_grad = True
    y_max.requires_grad = True
    u_y_min = model(t, x_min, y_min)
    u_y_max = model(t, x_min, y_max)
    du_dy_min = torch.autograd.grad(
        u_y_min, y_min, grad_outputs=torch.ones_like(u_y_min), create_graph=True)[0]
    du_dy_max = torch.autograd.grad(
        u_y_max, y_max, grad_outputs=torch.ones_like(u_y_max), create_graph=True)[0]

    return (
        torch.mean(du_dx_min**2)
        + torch.mean(du_dx_max**2)
        + torch.mean(du_dy_min**2)
        + torch.mean(du_dy_max**2)
    )


def compute_losses(model, batch, alpha, sheet):
    t, x, y, x_min, x_max, y_min, y_max = [b.to(device) for b in batch]

    # Compute all losses
    w_points = torch.cat([t, x, y], dim=1)
    w = sheet.simulate(w_points) * 0

    loss_residual = heat_residual_loss_2d(model, t, x, y, alpha, w)
    loss_initial = initial_condition_loss_2d(
        model, x, y, lambda x, y: torch.cos(torch.pi * x) * torch.cos(torch.pi * y))
    loss_boundary = neumann_boundary_condition_loss_2d(
        model, t, x_min, x_max, y_min, y_max)

    total_loss = loss_residual + 1 * loss_initial + 1 * loss_boundary

    return {
        'total_loss': total_loss,
        'residual_loss': loss_residual,
        'initial_loss': loss_initial,
        'boundary_loss': loss_boundary
    }


def train_heat_equation_2d_with_neumann(model, optimizer, alpha, epochs, batch_size):
    model = model.to(device)
    scaler = torch.amp.GradScaler()  # For mixed precision training

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100, factor=0.5)

    best_loss = float('inf')
    pbar = tqdm(range(epochs), desc="Training")
    for epoch in pbar:
        model.train()

        # Generate batch data
        t = torch.rand((batch_size, 1), device=device)
        x = torch.rand((batch_size, 1), device=device) * L
        y = torch.rand((batch_size, 1), device=device) * L
        x_min = torch.zeros((batch_size, 1), device=device)
        x_max = torch.ones((batch_size, 1), device=device) * L
        y_min = torch.zeros((batch_size, 1), device=device)
        y_max = torch.ones((batch_size, 1), device=device) * L

        batch = (t, x, y, x_min, x_max, y_min, y_max)

        # Mixed precision training
        with torch.amp.autocast(str(device)):
            losses = compute_losses(model, batch, alpha, sheet)
            loss = losses['total_loss']

        # Optimization step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update learning rate
        scheduler.step(loss)

        # Log metrics
        writer.add_scalar('Loss/total', losses['total_loss'].item(), epoch)
        writer.add_scalar('Loss/residual', losses['residual_loss'].item(), epoch)
        writer.add_scalar('Loss/initial', losses['initial_loss'].item(), epoch)
        writer.add_scalar('Loss/boundary', losses['boundary_loss'].item(), epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        pbar.set_postfix({key: f"{value.item():.2e}" for key, value in losses.items()})
        # Save best model
        if loss < best_loss:
            best_loss = loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, log_dir / 'best_model.pt')

    writer.close()


def gen_heat_snapshots(model, grid_size=100, time_steps=[0.1, 0.25, 0.5, 0.9]):
    # Define the grid size and spatial coordinates
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.vstack([xx.ravel(), yy.ravel()]).T

    device = next(model.parameters()).device

    # Convert spatial grid to torch tensor
    grid_tensor = torch.tensor(grid_points).float().to(device)

    # Precompute time tensors for the specified time steps
    time_tensors = [torch.full((grid_tensor.shape[0], 1), t, device=device) for t in time_steps]

    # Create subplots for the snapshots
    fig, axes = plt.subplots(1, len(time_steps), figsize=(15, 5), constrained_layout=True)
    if len(time_steps) == 1:  # Handle case for a single time step
        axes = [axes]

    for i, (ax, time_tensor, t) in enumerate(zip(axes, time_tensors, time_steps)):
        with torch.no_grad():
            output = model(time_tensor, grid_tensor[:, 0:1], grid_tensor[:, 1:2])
            if isinstance(output, tuple):
                output = output[0]
        solution = output.cpu().numpy().reshape(grid_size, grid_size)

        # Plot the solution
        im = ax.imshow(solution, extent=(0, 1, 0, 1), origin="lower", cmap="hot", vmin=-1, vmax=1)
        ax.set_title(f"t = {t:.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, label="Temperature")

    # Show the plot
    plt.suptitle("Stochastic Heat Equation Snapshots")
    plt.show()


if __name__ == "__main__":
    # Parameters
    L = 1.0
    T = 1.0
    alpha = 0.1

    # Initialize model and optimizer with improved architecture
    model = HeatDGM(
        input_dims=2,
        hidden_dims=[128, 128, 64],  # Deeper network
        dgm_dims=40,
        n_dgm_layers=3,
        hidden_activation='tanh',
        output_activation=None,
    )

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # Train the model
    train_heat_equation_2d_with_neumann(model, optimizer, alpha=alpha, epochs=2000, batch_size=2048)

    # Generate visualization
    gen_heat_snapshots(model)
