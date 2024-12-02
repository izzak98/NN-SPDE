import torch
import torch.optim as optim
from datetime import datetime
from tqdm import tqdm
from models import DGM, HeatMIM
from utils.viz_utils import gen_heat_snapshots
from heat_train import train_heat, TrainDGM, TrainMIM, adjusted_initial_condition


if __name__ == "__main__":
    alpha = 0.1
    torch.autograd.set_detect_anomaly(True)

    n_dims = 2  # Change this for different dimensions
    boundaries = [(0, 1) for _ in range(n_dims)]  # Boundaries for each dimension

    # model = DGM(
    #     input_dims=n_dims,
    #     hidden_dims=[128, 128, 64],
    #     dgm_dims=0,
    #     n_dgm_layers=3,
    #     hidden_activation="tanh",
    #     output_activation=None,
    # )
    model = HeatMIM(
        input_dims=n_dims,
        hidden_dims=[128, 128, 64],
        dgm_dims=0,
        n_dgm_layers=3,
        hidden_activation="tanh",
        output_activation=None,
        initial_conditions=adjusted_initial_condition,
    )
    # trainer = TrainDGM(lambda1=2, lambda2=5, use_stochastic=True, alpha=alpha)
    trainer = TrainMIM(lambda1=1, use_stochastic=True, alpha=alpha)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    train_heat(
        model,
        optimizer,
        epochs=1000,
        batch_size=2048,
        boundaries=boundaries,
        loss_calculator=trainer,
        num_samples=5,
    )

    gen_heat_snapshots(model, grid_size=100, time_steps=[0.1, 0.25, 0.5, 0.9], name="DGM")
