import torch
import torch.optim as optim
from datetime import datetime
from tqdm import tqdm
from models import DGM, HeatMIM
from utils.viz_utils import gen_heat_snapshots
from heat_train import train_heat, TrainDGM, TrainMIM, adjusted_initial_condition


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)

    n_dims = 2  # Change this for different dimensions
    boundaries = [(0, 1) for _ in range(n_dims)]  # Boundaries for each dimension

    for use_stochastic in [False, True]:
        post_fix = "stochastic" if use_stochastic else "deterministic"
        dgm_model = DGM(
            input_dims=n_dims,
            hidden_dims=[128, 128, 64],
            dgm_dims=0,
            n_dgm_layers=3,
            hidden_activation="tanh",
            output_activation=None,
        )

        dgm_trainer = TrainDGM(lambda1=2, lambda2=5, use_stochastic=use_stochastic)
        dgm_optimizer = optim.AdamW(dgm_model.parameters(), lr=1e-3, weight_decay=1e-5)

        # train_heat(
        #     dgm_model,
        #     dgm_optimizer,
        #     epochs=1000,
        #     batch_size=5000,
        #     boundaries=boundaries,
        #     loss_calculator=dgm_trainer,
        #     num_samples=5 if use_stochastic else 1,
        # )

        # gen_heat_snapshots(dgm_model, grid_size=100, time_steps=[
        #                    0.1, 0.25, 0.5, 0.9], name=f"DGM_{post_fix}", add_values=[0.1])

        mim_model = HeatMIM(
            input_dims=n_dims,
            hidden_dims=[128, 128, 64],
            dgm_dims=0,
            n_dgm_layers=3,
            hidden_activation="tanh",
            output_activation=None,
            initial_conditions=adjusted_initial_condition,
        )
        mim_trainer = TrainMIM(lambda1=1, use_stochastic=use_stochastic)
        mim_optimizer = optim.AdamW(mim_model.parameters(), lr=1e-3, weight_decay=1e-5)

        train_heat(
            mim_model,
            mim_optimizer,
            epochs=1000,
            batch_size=5000,
            boundaries=boundaries,
            loss_calculator=mim_trainer,
            num_samples=5 if use_stochastic else 1,
        )

        gen_heat_snapshots(mim_model, grid_size=100, time_steps=[
                           0.1, 0.25, 0.5, 0.9], name=f"MIM_{post_fix}", add_values=[0.1])
