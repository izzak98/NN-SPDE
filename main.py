import torch

from models import HeatMIM, HeatDGM
from loss_functions import HeatMIMLoss, HeatDGMLoss
from data_generators import HeatDataGenerator
from utils.train_utils import train_dgm_heat, train_mim_heat

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(271198)


def train_heat(input_dims=2, epochs=150):

    dgm_params = {
        "input_dims": input_dims,
        "hidden_dims": [64, 64, 128],
        "dgm_dims": 64,
        "n_dgm_layers": 3,
        "hidden_activation": 'relu',
        "output_activation": None,
    }
    dgm_train_params = {
        "batch_size": 2048,
        "bound_split": [0.34, 0.33, 0.33],
        "clip_grad": 0.278,
    }

    dgm = HeatDGM(**dgm_params).to(DEVICE)

    data_generator = HeatDataGenerator(input_dims, n_points=2000)

    dgm_loss_fn = HeatDGMLoss(1, 1, 0)

    dgm_optimizer = torch.optim.AdamW(dgm.parameters(), lr=1e-4, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        dgm_optimizer,
        mode='min',
        factor=0.5,      # Reduce LR by half when plateauing
        patience=20,     # Wait 20 epochs before reducing LR
        min_lr=1e-10,    # Don't reduce LR below this
    )

    dgm_best_loss, dgm_losses, dgm, dgm_stats = train_dgm_heat(
        dgm, dgm_optimizer, dgm_loss_fn, data_generator, epochs=epochs, train_params=dgm_train_params, scheduler=scheduler
    )

    print(f"DGM best loss: {dgm_best_loss}, stats: {dgm_stats}")

    mim_params = {
        "input_dims": input_dims,
        "u_hidden_dims": [64, 128],
        "u_activation": 'tanh',
        "u_dgm_dims": 30,
        "u_n_dgm_layers": 2,
        "u_out_activation": None,
        "p_hidden_dims": [128, 512],
        "p_activation": 'tanh',
        "p_dgm_dims": 15,
        "p_n_dgm_layers": 2,
        "p_out_activation": None
    }
    mim_train_params = {
        "batch_size": 2048,
        "bound_split": [1, 0.0, 0.0],
    }

    mim = HeatMIM(**mim_params).to(DEVICE)

    mim_loss_fn = HeatMIMLoss(0)

    mim_optimizer = torch.optim.AdamW(mim.parameters(), lr=1e-4, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        mim_optimizer,
        mode='min',
        factor=0.5,      # Reduce LR by half when plateauing
        patience=20,     # Wait 20 epochs before reducing LR
        min_lr=1e-10,    # Don't reduce LR below this
    )

    mim_best_loss, mim_losses, mim, mim_stats = train_mim_heat(
        mim, mim_optimizer, mim_loss_fn, data_generator, epochs=epochs, train_params=mim_train_params, scheduler=scheduler
    )

    print(f"MIM best loss: {mim_best_loss}, stats: {mim_stats}")

    return dgm, mim


if __name__ == "__main__":
    train_heat()
