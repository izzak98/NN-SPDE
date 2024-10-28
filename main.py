import torch

from models import HeatMIM, BurgersMIM, HeatDGM, BurgersDGM
from loss_functions import HeatMIMLoss, HeatDgmLoss, BurgerDGMLoss, BurgerMIMLoss
from data_generators import HeatDataGenerator, BurgersDataGenerator
from utils.train_utils import train_dgm_heat

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)


def train_heat():
    input_dims = 8
    dgm_params = {
        "input_dims": input_dims,
        "hidden_dims": [64],
        "dgm_dims": 10,
        "n_dgm_layers": 3,
        "hidden_activation": 'tanh',
        "dgm_activation": 'tanh',
        "output_activation": None
    }
    dgm_train_params = {
        "batch_size": 1024,
        "bound_split": [0.8, 0.1, 0.1],
    }

    # mim = HeatMIM(**mim_params).to(DEVICE)
    dgm = HeatDGM(**dgm_params).to(DEVICE)

    data_generator = HeatDataGenerator(input_dims, n_points=1000)

    mim_loss_fn = HeatMIMLoss(0.00)
    dgm_loss_fn = HeatDgmLoss(0.1, 0.1, 0)

    # mim_optimizer = torch.optim.Adam(mim.parameters(), lr=1e-5)
    dgm_optimizer = torch.optim.Adam(dgm.parameters(), lr=1e-4)

    # mim_best_loss, mim_losses, mim, mim_stats = train_mim_heat(
    #     mim, mim_optimizer, mim_loss_fn, data_generator, epochs=11
    # )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        dgm_optimizer,
        mode='min',
        factor=0.5,      # Reduce LR by half when plateauing
        patience=20,     # Wait 20 epochs before reducing LR
        min_lr=1e-6,    # Don't reduce LR below this
        verbose=True    # Print when LR is reduced
    )

    dgm_best_loss, dgm_losses, dgm, dgm_stats = train_dgm_heat(
        dgm, dgm_optimizer, dgm_loss_fn, data_generator, epochs=1000, train_params=dgm_train_params, scheduler=scheduler
    )

    # print(f"MIM best loss: {mim_best_loss}, stats: {mim_stats}")
    print(f"DGM best loss: {dgm_best_loss}, stats: {dgm_stats}")


def train_burger():
    input_dims = 16
    mim_params = {
        "input_dims": input_dims,
        "u_hidden_dims": [1024, 128, 64],
        "p_hidden_dims": [1024, 128, 64],
        "activation": "relu"
    }
    dgm_params = {
        "input_dims": input_dims,
        "hidden_dims": [128, 128],
        "activation": "relu"
    }

    mim = BurgersMIM(**mim_params).to(DEVICE)
    dgm = BurgersDGM(**dgm_params).to(DEVICE)

    data_generator = BurgersDataGenerator(2048, input_dims)

    mim_loss_fn = BurgerMIMLoss()
    dgm_loss_fn = BurgerDGMLoss(1.0, 1.0, 1.0)

    mim_optimizer = torch.optim.Adam(mim.parameters(), lr=1e-4)
    dgm_optimizer = torch.optim.Adam(dgm.parameters(), lr=1e-4)

    mim_best_loss, mim_losses, mim = train_mim_burgers(
        mim, mim_optimizer, mim_loss_fn, data_generator, epochs=100
    )

    dgm_best_loss, dgm_losses, dgm = train_dgm_burgers(
        dgm, dgm_optimizer, dgm_loss_fn, data_generator, epochs=100
    )

    print(f"MIM best loss: {mim_best_loss}")
    print(f"DGM best loss: {dgm_best_loss}")


if __name__ == "__main__":
    train_heat()
    # train_burger()
