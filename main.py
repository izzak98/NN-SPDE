import torch

from models import HeatMIM, BurgersMIM, HeatDGM, BurgersDGM
from loss_functions import HeatMIMLoss, HeatDgmLoss, BurgerDGMLoss, BurgerMIMLoss
from data_generators import HeatDataGenerator, BurgersDataGenerator
from utils.train_utils import train_mim_heat, train_dgm_heat, train_mim_burgers, train_dgm_burgers

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_heat():
    input_dims = 8
    mim_params = {
        "input_dims":input_dims,
        "u_hidden_dims":[50, 50],
        "u_activation":'tanh',
        "u_dgm_dims":50,
        "u_n_dgm_layers":2,
        "u_dgm_activation":'tanh',
        "u_out_activation":None,
        "p_hidden_dims":[50, 50],
        "p_activation":'tanh',
        "p_dgm_dims":50,
        "p_n_dgm_layers":2,
        "p_dgm_activation":'tanh',
        "p_out_activation":None
    }
    dgm_params = {
        "input_dims":input_dims,
        "hidden_dims":[50, 50],
        "dgm_dims":50,
        "n_dgm_layers":2,
        "hidden_activation":'tanh',
        "dgm_activation":'tanh',
        "output_activation":None
    }

    mim = HeatMIM(**mim_params).to(DEVICE)
    dgm = HeatDGM(**dgm_params).to(DEVICE)

    data_generator = HeatDataGenerator(256, input_dims)

    mim_loss_fn = HeatMIMLoss(0.00)
    dgm_loss_fn = HeatDgmLoss(1, 1)

    mim_optimizer = torch.optim.Adam(mim.parameters(), lr=1e-5)
    dgm_optimizer = torch.optim.Adam(dgm.parameters(), lr=1e-4)

    mim_best_loss, mim_losses, mim = train_mim_heat(
        mim, mim_optimizer, mim_loss_fn, data_generator, epochs=1000
    )

    dgm_best_loss, dgm_losses, dgm = train_dgm_heat(
        dgm, dgm_optimizer, dgm_loss_fn, data_generator, epochs=1000
    )

    print(f"MIM best loss: {mim_best_loss}")
    print(f"DGM best loss: {dgm_best_loss}")


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
