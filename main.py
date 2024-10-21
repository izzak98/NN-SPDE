import torch

from models import HeatMIM, DGM
from loss_functions import HeatMIMLoss, HeatDgmLoss
from data_generators import HeatDataGenerator
from utils.train_utils import train_mim_heat, train_dgm_heat

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    input_dims = 8
    mim_params = {
        "input_dims": input_dims,
        "u_hidden_dims": [128, 128],
        "p_hidden_dims": [128, 128],
        "activation": "relu"
    }
    dgm_params = {
        "input_dims": input_dims,
        "hidden_dims": [128, 128],
        "activation": "relu"
    }

    mim = HeatMIM(**mim_params).to(DEVICE)
    dgm = DGM(**dgm_params).to(DEVICE)

    data_generator = HeatDataGenerator(256, input_dims)

    mim_loss_fn = HeatMIMLoss()
    dgm_loss_fn = HeatDgmLoss(1.0, 1.0)

    mim_optimizer = torch.optim.Adam(mim.parameters(), lr=1e-4)
    dgm_optimizer = torch.optim.Adam(dgm.parameters(), lr=1e-4)

    mim_best_loss, mim_losses, mim = train_mim_heat(
        mim, mim_optimizer, mim_loss_fn, data_generator, epochs=1000
    )

    dgm_best_loss, dgm_losses, dgm = train_dgm_heat(
        dgm, dgm_optimizer, dgm_loss_fn, data_generator, epochs=1000
    )

    print(f"MIM best loss: {mim_best_loss}")
    print(f"DGM best loss: {dgm_best_loss}")

if __name__ == "__main__":
    main()