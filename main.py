import torch

from models import HeatMIM, HeatDGM
from loss_functions import HeatMIMLoss, HeatDGMLoss, BurgerMIMLoss
from data_generators import HeatDataGenerator
from utils.train_utils import train_dgm_heat, train_mim_heat

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(271198)

def train_heat(input_dims=16):
    input_dims = input_dims
    epochs = 300
    dgm_params = {
        "input_dims": input_dims,
        "hidden_dims": [64],
        "dgm_dims": 32,
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

    data_generator = HeatDataGenerator(input_dims, n_points=500)

    dgm_loss_fn = HeatDGMLoss(9, 3, 0.001)

    dgm_optimizer = torch.optim.Adam(dgm.parameters(), lr=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        dgm_optimizer,
        mode='min',
        factor=0.5,      # Reduce LR by half when plateauing
        patience=20,     # Wait 20 epochs before reducing LR
        min_lr=1e-6,    # Don't reduce LR below this
    )

    dgm_best_loss, dgm_losses, dgm, dgm_stats = train_dgm_heat(
        dgm, dgm_optimizer, dgm_loss_fn, data_generator, epochs=epochs, train_params=dgm_train_params, scheduler=scheduler
    )

    print(f"DGM best loss: {dgm_best_loss}, stats: {dgm_stats}")

    mim_params = {
        "input_dims": input_dims, 
        "u_hidden_dims": [64, 128, 512], 
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
        "batch_size": 512,
        "bound_split": [1, 0.0, 0.0],
    }

    mim = HeatMIM(**mim_params).to(DEVICE)

    mim_loss_fn = HeatMIMLoss(0.00001)

    mim_optimizer = torch.optim.Adam(mim.parameters(), lr=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        mim_optimizer,
        mode='min',
        factor=0.5,      # Reduce LR by half when plateauing
        patience=20,     # Wait 20 epochs before reducing LR
        min_lr=1e-6,    # Don't reduce LR below this
    )

    mim_best_loss, mim_losses, mim, mim_stats = train_mim_heat(
        mim, mim_optimizer, mim_loss_fn, data_generator, epochs=epochs, train_params=mim_train_params, scheduler=scheduler
    )

    print(f"MIM best loss: {mim_best_loss}, stats: {mim_stats}")

    return dgm, mim

def train_burger():
    input_dims = 2
    mim_params = {
        "input_dims": input_dims, 
        "u_hidden_dims": [64], 
        "u_activation": 'tanh',
        "u_dgm_dims": 256,
        "u_n_dgm_layers": 2, 
        "u_dgm_activation": 'tanh', 
        "u_out_activation": None,
        "p_hidden_dims": [32],
        "p_activation": 'tanh',
        "p_dgm_dims": 10,
        "p_n_dgm_layers": 1, 
        "p_dgm_activation": 'tanh', 
        "p_out_activation": None
    }
    mim_train_params = {
        "batch_size": 1024,
        "bound_split": [0.8, 0.1, 0.1],
    }

    mim = HeatMIM(**mim_params).to(DEVICE)

    mim_loss_fn = BurgerMIMLoss(0.000001)

    mim_optimizer = torch.optim.Adam(mim.parameters(), lr=1e-4)

    generator = BurgerDataGenerator(n_points=1000, d=input_dims)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        mim_optimizer,
        mode='min',
        factor=0.5,      # Reduce LR by half when plateauing
        patience=20,     # Wait 20 epochs before reducing LR
        min_lr=1e-6,    # Don't reduce LR below this
    )

    mim_best_loss, mim_losses, mim, mim_stats = train_mim_burger(
        model=mim,
        optimizer=mim_optimizer,
        loss_fn=mim_loss_fn,
        data_generator=generator,
        epochs=500,
        train_params=mim_train_params,
        scheduler=scheduler,
        verbose=True
    )

    print(f"MIM best loss: {mim_best_loss}, stats: {mim_stats}")
    return mim

if __name__ == "__main__":
    train_heat()
    # train_burger()
