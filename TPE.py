import optuna
import torch

from utils.optuna_utils import gen_dgm_params, gen_mim_params
from utils.train_utils import train_mim_heat, train_dgm_heat, train_mim_burgers, train_dgm_burgers
from utils.config_utils import load_config
from data_generators import HeatDataGenerator, BurgersDataGenerator
from loss_functions import HeatMIMLoss, HeatDgmLoss, BurgerDGMLoss, BurgerMIMLoss
from models import HeatMIM, HeatDGM, BurgersMIM, BurgersDGM


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG = load_config()

def dgm_heat_objective(trial, dims, epochs, batch_size):
    dgm_params = gen_dgm_params(trial)
    lambda_1 = trial.suggest_float("lambda_1", 0.1, 10.0)
    lambda_2 = trial.suggest_float("lambda_2", 0.1, 10.0)
    lr = trial.suggest_float("lr", 1e-6, 1e-2)
    HeatDgmLoss(lambda_1, 1.0)
    model = HeatDGM(input_dims=dims, **dgm_params).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    data_generator = HeatDataGenerator(batch_size, dims)
    loss_fn = HeatDgmLoss(lambda_1, lambda_2)
    best_loss, _, _ = train_dgm_heat(
        model, optimizer, loss_fn, data_generator, epochs)
    return best_loss


def mim_heat_objective(trial, dims, epochs, batch_size):
    mim_params = gen_mim_params(trial)
    lambda_1 = trial.suggest_float("lambda_1", 0.1, 10.0)
    lambda_2 = trial.suggest_float("lambda_2", 0.1, 10.0)
    lr = trial.suggest_float("lr", 1e-6, 1e-2)
    model = HeatMIM(input_dims=dims, **mim_params).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    data_generator = HeatDataGenerator(batch_size, dims)
    loss_fn = HeatMIMLoss(lambda_1, lambda_2)
    best_loss, _, _ = train_mim_heat(
        model, optimizer, loss_fn, data_generator, epochs)
    return best_loss


def main():
    epochs = CONFIG["epochs"]
    batch_size = CONFIG["batch_size"]
    trials = CONFIG["trials"]
    threads = CONFIG["threads"]

    for dims in [1, 2, 4, 8, 16, 32]:
        dgm_study = optuna.create_study(
                direction="minimize", study_name=f"DGM Heat {dims}D",
                storage="sqlite:///dgm_heat.db", load_if_exists=True)
        completed_trials = len(dgm_study.trials)

if __name__ == "__main__":
    main()