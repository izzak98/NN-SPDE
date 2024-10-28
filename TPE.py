import optuna
from optuna.samplers import TPESampler
from optuna.trial import TrialState
import torch
import numpy as np
import random

from utils.optuna_utils import gen_dgm_params, gen_mim_params
from utils.train_utils import train_dgm_heat
from utils.config_utils import load_config
from data_generators import HeatDataGenerator, BurgersDataGenerator
from loss_functions import HeatMIMLoss, HeatDgmLoss, BurgerDGMLoss, BurgerMIMLoss
from models import HeatMIM, HeatDGM, BurgersMIM, BurgersDGM


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG = load_config()

np.random.seed(CONFIG["random_seed"])
random.seed(CONFIG["random_seed"])
torch.manual_seed(CONFIG["random_seed"])


def dgm_heat_objective(trial, dims, epochs, batch_size):
    dgm_params = gen_dgm_params(trial)
    lambda_1 = trial.suggest_float("lambda_1", 0.1, 10.0, log=True)
    lambda_2 = trial.suggest_float("lambda_2", 0.1, 10.0, log=True)
    lambda_3 = trial.suggest_float("lambda_3", 0.1, 10.0, log=True)
    lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
    model = HeatDGM(input_dims=dims, **dgm_params).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    data_generator = HeatDataGenerator(
        d = dims,
        n_points=CONFIG["n_points"],
        base_seed=CONFIG["random_seed"]
    )
    use_scheduler = trial.suggest_categorical("use_scheduler", [0, 1])
    use_scheduler = use_scheduler == 1
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=trial.suggest_float("scheduler_factor", 0.1, 0.9),
            patience=trial.suggest_int("scheduler_patience", 5, 50),
            min_lr=1e-6,
        )
    else:
        scheduler = None

    train_params = {}

    batch_size = trial.suggest_categorical(
        "batch_size", [32, 64, 128, 256, 512, 1024, 2048])
    train_params["batch_size"] = batch_size

    use_clip_grad = trial.suggest_categorical("use_clip_grad", [0, 1])
    use_clip_grad = use_clip_grad == 1
    if use_clip_grad:
        clip_grad = trial.suggest_float("clip_grad", 0.1, 1.0)
        train_params["clip_grad"] = clip_grad

    proportion_base = trial.suggest_float("proportion_base", 0, 1)
    proportion_bound = trial.suggest_float("proportion_bound", 0, 1)
    proportion_init = trial.suggest_float("proportion_init", 0, 1)

    proportion_base = proportion_base / \
        (proportion_base + proportion_bound + proportion_init)
    proportion_bound = proportion_bound / \
        (proportion_base + proportion_bound + proportion_init)
    proportion_init = proportion_init / \
        (proportion_base + proportion_bound + proportion_init)

    train_params["bound_split"] = [
        proportion_base, proportion_bound, proportion_init]

    loss_fn = HeatDgmLoss(lambda_1, lambda_2, lambda_3)
    best_loss, _, _, stats = train_dgm_heat(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        data_generator=data_generator,
        epochs=epochs,
        train_params=train_params,
        scheduler=scheduler,
        verbose=False
    )
    print(f"Trial {trial.number} stats: {stats}")
    return best_loss


def mim_heat_objective(trial, dims, epochs, batch_size):
    mim_params = gen_mim_params(trial)
    lr = trial.suggest_float("lr", 1e-6, 1e-2)
    model = HeatMIM(input_dims=dims, **mim_params).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    data_generator = HeatDataGenerator(batch_size, dims)
    loss_fn = HeatMIMLoss()
    best_loss, _, _ = train_mim_heat(
        model, optimizer, loss_fn, data_generator, epochs)
    return best_loss


def main():
    epochs = CONFIG["epochs"]
    batch_size = CONFIG["batch_size"]
    trials = CONFIG["trials"]
    threads = CONFIG["threads"]
    study_dims = CONFIG["study_dims"]

    for dims in study_dims:
        sampler = TPESampler(seed=CONFIG["random_seed"])
        dgm_study = optuna.create_study(
            direction="minimize", study_name=f"DGM Heat {dims}D",
            storage="sqlite:///dgm_heat.db", load_if_exists=True, sampler=sampler)
        completed_trials = len(
            [s for s in dgm_study.trials if s.state == TrialState.COMPLETE])
        dgm_study.optimize(lambda trial: dgm_heat_objective(trial, dims, epochs, batch_size),
                           n_trials=trials-completed_trials, n_jobs=threads)

    # for dims in study_dims:
    #     mim_study = optuna.create_study(
    #         direction="minimize", study_name=f"MIM Heat {dims}D",
    #         storage="sqlite:///mim_heat.db", load_if_exists=True)
    #     completed_trials = len(
    #         [s for s in mim_study.trials if s.state == TrialState.COMPLETE])
    #     mim_study.optimize(lambda trial: mim_heat_objective(trial, dims, epochs, batch_size),
    #                        n_trials=trials-completed_trials, n_jobs=threads)


if __name__ == "__main__":
    main()
