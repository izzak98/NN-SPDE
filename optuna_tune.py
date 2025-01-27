import optuna
from optuna.trial import TrialState
import numpy as np
import torch
from accelerate import Accelerator
from optuna.samplers import TPESampler
from models import DGM, HeatMIM, BurgerMIM, KPZMIM
from heat_train import HeatTrainDGM, HeatTrainMIM, heat_initial_condition, train_heat
from burger_train import BurgerTrainDGM, BurgerTrainMIM, burger_initial_condition, train_burger
from kpz_train import KPZTrainDGM, KPZTrainMIM, kpz_initial_condition, train_kpz
from utils.optuna_utils import get_model_hyper_params
from utils.config_utils import load_config

CONFIG = load_config()


def tune(trial, n_dims, mim: bool, equation: str):
    boundaries = [(0, 1) for _ in range(n_dims)]  # Boundaries for each dimension
    use_stochastic = True

    if equation == "heat":
        initial_condition = heat_initial_condition
        train_func = train_heat
        mim_model = HeatMIM
        add_dims = 2
        mim_trainer = HeatTrainMIM
        dgm_trainer = HeatTrainDGM
    elif equation == "burger":
        initial_condition = burger_initial_condition
        train_func = train_burger
        mim_model = BurgerMIM
        add_dims = 3
        mim_trainer = BurgerTrainMIM
        dgm_trainer = BurgerTrainDGM
    elif equation == "kpz":
        initial_condition = kpz_initial_condition
        train_func = train_kpz
        mim_model = KPZMIM
        add_dims = 4
        mim_trainer = KPZTrainMIM
        dgm_trainer = KPZTrainDGM
    else:
        raise ValueError(f"Equation {equation} not supported")

    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024, 2048])

    # Model creation based on type (MIM or DGM)
    if mim:
        u_model_params = get_model_hyper_params(trial, "u")
        u_model_params = {f"u_{k}": v for k, v in u_model_params.items()}
        p_model_params = get_model_hyper_params(trial, "p")
        p_model_params = {f"p_{k}": v for k, v in p_model_params.items()}
        model_params = {**u_model_params, **p_model_params}
        model_params["spatial_dims"] = n_dims
        model_params["add_dims"] = add_dims
        model_params["initial_conditions"] = initial_condition
        model = mim_model(**model_params)

        lambda1 = trial.suggest_float("lambda1", 1, 10, log=True)
        trainer = mim_trainer(batch_size, lambda1=lambda1, use_stochastic=use_stochastic)
    else:
        model_params = get_model_hyper_params(trial, "dgm")
        model_params["spatial_dims"] = n_dims
        model_params["add_dims"] = add_dims
        model_params["name"] = f"{equation}_DGM"
        model = DGM(**model_params)
        lambda1 = trial.suggest_float("lambda1", 1, 10, log=True)
        lambda2 = trial.suggest_float("lambda2", 1, 10, log=True)
        trainer = dgm_trainer(batch_size, lambda1=lambda1, lambda2=lambda2,
                              use_stochastic=use_stochastic)

    # Optimizer and scheduler setup
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    l2_reg = trial.suggest_float("l2_reg", 1e-6, 1e-3, log=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=l2_reg)

    use_scheduler = trial.suggest_categorical("use_scheduler", [1, 0])
    use_scheduler = bool(use_scheduler)
    if use_scheduler:
        patience = trial.suggest_int("patience", 5, 20)
        factor = trial.suggest_float("factor", 0.1, 0.9, log=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=factor, patience=patience)
    else:
        scheduler = None

    epochs = CONFIG["epochs"]
    n_points = CONFIG["n_points"]
    num_samples = CONFIG["num_samples"]

    # For MIM models, disable mixed precision
    is_mim = isinstance(trainer, (HeatTrainMIM, BurgerTrainMIM, KPZTrainMIM))

    best_loss = train_func(
        model=model,
        optimizer=optimizer,
        epochs=epochs,
        n_points=n_points,
        boundaries=boundaries,
        loss_calculator=trainer,
        num_samples=num_samples,
        scheduler=scheduler,
        trial_n=str(trial.number),
    )
    return best_loss


def main():
    dims = CONFIG["dims"]
    db_path = CONFIG["db_path"]
    n_optuna_trials = CONFIG["n_optuna_trials"]
    n_jobs = CONFIG["n_jobs"]

    equations = ["heat", "burger", "kpz"]
    for equation in equations:
        for dim in dims:
            completed_trials = 0
            while True:
                # DGM optimization
                dgm_study = optuna.create_study(
                    direction="minimize",
                    storage=f"{db_path}",
                    study_name=f"dgm_{equation}_{dim}D",
                    load_if_exists=True,)
                completed_trials = len(
                    [s for s in dgm_study.trials if s.state == TrialState.COMPLETE])
                if completed_trials >= n_optuna_trials:
                    break

                completed_trials = len(
                    [s for s in dgm_study.trials if s.state == TrialState.COMPLETE])
                dgm_study.optimize(
                    lambda trial: tune(trial, dim, mim=False, equation=equation),
                    n_trials=1,
                    timeout=None,
                    n_jobs=n_jobs,
                )

            completed_trials = 0
            while True:
                # MIM optimization
                mim_study = optuna.create_study(
                    direction="minimize",
                    storage=f"{db_path}",
                    study_name=f"mim_{equation}_{dim}D",
                    load_if_exists=True,)

                completed_trials = len(
                    [s for s in mim_study.trials if s.state == TrialState.COMPLETE])
                if completed_trials >= n_optuna_trials:
                    break
                mim_study.optimize(
                    lambda trial: tune(trial, dim, mim=True, equation=equation),
                    n_trials=1,
                    timeout=None,
                    n_jobs=n_jobs,
                )


if __name__ == "__main__":
    main()
