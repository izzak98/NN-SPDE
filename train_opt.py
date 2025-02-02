import torch
import os
import optuna
from utils.optuna_utils import get_model_hyper_params
from models import DGM, HeatMIM, BurgerMIM, KPZMIM
from heat_train import HeatTrainDGM, HeatTrainMIM, heat_initial_condition, train_heat
from burger_train import BurgerTrainDGM, BurgerTrainMIM, burger_initial_condition, train_burger
from kpz_train import KPZTrainDGM, KPZTrainMIM, kpz_initial_condition, train_kpz
from utils.optuna_utils import get_best_model_hyper_params
from utils.config_utils import load_config

CONFIG = load_config()


def train_with_best_params(best_trial, n_dims, mim: bool, equation: str, model_name: str):
    """Trains a model using the best hyperparameters from an Optuna study."""

    boundaries = [(0, 1) for _ in range(n_dims)]
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

    batch_size = best_trial.params["batch_size"]

    # Model creation based on type (MIM or DGM)
    if mim:
        u_model_params = get_best_model_hyper_params(best_trial, "u")
        u_model_params = {f"u_{k}": v for k, v in u_model_params.items()}
        p_model_params = get_best_model_hyper_params(best_trial, "p")
        p_model_params = {f"p_{k}": v for k, v in p_model_params.items()}
        model_params = {**u_model_params, **p_model_params}
        model_params["spatial_dims"] = n_dims
        model_params["add_dims"] = add_dims
        model_params["initial_conditions"] = initial_condition
        model = mim_model(**model_params)

        lambda1 = best_trial.params["lambda1"]
        trainer = mim_trainer(batch_size, lambda1=lambda1, use_stochastic=use_stochastic)
    else:
        model_params = get_best_model_hyper_params(best_trial, "dgm")
        model_params["spatial_dims"] = n_dims
        model_params["add_dims"] = add_dims
        model_params["name"] = f"{equation}_DGM"
        model = DGM(**model_params)

        lambda1 = best_trial.params["lambda1"]
        lambda2 = best_trial.params["lambda2"]
        trainer = dgm_trainer(batch_size, lambda1=lambda1, lambda2=lambda2,
                              use_stochastic=use_stochastic)

    # Optimizer and scheduler setup
    lr = best_trial.params["lr"]
    l2_reg = best_trial.params["l2_reg"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=l2_reg)

    use_scheduler = bool(best_trial.params["use_scheduler"])
    if use_scheduler:
        patience = best_trial.params["patience"]
        factor = best_trial.params["factor"]
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=factor, patience=patience)
    else:
        scheduler = None

    epochs = CONFIG["epochs"]
    n_points = CONFIG["n_points"]
    num_samples = CONFIG["num_samples"]

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
        trial_n="best_trial",
    )
    torch.save(model, f"models/{model_name}.pth")
    return best_loss


def train_models():
    dims = CONFIG["dims"]
    db_path = CONFIG["db_path"]

    equations = ["heat", "burger", "kpz"]
    if not os.path.exists("models"):
        os.makedirs("models")
    for equation in equations:
        for dim in dims:
            model_name = f"dgm_{equation}_{dim}D"
            if os.path.exists(f"models/{model_name}.pth"):
                print(f"Model {model_name} is already trained. Skipping...")
                continue
            study = optuna.load_study(
                study_name=model_name,
                storage=db_path,
            )
            best_study = study.best_trial
            train_with_best_params(best_study,
                                   dim,
                                   mim=False,
                                   equation=equation,
                                   model_name=model_name)
            print(f"Model {model_name} has been trained.")

            model_name = f"mim_{equation}_{dim}D"
            if os.path.exists(f"models/{model_name}.pth"):
                print(f"Model {model_name} is already trained. Skipping...")
                continue
            study = optuna.load_study(
                study_name=model_name,
                storage=db_path,
            )
            best_study = study.best_trial
            train_with_best_params(best_study,
                                   dim,
                                   mim=True,
                                   equation=equation,
                                   model_name=model_name)
            print(f"Model {model_name} has been trained.")


if __name__ == "__main__":
    train_models()
