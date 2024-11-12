import optuna
from optuna.samplers import TPESampler
from optuna.trial import TrialState
import torch
import numpy as np
import random

from utils.optuna_utils import gen_dgm_params, gen_mim_params
from utils.train_utils import train_dgm_heat, train_mim_heat
from utils.config_utils import load_config
from data_generators import HeatDataGenerator
from loss_functions import HeatMIMLoss, HeatDGMLoss
from models import HeatMIM, HeatDGM


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG = load_config()

np.random.seed(CONFIG["random_seed"])
random.seed(CONFIG["random_seed"])
torch.manual_seed(CONFIG["random_seed"])


def dgm_heat_objective(trial, dims, epochs):
    dgm_params = gen_dgm_params(trial)
    lambda_1 = trial.suggest_float("lambda_1", 1, 10.0, log=True)
    lambda_2 = trial.suggest_float("lambda_2", 1, 10.0, log=True)
    lambda_3 = trial.suggest_float("lambda_3", 0.001, 10.0, log=True)
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

    # proportion_base = trial.suggest_float("proportion_base", 0, 1)
    # proportion_bound = trial.suggest_float("proportion_bound", 0, 1)
    # proportion_init = trial.suggest_float("proportion_init", 0, 1)

    # proportion_base = proportion_base / \
    #     (proportion_base + proportion_bound + proportion_init)
    # proportion_bound = proportion_bound / \
    #     (proportion_base + proportion_bound + proportion_init)
    # proportion_init = proportion_init / \
    #     (proportion_base + proportion_bound + proportion_init)
    proportion_base = 0.34
    proportion_bound = 0.33
    proportion_init = 0.33

    train_params["bound_split"] = [
        proportion_base, proportion_bound, proportion_init]

    loss_fn = HeatDGMLoss(lambda_1, lambda_2, lambda_3)
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

def train_best_dgm_model(study, dims, epochs):
    # Retrieve the best trial's parameters
    best_params = study.best_trial.params
    
    # Extract DGM specific parameters
    n_hidden = best_params["n_hidden"]
    hidden_dims = [best_params[f"n_params_{i}"] for i in range(n_hidden)]
    dgm_dims = best_params["dgm_dims"]
    
    dgm_params = {
        "hidden_dims": hidden_dims,
        "dgm_dims": dgm_dims,
        "n_dgm_layers": best_params["n_dgm_layers"] if dgm_dims > 0 else 0,
        "hidden_activation": best_params["hidden_activation"],
        "output_activation": best_params["output_activation"],
    }
    
    # Initialize model with best parameters
    model = HeatDGM(input_dims=dims, **dgm_params).to(DEVICE)
    
    # Setup optimizer with the best learning rate
    lr = best_params["lr"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Set up data generator
    data_generator = HeatDataGenerator(
        d=dims,
        n_points=CONFIG["n_points"],
        base_seed=CONFIG["random_seed"]
    )
    
    # Configure scheduler if enabled in best trial
    use_scheduler = best_params["use_scheduler"]
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=best_params["scheduler_factor"],
            patience=best_params["scheduler_patience"],
            min_lr=1e-6
        )
    else:
        scheduler = None

    # Prepare training parameters
    train_params = {"batch_size": best_params["batch_size"]}
    if best_params["use_clip_grad"]:
        train_params["clip_grad"] = best_params["clip_grad"]
    
    train_params["bound_split"] = [
        0.34, 
        0.33, 
        0.33
    ]
    
    # Define loss function with best lambda values
    loss_fn = HeatDGMLoss(best_params["lambda_1"], best_params["lambda_2"], best_params["lambda_3"])

    # Train the model
    best_loss, _, _, stats = train_dgm_heat(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        data_generator=data_generator,
        epochs=epochs,
        train_params=train_params,
        scheduler=scheduler,
        verbose=True
    )
    
    print(f"Training completed with best loss: {best_loss}")
    return model, best_loss, stats

def mim_heat_objective(trial, dims, epochs):
    mim_params = gen_mim_params(trial)
    lambda_1 = trial.suggest_float("lambda_1", 0.001, 10.0, log=True)
    lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True)

    model = HeatMIM(input_dims=dims, **mim_params).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    data_generator = HeatDataGenerator(
        d=dims,
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
            min_lr=1e-7,
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

    train_params["bound_split"] = [
        1, 0, 0]
    
    loss_fn = HeatMIMLoss(lambda_1)

    best_loss, _, _, stats = train_mim_heat(
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

def train_best_mim_model(study, dims, epochs, batch_size):
    # Retrieve the best trial's parameters
    best_params = study.best_trial.params
    
    # Extract MIM specific parameters for u and p
    n_u_hidden = best_params["n_u_hidden"]
    u_hidden_dims = [best_params[f"n_u_params_{i}"] for i in range(n_u_hidden)]
    n_p_hidden = best_params["n_p_hidden"]
    p_hidden_dims = [best_params[f"n_p_params_{i}"] for i in range(n_p_hidden)]
    
    u_dgm_dims = best_params["u_dgm_dims"]
    p_dgm_dims = best_params["p_dgm_dims"]
    
    mim_params = {
        "u_hidden_dims": u_hidden_dims,
        "p_hidden_dims": p_hidden_dims,
        "u_activation": best_params["u_activation"],
        "u_out_activation": best_params["u_output_activation"],
        "p_activation": best_params["p_activation"],
        "p_out_activation": best_params["p_output_activation"],
        "u_dgm_dims": u_dgm_dims,
        "u_n_dgm_layers": best_params["u_n_dgm_layers"] if u_dgm_dims > 0 else 0,
        "u_dgm_activation": best_params["u_dgm_activation"] if u_dgm_dims > 0 else "",
        "p_dgm_dims": p_dgm_dims,
        "p_n_dgm_layers": best_params["p_n_dgm_layers"] if p_dgm_dims > 0 else 0,
        "p_dgm_activation": best_params["p_dgm_activation"] if p_dgm_dims > 0 else "",
    }
    
    # Initialize model with best parameters
    model = HeatMIM(input_dims=dims, **mim_params).to(DEVICE)
    
    # Setup optimizer with the best learning rate
    lr = best_params["lr"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Set up data generator
    data_generator = HeatDataGenerator(
        d=dims,
        n_points=CONFIG["n_points"],
        base_seed=CONFIG["random_seed"]
    )
    
    # Configure scheduler if enabled in best trial
    use_scheduler = best_params["use_scheduler"]
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=best_params["scheduler_factor"],
            patience=best_params["scheduler_patience"],
            min_lr=1e-7
        )
    else:
        scheduler = None

    # Prepare training parameters
    train_params = {"batch_size": best_params["batch_size"]}
    if best_params["use_clip_grad"]:
        train_params["clip_grad"] = best_params["clip_grad"]
    
    # Proportion settings
    proportion_base = best_params["proportion_base"]
    proportion_bound = best_params["proportion_bound"]
    proportion_init = best_params["proportion_init"]

    # Normalize proportions
    total = proportion_base + proportion_bound + proportion_init
    train_params["bound_split"] = [
        proportion_base / total, 
        proportion_bound / total, 
        proportion_init / total
    ]
    
    # Define loss function with best lambda value
    loss_fn = HeatMIMLoss(best_params["lambda_1"])

    # Train the model
    best_loss, _, _, stats = train_mim_heat(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        data_generator=data_generator,
        epochs=epochs,
        train_params=train_params,
        scheduler=scheduler,
        verbose=False
    )
    
    print(f"Training completed with best loss: {best_loss}")
    return model, best_loss, stats

def train_best_mim_model(study, dims, epochs):
    # Retrieve the best trial's parameters
    best_params = study.best_trial.params
    
    # Extract MIM specific parameters for u and p
    n_u_hidden = best_params["n_u_hidden"]
    u_hidden_dims = [best_params[f"n_u_params_{i}"] for i in range(n_u_hidden)]
    n_p_hidden = best_params["n_p_hidden"]
    p_hidden_dims = [best_params[f"n_p_params_{i}"] for i in range(n_p_hidden)]
    
    u_dgm_dims = best_params["u_dgm_dims"]
    p_dgm_dims = best_params["p_dgm_dims"]
    
    mim_params = {
        "u_hidden_dims": u_hidden_dims,
        "p_hidden_dims": p_hidden_dims,
        "u_activation": best_params["u_activation"],
        "u_out_activation": best_params["u_output_activation"],
        "p_activation": best_params["p_activation"],
        "p_out_activation": best_params["p_output_activation"],
        "u_dgm_dims": u_dgm_dims,
        "u_n_dgm_layers": best_params["u_n_dgm_layers"] if u_dgm_dims > 0 else 0,
        "p_dgm_dims": p_dgm_dims,
        "p_n_dgm_layers": best_params["p_n_dgm_layers"] if p_dgm_dims > 0 else 0,
    }
    
    # Initialize model with best parameters
    model = HeatMIM(input_dims=dims, **mim_params).to(DEVICE)
    
    # Setup optimizer with the best learning rate
    lr = best_params["lr"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Set up data generator
    data_generator = HeatDataGenerator(
        d=dims,
        n_points=CONFIG["n_points"],
        base_seed=CONFIG["random_seed"]
    )
    
    # Configure scheduler if enabled in best trial
    use_scheduler = best_params["use_scheduler"]
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=best_params["scheduler_factor"],
            patience=best_params["scheduler_patience"],
            min_lr=1e-7
        )
    else:
        scheduler = None

    # Prepare training parameters
    train_params = {"batch_size": best_params["batch_size"]}
    if best_params["use_clip_grad"]:
        train_params["clip_grad"] = best_params["clip_grad"]
    

    # Normalize proportions
    train_params["bound_split"] = [
        1,0,0
    ]
    
    # Define loss function with best lambda value
    loss_fn = HeatMIMLoss(best_params["lambda_1"])

    # Train the model
    best_loss, _, _, stats = train_mim_heat(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        data_generator=data_generator,
        epochs=epochs,
        train_params=train_params,
        scheduler=scheduler,
        verbose=True
    )
    
    print(f"Training completed with best loss: {best_loss}")
    return model, best_loss, stats


def main():
    epochs = CONFIG["epochs"]
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
        dgm_study.optimize(lambda trial: dgm_heat_objective(trial, dims, epochs),
                           n_trials=trials-completed_trials, n_jobs=threads)

    for dims in study_dims:
        mim_study = optuna.create_study(
            direction="minimize", study_name=f"MIM Heat {dims}D",
            storage="sqlite:///mim_heat.db", load_if_exists=True)
        completed_trials = len(
            [s for s in mim_study.trials if s.state == TrialState.COMPLETE])
        mim_study.optimize(lambda trial: mim_heat_objective(trial, dims, epochs),
                           n_trials=trials-completed_trials, n_jobs=threads)


if __name__ == "__main__":
    main()
