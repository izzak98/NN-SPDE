def get_model_hyper_params(trial, sub_model_name):
    n_fc_layers = trial.suggest_int(f"{sub_model_name}_n_fc_layers", 1, 5)
    hidden_dims = []
    for i in range(n_fc_layers):
        hidden_dims.append(trial.suggest_categorical(
            f"{sub_model_name}_fc_layer_{i}_dim", [32, 64, 128, 256, 512]))
    use_dgm = trial.suggest_categorical(f"{sub_model_name}_use_dgm", [1, 0])
    use_dgm = bool(use_dgm)
    if use_dgm:
        dgm_dims = trial.suggest_categorical(
            f"{sub_model_name}_dgm_dims", [2, 4, 8, 16, 32, 64, 128])
        n_dgm_layers = trial.suggest_int(f"{sub_model_name}_n_dgm_layers", 1, 5)
    else:
        dgm_dims = 0
        n_dgm_layers = 0
    output_activation = trial.suggest_categorical(
        f"{sub_model_name}_output_activation", ["relu", "tanh", "sigmoid", None])
    hidden_activation = trial.suggest_categorical(
        f"{sub_model_name}_hidden_activation", ["relu", "tanh", "sigmoid"])
    model_params = {
        "hidden_dims": hidden_dims,
        "dgm_dims": dgm_dims,
        "n_dgm_layers": n_dgm_layers,
        "output_activation": output_activation,
        "hidden_activation": hidden_activation
    }
    return model_params


def get_best_model_hyper_params(best_trial, sub_model_name):
    """Extracts best hyperparameters for the given sub-model from an Optuna best trial."""
    n_fc_layers = best_trial.params[f"{sub_model_name}_n_fc_layers"]
    hidden_dims = [
        best_trial.params[f"{sub_model_name}_fc_layer_{i}_dim"] for i in range(n_fc_layers)
    ]

    use_dgm = bool(best_trial.params[f"{sub_model_name}_use_dgm"])
    if use_dgm:
        dgm_dims = best_trial.params[f"{sub_model_name}_dgm_dims"]
        n_dgm_layers = best_trial.params[f"{sub_model_name}_n_dgm_layers"]
    else:
        dgm_dims = 0
        n_dgm_layers = 0

    model_params = {
        "hidden_dims": hidden_dims,
        "dgm_dims": dgm_dims,
        "n_dgm_layers": n_dgm_layers,
        "output_activation": best_trial.params[f"{sub_model_name}_output_activation"],
        "hidden_activation": best_trial.params[f"{sub_model_name}_hidden_activation"],
    }
    return model_params
