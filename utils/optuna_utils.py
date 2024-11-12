def gen_dgm_params(trial):
    n_hidden = trial.suggest_int("n_hidden", 0, 5)
    hidden_dims = [trial.suggest_categorical(
        f"n_params_{i}", [4, 8, 16, 32, 64, 128, 256, 512]) for i in range(n_hidden)]
    dgm_dims = trial.suggest_categorical("dgm_dims", [0, 4, 8, 16, 32, 64, 128, 256])
    params = {
        "hidden_dims": hidden_dims,
        "dgm_dims": dgm_dims,
        "n_dgm_layers": trial.suggest_int("n_dgm_layers", 1, 5) if dgm_dims > 0 else 0,
        "hidden_activation": trial.suggest_categorical("hidden_activation", ["relu", "tanh", "sigmoid"]),
        "output_activation": trial.suggest_categorical("output_activation", ["relu", "tanh", "sigmoid", ""])
    }
    return params


def gen_mim_params(trial):
    n_u_hidden = trial.suggest_int("n_u_hidden", 1, 5)
    u_hidden_dims = [trial.suggest_categorical(
        f"n_u_params_{i}", [4, 8, 16, 32, 64, 128, 256, 512]) for i in range(n_u_hidden)]
    n_p_hidden = trial.suggest_int("n_p_hidden", 1, 5)
    p_hidden_dims = [trial.suggest_categorical(
        f"n_p_params_{i}", [4, 8, 16, 32, 64, 128, 256, 512]) for i in range(n_p_hidden)]
    u_dgm_dims = trial.suggest_categorical("u_dgm_dims", [0, 4, 8, 16, 32, 64, 128, 256])
    p_dgm_dims = trial.suggest_categorical("p_dgm_dims", [0, 4, 8, 16, 32, 64, 128, 256])
    params = {
        "u_hidden_dims": u_hidden_dims,
        "p_hidden_dims": p_hidden_dims,
        "u_activation": trial.suggest_categorical("u_activation", ["relu", "tanh", "sigmoid"]),
        "u_out_activation": trial.suggest_categorical("u_output_activation", ["relu", "tanh", "sigmoid", ""]),
        "p_activation": trial.suggest_categorical("p_activation", ["relu", "tanh", "sigmoid"]),
        "p_out_activation": trial.suggest_categorical("p_output_activation", ["relu", "tanh", "sigmoid", ""]),
        "u_dgm_dims": u_dgm_dims,
        "u_n_dgm_layers": trial.suggest_int("u_n_dgm_layers", 1, 5) if u_dgm_dims > 0 else 0,
        "p_dgm_dims": p_dgm_dims,
        "p_n_dgm_layers": trial.suggest_int("p_n_dgm_layers", 1, 5) if p_dgm_dims > 0 else 0,
    }
    return params
