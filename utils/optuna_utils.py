def gen_dgm_params(trial):
    n_hidden = trial.suggest_int("n_hidden", 1, 3)
    hidden_dims = [trial.suggest_categorical(
        f"n_params_{i}", 4, 8, 16, 32, 64, 128, 256, 512) for i in range(n_hidden)]
    params = {
        "hidden_dims": hidden_dims,
        "activation": trial.suggest_categorical("activation", ["relu", "tanh", "sigmoid", ""]),
        "output_activation": trial.suggest_categorical("output_activation", ["relu", "tanh", "sigmoid", ""])
    }
    return params

def gen_mim_params(trial):
    n_u_hidden = trial.suggest_int("n_u_hidden", 1, 3)
    u_hidden_dims = [trial.suggest_categorical(
        f"n_u_params_{i}", 4, 8, 16, 32, 64, 128, 256, 512) for i in range(n_u_hidden)]
    n_p_hidden = trial.suggest_int("n_p_hidden", 1, 3)
    p_hidden_dims = [trial.suggest_categorical(
        f"n_p_params_{i}", 4, 8, 16, 32, 64, 128, 256, 512) for i in range(n_p_hidden)]
    params = {
        "u_hidden_dims": u_hidden_dims,
        "p_hidden_dims": p_hidden_dims,
        "u_activation": trial.suggest_categorical("u_activation", ["relu", "tanh", "sigmoid", ""]),
        "u_output_activation": trial.suggest_categorical("u_output_activation", ["relu", "tanh", "sigmoid", ""]),
        "p_activation": trial.suggest_categorical("p_activation", ["relu", "tanh", "sigmoid", ""]),
        "p_output_activation": trial.suggest_categorical("p_output_activation", ["relu", "tanh", "sigmoid", ""]),
    }
    return params




