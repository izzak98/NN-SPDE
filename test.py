import torch
import numpy as np
from models import DGM
from utils.noise_utils import shifted_gaussian_white_noise, time_dependant_gaussian_white_noise
from heat_train import HeatTrainLEC, train_heat


def test():
    model = DGM(
        spatial_dims=2,
        add_dims=4,
        hidden_dims=[128, 128, 64],
        dgm_dims=32,
        n_dgm_layers=4,
        hidden_activation='tanh',
        output_activation='sigmoid',
        name='test_model',
    )

    def inital_condition(coords):
        x = torch.cat(coords, dim=1)
        return torch.prod(torch.sin(torch.pi * x), dim=1, keepdim=True)

    def boundary_condition(coords):
        return torch.zeros_like(coords[0])

    forcing_term = inital_condition

    def boundary_conversion_func(coord):
        # Convert to boundary by rounding to 0 or 1 but keep the same shape and dtype
        return torch.round(coord)

    def E_func(t, mu):
        # t shape (batch_size, 1)
        # mu shape (batch_size, 1)
        return t * mu

    def analytical_solution_constant_case(t, nu, noise_args, coords):  # -> Any:
        # t shape (batch_size, 1)
        # nu shape (batch_size, 1)
        # noise_args list of tensors but first one is the mu (batch_size, 1)
        # coords list of tensors of length dims each of shape (batch_size, 1)

        x = torch.cat(coords, dim=1)
        d = x.shape[-1]
        pi = torch.pi
        lambda_ = nu * d * pi**2  # ν d π²
        S0 = noise_args[0]

        # Compute φ(x) = ∏ sin(π x_i)
        phi_x = torch.prod(torch.sin(pi * x), keepdim=True, dim=1)

        # Compute time-dependent scalar part
        exp_term = torch.exp(-lambda_ * t)
        A_t = 1 * exp_term + (S0 / lambda_) * (1 - exp_term)

        # Return u(t, x) = A(t) * φ(x), broadcasting over space
        return (A_t * phi_x)  # squeeze to remove extra dimensions if t was scalar

    def analytical_solution_time_dependent_case(t, nu, noise_args, coords, n_steps=100):
        x = torch.cat(coords, dim=1)                         # (batch_size, d)
        d = x.shape[-1]
        pi = torch.pi
        lambda_ = nu * d * pi**2                             # (batch_size, 1)

        # Compute φ(x) = ∏ sin(π x_i)
        phi_x = torch.prod(torch.sin(pi * x), dim=1, keepdim=True)  # (batch_size, 1)

        # Time-dependent part via numerical integration
        batch_size = t.shape[0]

        # Create integration points s in [0, t]
        s_grid = torch.linspace(0, 1, n_steps, device=t.device).reshape(1, -1)  # shape (1, n_steps)
        s_grid = s_grid * t  # shape (batch_size, n_steps)

        # Evaluate S(s) = E(s, *noise_args)
        s_flat = s_grid.reshape(-1, 1)  # (batch_size * n_steps, 1)
        sigma_expanded = noise_args[0].repeat_interleave(
            n_steps, dim=0)  # (batch_size * n_steps, 1)
        S_s = E_func(s_flat, sigma_expanded).reshape(
            batch_size, n_steps)      # (batch_size, n_steps)

        # Compute integrand: exp(-λ(t - s)) * S(s)
        lambda_expanded = lambda_.expand(-1, n_steps)                     # (batch_size, n_steps)
        t_expanded = t.expand(-1, n_steps)
        integrand = torch.exp(-lambda_expanded * (t_expanded - s_grid)) * S_s

        # Integrate using trapezoidal or rectangle rule
        dt = t / n_steps  # shape (batch_size, 1)
        integral = torch.sum(integrand, dim=1, keepdim=True) * \
            dt  # Riemann sum, shape (batch_size, 1)

        # A(t) = α e^{-λ t} + integral
        A_t = torch.exp(-lambda_ * t) + integral  # assuming α = 1

        # Final solution: u(t, x) = A(t) * φ(x)
        return A_t * phi_x

    analytic_solution = analytical_solution_time_dependent_case

    noise_function = time_dependant_gaussian_white_noise

    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_helper = HeatTrainLEC(
        model=model,
        inital_condition=inital_condition,
        boundary_condition=boundary_condition,
        forcing_term=forcing_term,
        boundary_conversion_func=boundary_conversion_func,
        lambda1=100.0,
        lambda2=100.0,
        stochastic=True,
    )
    train_heat(
        model=model,
        optimizer=optimizer,
        analytic_solution=analytic_solution,
        noise_func=noise_function,
        loss_helper=loss_helper,
        n_dims=2,
        n_train_points=1000,
        verbose=True,
        name="test_model-t-dependant",
        m_samples=5,
    )


def analytical_solution_constant_case(t, nu, noise_args, coords):  # -> Any:
    # t shape (batch_size, 1)
    # nu shape (batch_size, 1)
    # noise_args list of tensors but first one is the mu (batch_size, 1)
    # coords list of tensors of length dims each of shape (batch_size, 1)

    x = torch.cat(coords, dim=1)
    d = x.shape[-1]
    pi = torch.pi
    lambda_ = nu * d * pi**2  # ν d π²
    S0 = noise_args[0]

    # Compute φ(x) = ∏ sin(π x_i)
    phi_x = torch.prod(torch.sin(pi * x), keepdim=True, dim=1)

    # Compute time-dependent scalar part
    exp_term = torch.exp(-lambda_ * t)
    A_t = 1 * exp_term + (S0 / lambda_) * (1 - exp_term)

    # Return u(t, x) = A(t) * φ(x), broadcasting over space
    return (A_t * phi_x)  # squeeze to remove extra dimensions if t was scalar


def sanity_check():
    points = 10000
    domain = (0, 1)
    noise_dims = [[0, 1], [0, 1]]
    n_dims = 2
    val_t = torch.rand(points, 1)
    val_coords = [torch.rand(points, 1) * (domain[1] -
                                           domain[0]) + domain[0] for _ in range(n_dims)]
    val_noise_args = [torch.rand(points, 1) for _ in range(len(noise_dims))]
    val_nu = torch.rand(points, 1)

    no_stocastic_case = val_noise_args.copy()
    no_stocastic_case[0] = torch.zeros_like(val_noise_args[0]) + 1
    val_solution = analytical_solution_constant_case(val_t, val_nu, val_noise_args, val_coords)
    no_stoc_solution = analytical_solution_constant_case(
        val_t, val_nu, no_stocastic_case, val_coords)
    rmse = torch.sqrt(torch.mean((val_solution - no_stoc_solution)**2))
    print(f"RMSE: {rmse.item()}")


if __name__ == "__main__":
    test()
    # sanity_check()
