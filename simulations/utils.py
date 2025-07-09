# === Imports ===
import re
import time
import torch
import torchsde
import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
from torch import nn
from torchdiffeq import odeint
from include import neural_net


# === SDE Solvers ===

def EulerMaruyamaMA(*,
                    S: torch.Tensor,
                    y0_EGFR: torch.Tensor,
                    J_f: torch.Tensor,
                    y_ss_flux: torch.Tensor,
                    y_steady: torch.Tensor,
                    num_steps: int,
                    final_t: float,
                    epsilon: torch.Tensor = torch.tensor(0.0),
                    dt: float = 1e-5
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    Euler-Maruyama integration of a linearised chemical SDE system.

    Input:
        S (torch.Tensor): Stoichiometry matrix.
        y0_EGFR (torch.Tensor): Initial state.
        J_f (torch.Tensor): Jacobian of flux function.
        y_ss_flux (torch.Tensor): Steady-state flux.
        y_steady (torch.Tensor): Steady-state concentrations.
        num_steps (int): Number of time steps.
        final_t (float): Final time point.
        epsilon (torch.Tensor): Diffusion scaling factor.
        dt (float): Time step size.

    Return:
        data_EGFR_linear (torch.Tensor): Simulated trajectory.
        ts (torch.Tensor): Time points.
        elapsed_time (float): Simulation runtime in seconds.
    """
    model = GenericChemSDELinearized(S, J_f, y_ss_flux[0], y_steady, epsilon=epsilon)
    ts = torch.linspace(0, final_t, num_steps)
    start = time.time()
    data = torchsde.sdeint(model, y0_EGFR, ts, method='euler', dt=dt, adaptive=False)
    elapsed_time = time.time() - start
    return data, ts, elapsed_time


def EulerMaruyamaMat(*,
                     cfg: dict,
                     num_steps: int,
                     init_state: torch.Tensor,
                     dt: float = 1.0
) -> torch.Tensor:
    """
    Vectorised Euler-Maruyama solver for a linear SDE.

    Input:
        cfg (dict): Dictionary of A-matrix entries and sigma.
        num_steps (int): Number of steps.
        init_state (torch.Tensor): Initial condition.
        dt (float): Time step.

    Return:
        data (torch.Tensor): Simulated trajectory over time.
    """
    data = [init_state]
    dim = int(np.sqrt(len(cfg) - 1))
    K = torch.stack([torch.stack([cfg[f'K_{i},{j}'] for j in range(dim)]) for i in range(dim)])
    sigma = cfg["sigma"]

    for _ in range(num_steps):
        noise = sigma * torch.normal(0.0, 1.0, size=data[-1].shape) * np.sqrt(dt)
        drift = torch.matmul(data[-1], K.T) * dt
        step = data[-1] + drift + noise
        data.append(torch.clip(step, -50, 50))

    return torch.stack(data)


# === Flux & Jacobian ===

def compute_flux_jacobian(flux_function: callable,
                          S: torch.Tensor,
                          flux_expressions: list[str],
                          variables: list[str],
                          params: dict,
                          y_steady: torch.Tensor
) -> torch.Tensor:
    """
    Compute Jacobian of the flux function at steady state.

    Input:
        flux_function (callable): Symbolic flux function.
        S (torch.Tensor): Stoichiometry matrix.
        flux_expressions (list[str]): Symbolic expressions for flux.
        variables (list[str]): Names of variables.
        params (dict): Parameter dictionary.
        y_steady (torch.Tensor): Steady state vector.

    Return:
        J_f (torch.Tensor): Flux Jacobian.
    """
    y_steady = y_steady.clone().detach().requires_grad_(True)

    def flux_wrapper(y):
        return flux_function(flux_expressions, variables, params, y) @ S.T

    J_f = torch.autograd.functional.jacobian(flux_wrapper, y_steady, create_graph=True)
    return J_f.squeeze()

def compute_flux(flux_expressions: list[str],
                 variables: list[str],
                 params: dict,
                 y: torch.Tensor
) -> torch.Tensor:
    """
    Evaluate the symbolic flux expressions.

    Input:
        flux_expressions (list[str]): Expressions as strings.
        variables (list[str]): Variable names.
        params (dict): Parameter dictionary.
        y (torch.Tensor): Current state.

    Return:
        flux_values (torch.Tensor): Flux evaluated at y.
    """
    flux_values = []
    for expr in flux_expressions:
        expr_sub = expr
        for p_name, p_value in params.items():
            expr_sub = re.sub(rf'\b{p_name}\b', f'params["{p_name}"]', expr_sub)
        for idx, var_name in enumerate(variables):
            expr_sub = re.sub(rf'\b{var_name}\b', f'y[...,{idx}]', expr_sub)
        flux_values.append(eval(expr_sub, {"params": params}, {"y": y}))
    return torch.stack(flux_values, dim=-1)


# === Auxiliary variable implementation ===

def construct_Ks(pp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Construct submatrices K_ss and K_sb from a full reaction rate matrix.

    Input:
        pp (torch.Tensor): 1D tensor of parameter values (requires gradients).
                           Assumed shape: (n_params,) and device-aware.

    Return:
        K_ss (torch.Tensor): Submatrix for observed variables (n_s × n_s).
        K_sb (torch.Tensor): Cross-submatrix coupling observed and hidden variables (n_s × n_b).
        K_bs (torch.Tensor): Cross-submatrix coupling observed and hidden variables (n_b × n_s).
        K_bb (torch.Tensor): Submatrix for hidden variables (n_b × n_b).
    """
    device = pp.device  # ensure all constants live on same device
    zero = lambda: pp[0] * 0  # gradient-safe zero
    one = lambda: pp[0] / pp[0]  # gradient-safe one
    
    K_full = torch.stack([
        torch.stack([-1.0*pp[0]*ss_xEGF, -1.0*pp[0]*ss_xR, 1.0*pp[1], zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero()]),
        torch.stack([-1.0*pp[0]*ss_xEGF, -1.0*pp[0]*ss_xR, 1.0*pp[1], zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero()]),
        torch.stack([1.0*pp[0]*ss_xEGF, 1.0*pp[0]*ss_xR, -4.0*pp[2]*ss_xRa - 1.0*pp[1], 2.0*pp[3], zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero()]),
        torch.stack([zero(), zero(), 2.0*pp[2]*ss_xRa, -1.0*pp[4] - 1.0*pp[3], 1.0*pp[5], zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero()]),
        torch.stack([zero(), zero(), zero(), 1.0*pp[4], pp[44]*ss_xRP/(ss_xRP + K4)**2 - pp[44]/(ss_xRP + K4), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero()]),
        torch.stack([zero(), zero(), zero(), zero(), -1.0*pp[6]*ss_xPLCg, -1.0*pp[6]*ss_xRP, 1.0*pp[7], zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero()]),
        torch.stack([zero(), zero(), zero(), zero(), 1.0*pp[6]*ss_xPLCg, 1.0*pp[6]*ss_xRP, -1.0*pp[8] - 1.0*pp[7], 1.0*pp[9], zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero()]),
        torch.stack([zero(), zero(), zero(), zero(), 1.0*pp[11]*ss_xPLCgP, zero(), 1.0*pp[8], -1.0*pp[10] - 1.0*pp[9], 1.0*pp[11]*ss_xRP, zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero()]),
        torch.stack([zero(), zero(), zero(), zero(), -1.0*pp[11]*ss_xPLCgP, zero(), zero(), 1.0*pp[10], -1.0*pp[42] - 1.0*pp[11]*ss_xRP + pp[45]*ss_xPLCgP/(ss_xPLCgP + K8)**2 - pp[45]/(ss_xPLCgP + K8), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), 1.0*pp[43]]),
        torch.stack([zero(), zero(), zero(), zero(), -1.0*pp[12]*ss_xGrb, zero(), zero(), zero(), zero(), -1.0*pp[26]*ss_xRShP - 1.0*pp[34]*ss_xShP - 1.0*pp[12]*ss_xRP - 1.0*pp[19]*ss_xSOS, 1.0*pp[13], -1.0*pp[19]*ss_xGrb, zero(), 1.0*pp[18], zero(), zero(), -1.0*pp[26]*ss_xGrb, -1.0*pp[34]*ss_xGrb, 1.0*pp[27], 1.0*pp[35], zero(), zero(), zero()]),
        torch.stack([zero(), zero(), zero(), zero(), 1.0*pp[12]*ss_xGrb, zero(), zero(), zero(), zero(), 1.0*pp[12]*ss_xRP, -1.0*pp[14]*ss_xSOS - 1.0*pp[13], -1.0*pp[14]*ss_xRG, 1.0*pp[15], zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero()]),
        torch.stack([zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), -1.0*pp[19]*ss_xSOS, -1.0*pp[14]*ss_xSOS, -1.0*pp[14]*ss_xRG - 1.0*pp[30]*ss_xRShG - 1.0*pp[36]*ss_xShG - 1.0*pp[19]*ss_xGrb, 1.0*pp[15], 1.0*pp[18], zero(), zero(), zero(), zero(), -1.0*pp[30]*ss_xSOS, -1.0*pp[36]*ss_xSOS, 1.0*pp[31], 1.0*pp[37], zero()]),
        torch.stack([zero(), zero(), zero(), zero(), 1.0*pp[17]*ss_xGS, zero(), zero(), zero(), zero(), zero(), 1.0*pp[14]*ss_xSOS, 1.0*pp[14]*ss_xRG, -1.0*pp[16] - 1.0*pp[15], 1.0*pp[17]*ss_xRP, zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero()]),
        torch.stack([zero(), zero(), zero(), zero(), -1.0*pp[17]*ss_xGS, zero(), zero(), zero(), zero(), 1.0*pp[19]*ss_xSOS, zero(), 1.0*pp[19]*ss_xGrb, 1.0*pp[16], -1.0*pp[18] - 1.0*pp[40]*ss_xRShP - 1.0*pp[17]*ss_xRP - 1.0*pp[39]*ss_xShP, zero(), zero(), -1.0*pp[40]*ss_xGS, -1.0*pp[39]*ss_xGS, zero(), zero(), 1.0*pp[41], 1.0*pp[38], zero()]),
        torch.stack([zero(), zero(), zero(), zero(), -1.0*pp[20]*ss_xShc, zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), -1.0*pp[20]*ss_xRP, 1.0*pp[21], zero(), zero(), zero(), zero(), zero(), zero(), zero()]),
        torch.stack([zero(), zero(), zero(), zero(), 1.0*pp[20]*ss_xShc, zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), 1.0*pp[20]*ss_xRP, -1.0*pp[22] - 1.0*pp[21], 1.0*pp[23], zero(), zero(), zero(), zero(), zero(), zero()]),
        torch.stack([zero(), zero(), zero(), zero(), 1.0*pp[25]*ss_xShP, zero(), zero(), zero(), zero(), -1.0*pp[26]*ss_xRShP, zero(), zero(), zero(), -1.0*pp[40]*ss_xRShP, zero(), 1.0*pp[22], -1.0*pp[24] - 1.0*pp[26]*ss_xGrb - 1.0*pp[40]*ss_xGS - 1.0*pp[23], 1.0*pp[25]*ss_xRP, 1.0*pp[27], zero(), 1.0*pp[41], zero(), zero()]),
        torch.stack([zero(), zero(), zero(), zero(), -1.0*pp[25]*ss_xShP, zero(), zero(), zero(), zero(), -1.0*pp[34]*ss_xShP, zero(), zero(), zero(), -1.0*pp[39]*ss_xShP, zero(), zero(), 1.0*pp[24], -1.0*pp[34]*ss_xGrb - 1.0*pp[25]*ss_xRP - 1.0*pp[39]*ss_xGS + pp[46]*ss_xShP/(ss_xShP + K16)**2 - pp[46]/(ss_xShP + K16), zero(), 1.0*pp[35], zero(), 1.0*pp[38], zero()]),
        torch.stack([zero(), zero(), zero(), zero(), 1.0*pp[29]*ss_xShG, zero(), zero(), zero(), zero(), 1.0*pp[26]*ss_xRShP, zero(), -1.0*pp[30]*ss_xRShG, zero(), zero(), zero(), zero(), 1.0*pp[26]*ss_xGrb, zero(), -1.0*pp[28] - 1.0*pp[30]*ss_xSOS - 1.0*pp[27], 1.0*pp[29]*ss_xRP, 1.0*pp[31], zero(), zero()]),
        torch.stack([zero(), zero(), zero(), zero(), -1.0*pp[29]*ss_xShG, zero(), zero(), zero(), zero(), 1.0*pp[34]*ss_xShP, zero(), -1.0*pp[36]*ss_xShG, zero(), zero(), zero(), zero(), zero(), 1.0*pp[34]*ss_xGrb, 1.0*pp[28], -1.0*pp[36]*ss_xSOS - 1.0*pp[29]*ss_xRP - 1.0*pp[35], zero(), 1.0*pp[37], zero()]),
        torch.stack([zero(), zero(), zero(), zero(), 1.0*pp[33]*ss_xShGS, zero(), zero(), zero(), zero(), zero(), zero(), 1.0*pp[30]*ss_xRShG, zero(), 1.0*pp[40]*ss_xRShP, zero(), zero(), 1.0*pp[40]*ss_xGS, zero(), 1.0*pp[30]*ss_xSOS, zero(), -1.0*pp[32] - 1.0*pp[31] - 1.0*pp[41], 1.0*pp[33]*ss_xRP, zero()]),
        torch.stack([zero(), zero(), zero(), zero(), -1.0*pp[33]*ss_xShGS, zero(), zero(), zero(), zero(), zero(), zero(), 1.0*pp[36]*ss_xShG, zero(), 1.0*pp[39]*ss_xShP, zero(), zero(), zero(), 1.0*pp[39]*ss_xGS, zero(), 1.0*pp[36]*ss_xSOS, 1.0*pp[32], -1.0*pp[38] - 1.0*pp[33]*ss_xRP - 1.0*pp[37], zero()]),
        torch.stack([zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), 1.0*pp[42], zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), -1.0*pp[43]]),
    ])

    # Assume K_full is a 2D tensor of shape [n, n]
    n = K_full.shape[0]
    
    # Select rows and columns
    K_ss = K_full.index_select(0, idx).index_select(1, idx)
    K_bb = K_full.index_select(0, idx_b).index_select(1, idx_b)
    K_sb = K_full.index_select(0, idx).index_select(1, idx_b)
    K_bs = K_full.index_select(0, idx_b).index_select(1, idx)
    
    return K_ss, K_sb, K_bs, K_bb

def traj_gen(x_t: torch.Tensor,
             z_t: torch.Tensor,
             Kss: torch.Tensor,
             Ksb: torch.Tensor,
             ts: torch.Tensor,
             state: torch.Tensor
) -> torch.Tensor:
    """
    Simulate observed trajectory from linearised system with hidden variables.

    Input:
        x_t (torch.Tensor): Initial state of observed species (n_s,).
        z_t (torch.Tensor): Initial state of hidden species (n_b,).
        Kss (torch.Tensor): Linear operator on observed species (n_s × n_s).
        Ksb (torch.Tensor): Coupling matrix from hidden to observed (n_s × n_b).
        ts (torch.Tensor): Time points (T,).
        state (torch.Tensor): Initial full system state [x_t; z_t] (n_s + n_b,).

    Return:
        trajectory (torch.Tensor): Simulated trajectory of observed species.
                                   Shape: (T', n_s), subsampled every 1000 steps.
    """
    # Store trajectories
    trajectory = []
    z = z_t
    for t in ts:
        noise = noise_level * torch.randn(n_s)
    
        dxs_dt = Kss @ x_t + Ksb @ z + noise
    
        dstate_dt = torch.cat((dxs_dt, dz_dt))
        state = state + dstate_dt * dt  # Euler integration
        z = state[n_s:]
    
        trajectory.append(state.unsqueeze(0))

    trajectory = torch.cat(trajectory)
    return trajectory[::1000,:n_s]
    

# === SDE & ODE Class Definitions ===

class GenericChemSDELinearized(nn.Module):
    """
    Linearised chemical reaction SDE class.

    Input:
        s_matrix (torch.Tensor): Stoichiometry matrix.
        flux_jacobian (torch.Tensor): Jacobian at steady state.
        steady_state_flux (torch.Tensor): f(y*).
        steady_state_y (torch.Tensor): y*.
        epsilon (float): Noise scaling factor.
    """
    def __init__(self, s_matrix, flux_jacobian, steady_state_flux, steady_state_y, epsilon=1.0):
        super().__init__()
        self.epsilon = epsilon
        self.noise_type = "general"
        self.sde_type = "ito"
        self.S = s_matrix
        self.J_f = flux_jacobian
        self.f_y_star = steady_state_flux
        self.y_star = steady_state_y
        
    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Input:
            t (torch.Tensor): Time.
            y (torch.Tensor): State.
        Return:
            drift (torch.Tensor): Drift term.
        """
        return (self.J_f @ y.T).T

    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Input:
            t (torch.Tensor): Time.
            y (torch.Tensor): State.
        Return:
            diffusion (torch.Tensor): Diffusion term.
        """
        f_vals_sqrt = torch.sqrt(torch.abs(self.f_y_star))
        diag_f = torch.diag(f_vals_sqrt.squeeze(0))
        S_diagf = diag_f @ self.S.T
        return self.epsilon * S_diagf.T.unsqueeze(0)


class GenericODELinearised(nn.Module):
    """
    Linearised ODE class from Jacobian.

    Input:
        flux_jacobian (torch.Tensor): Jacobian matrix.
    """
    def __init__(self, flux_jacobian):
        super().__init__()
        self.J_f = flux_jacobian

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Input:
            t (torch.Tensor): Time.
            y (torch.Tensor): State.
        Return:
            drift (torch.Tensor): Time derivative.
        """
        return (self.J_f @ y.T).T


# === Utility Functions ===

def make_cfg(K: torch.Tensor, sigma: float) -> dict:
    """
    Convert matrix K and noise level into config dict.

    Input:
        K (torch.Tensor): Coefficient matrix.
        sigma (float): Noise magnitude.

    Return:
        cfg (dict): Dictionary of parameters and sigma.
    """
    return {f'K_{i},{j}': K[i][j] for i in range(len(K)) for j in range(len(K[i]))} | {"sigma": torch.tensor(sigma)}


def make_K(cfg: dict) -> torch.Tensor:
    """
    Reconstruct matrix K from cfg dictionary.

    Input:
        cfg (dict): Contains K_ij keys.

    Return:
        K (torch.Tensor): Reconstructed matrix.
    """
    dim = int(np.sqrt(len(cfg) - 1))
    return torch.as_tensor([[cfg[f'K_{i}{j}'] for j in range(dim)] for i in range(dim)])


def weighted_mse(y_true, y_pred):
    """
    Weighted MSE with weights = 1 / |y_true|.

    Returns
    -------
    float
        Weighted error.
    """
    weights = 1 / np.abs(y_true)
    return np.sum(weights * (y_true - y_pred) ** 2) / np.sum(weights)


# === Plotting ===

def plot(ts: torch.Tensor,
         samples: torch.Tensor,
         xlabel: str,
         ylabel: str,
         marker: bool = False,
         title: str = '',
         var_names: list[str] = '',
         color_scheme: str = 'tab10') -> None:
    """
    Plot time series from simulated trajectories.

    Input:
        ts (torch.Tensor): Time points.
        samples (torch.Tensor): Shape [T, B, D].
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        marker (bool): Use markers.
        title (str): Plot title.
        var_names (list[str] or str): Names of variables.
        color_scheme (str): Seaborn palette name.

    Return:
        None
    """
    ts = ts.cpu()
    if samples.dim() < 3:
        samples = samples.unsqueeze(-1)

    t_size, batch_size, num_vars = samples.shape
    var_names = [f'Var {i}' for i in range(num_vars)] if var_names == '' else var_names
    palette = sns.color_palette(color_scheme, num_vars)
    color_dict = dict(zip(var_names, palette))

    plt.figure(figsize=(12, 6))
    for b in range(batch_size):
        for v in range(num_vars):
            plt.plot(ts, samples[:, b, v].detach().numpy(), marker='x' if marker else None,
                     label=var_names[v], color=color_dict[var_names[v]])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(fontsize=8)
    plt.show()

def plot_loss(losses: list[float], title: str, yscale: str = 'log') -> None:
    """
    Plot training loss over epochs.

    Input:
        losses (list[float]): Loss values.
        title (str): Plot title.
        yscale (str): Y-axis scale ('log' or 'linear').

    Return:
        None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(losses)), losses)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.yscale(yscale)
    plt.show()

def plot_time_series(true_data: torch.Tensor,
                     predicted_time_series: torch.Tensor,
                     variables: list[str],
                     color_scheme: str = 'viridis') -> None:
    """
    Compare true vs predicted time series.

    Input:
        true_data (torch.Tensor): True values.
        predicted_time_series (torch.Tensor): Model predictions.
        variables (list[str]): Variable names.
        color_scheme (str): Seaborn palette name.

    Return:
        None
    """
    true_data_xr = xr.DataArray(true_data.squeeze(), dims=["time", "variable"],
                                 coords={"time": np.arange(true_data.shape[0]), "variable": variables})
    pred_data_xr = xr.DataArray(predicted_time_series.detach().squeeze(), dims=["time", "variable"],
                                 coords={"time": np.arange(predicted_time_series.shape[0]), "variable": variables})

    palette = sns.color_palette(color_scheme, len(variables))
    color_dict = dict(zip(variables, palette))

    plt.figure(figsize=(12, 6))
    for var in variables:
        plt.plot(true_data_xr['time'], true_data_xr.sel(variable=var), label=var, color=color_dict[var])
        plt.plot(pred_data_xr['time'], pred_data_xr.sel(variable=var), linestyle='--', color=color_dict[var])
    plt.legend(fontsize=8)
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Time Series Comparison')
    plt.show()

def plot_parameter_preds(params: list[str],
                         cfg: dict,
                         parameters_ts: list[torch.Tensor]) -> None:
    """
    Plot learned parameters vs true values.

    Input:
        params (list[str]): Parameter names.
        cfg (dict): True parameters.
        parameters_ts (list[torch.Tensor]): Learned parameter tensors over epochs.

    Return:
        None
    """
    true_values = {key: float(value) for key, value in cfg.items() if key in params}
    parameters_ts_xr = xr.DataArray(
        data=torch.stack(parameters_ts).cpu().numpy(),
        dims=["epoch", "parameter"],
        coords={"epoch": np.arange(len(parameters_ts)), "parameter": list(true_values.keys())}
    )

    num_params = len(true_values)
    ncols, nrows = 6, (num_params + 5) // 6
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2.5), sharex=True)
    axes = axes.flatten()

    for i, param in enumerate(true_values):
        parameters_ts_xr.sel(parameter=param).plot.line(ax=axes[i], color="tab:blue")
        axes[i].set_title(f"{param} over epochs")
        axes[i].legend(fontsize=6)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()
