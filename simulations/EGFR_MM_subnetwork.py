# --- Imports and Path Setup ---
import os
import sys
import time
import re
import torch
import torchsde
import numpy as np
import pandas as pd

from torch import nn
from itertools import zip_longest
from torchdiffeq import odeint

# Project-specific imports
from include import neural_net
from utils import (
    compute_flux_jacobian, compute_flux,
    EulerMaruyamaMA, EulerMaruyamaMat,
    make_cfg, make_K,
    GenericChemSDELinearised, GenericODELinearised,
    plot, plot_loss, plot_time_series,
    plot_parameter_preds, weighted_mse
)

# --- Add parent directory to module path ---
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)


# --- Data and Parameter Setup ---
ind_2 = list(range(4,8)) + list(range(10,20)) + [22]  # Selected boundary parameters

# --- Load initial conditions ---
df = pd.read_csv("ics_EGFR_MM_15_20_14_18_21_16_19_17.csv", header=None)
ics_state = torch.tensor(df.to_numpy(dtype=float), dtype=torch.float32).squeeze()

# Full-state initial conditions
ics_state_full = torch.tensor(pd.read_csv("ics_EGFR.csv", header=None).to_numpy(dtype=float), dtype=torch.float32).squeeze()

# --- Load true parameters ---
df = pd.read_csv("params_EGFR_MM_15_20_14_18_21_16_19_17.csv")
param_names = list(df.keys())
true_params = torch.tensor(df.to_numpy(dtype=float)[0], dtype=torch.float32)
params = dict(zip(param_names, true_params))

# --- Load system matrices ---
S = torch.tensor(pd.read_csv("S_EGFR_MM_15_20_14_18_21_16_19_17.csv", header=None).to_numpy(dtype=float), dtype=torch.float32)
f = pd.read_csv("f_EGFR_MM_15_20_14_18_21_16_19_17.csv", header=None).values.T[0].tolist()
x = pd.read_csv("x_EGFR_MM_15_20_14_18_21_16_19_17.csv", header=None).values.T[0].tolist()
x_full = pd.read_csv("x_EGFR.csv", header=None).values.T[0].tolist()
ss_x = torch.tensor(pd.read_csv("steady_states_EGFR_MM_15_20_14_18_21_16_19_17.csv", header=None).values.T[0].tolist())
ss_x_full = torch.tensor(pd.read_csv("steady_states_EGFR.csv", header=None).values.T[0].tolist())

# --- Load data and time grid ---
data_linear = torch.tensor(pd.read_csv('data_linear_MM.csv').values, dtype=torch.float32)[:,[i for i in range(23) if i not in [15,20,14,18,21,16,19,17]]]
n_steps = data_linear.shape[0]
ts = torch.linspace(0, 3, n_steps)

# --- Linear SDE Setup ---
init_state = torch.tensor(ss_x * ics_state).reshape(1,-1)
x_ss = ss_x.reshape(1, 1, -1)
x_ss_flux = compute_flux(f, x, params, x_ss)
J_f = compute_flux_jacobian(compute_flux, S, f, x, params, x_ss)
flux_jacobian = J_f
linear_sde = GenericChemSDELinearised(S, flux_jacobian, x_ss_flux[0], x_ss, epsilon=0.1)


# --- Define true steady-state concentrations ---
# Subset of all species, manually extracted
ss_xR = torch.tensor(0.5995697475164046)
ss_xEGF = torch.tensor(580.5995697402507)
ss_xRa = torch.tensor(17.405496871864763)
ss_xR2 = torch.tensor(30.29513214)
ss_xRP = torch.tensor(3.604495797340412)
ss_xPLCg = torch.tensor(0.178523494483829)
ss_xRPL = torch.tensor(0.045414074)
ss_xRPLP = torch.tensor(0.3177531558421101)
ss_xPLCgP = torch.tensor(3.042475015361836)
ss_xGrb = torch.tensor(14.16922965141578)
ss_xRG = torch.tensor(3.067311966229132)
ss_xSOS = torch.tensor(1.7799299517496816)
ss_xRGS = torch.tensor(0.9123802753622172)
ss_xGS = torch.tensor(1.696536888576733)
ss_xPLCgPl = torch.tensor(101.41583384539453)
# Bulk variables
ss_xShc = torch.tensor(1.151684029)
ss_xRSh = torch.tensor(0.073300997)
ss_xRShP = torch.tensor(1.836249351)
ss_xShP = torch.tensor(81.78421278)
ss_xRShG = torch.tensor(0.476292741)
ss_xShG = torch.tensor(35.06713727)
ss_xRShGS = torch.tensor(0.335544075)
ss_xShGS = torch.tensor(29.27560748)
# Enzymes
ss_xe1 = torch.tensor(0.9327575856796135)
ss_xerp = torch.tensor(0.06724241595039111)
ss_xe2 = torch.tensor(0.9704735836856779)
ss_xeplcg = torch.tensor(0.029526416314323383)
ss_xe3 = torch.tensor(0.40304969898755355)
ss_xeshp = torch.tensor(0.09695030101244738)

# --- Define subset of true parameters ---
k1, k_1 = torch.tensor(0.003), torch.tensor(0.06)
k2, k_2 = torch.tensor(0.01), torch.tensor(0.1)
k3, k_3 = torch.tensor(1), torch.tensor(0.01)
k5, k_5 = torch.tensor(0.06), torch.tensor(0.2)
k6, k_6 = torch.tensor(1), torch.tensor(0.05)
k7, k_7 = torch.tensor(0.3), torch.tensor(0.006)
k9, k_9 = torch.tensor(0.003), torch.tensor(0.05)
k10, k_10 = torch.tensor(0.01), torch.tensor(0.06)
k11, k_11 = torch.tensor(0.03), torch.tensor(0.0045)
k12, k_12 = torch.tensor(0.0015), torch.tensor(0.0001)
k13, k_13 = torch.tensor(0.09), torch.tensor(0.6)
k14, k_14 = torch.tensor(6), torch.tensor(0.06)
k15, k_15 = torch.tensor(0.3), torch.tensor(0.0009)
k17, k_17 = torch.tensor(0.003), torch.tensor(0.1)
k18, k_18 = torch.tensor(0.3), torch.tensor(0.0009)
k19, k_19 = torch.tensor(0.01), torch.tensor(0.0214)
k20, k_20 = torch.tensor(0.12), torch.tensor(0.00024)
k21, k_21 = torch.tensor(0.003), torch.tensor(0.1)
k22, k_22 = torch.tensor(0.03), torch.tensor(0.064)
k23, k_23 = torch.tensor(0.1), torch.tensor(0.021)
k24, k_24 = torch.tensor(0.009), torch.tensor(0.0429)
k25, k_25 = torch.tensor(1), torch.tensor(0.03)
V4, K4 = torch.tensor(450), torch.tensor(50)
V8, K8 = torch.tensor(1), torch.tensor(100)
V16, K16 = torch.tensor(1.7), torch.tensor(1)
ke11, ke12, ke13 = torch.tensor(809), torch.tensor(40000), torch.tensor(450)
ke21, ke22, ke23 = torch.tensor(3.01), torch.tensor(300), torch.tensor(1)
ke31, ke32, ke33 = torch.tensor(20), torch.tensor(6796.6), torch.tensor(3.4)

# Ground truth parameters to assess training performance
t_ps = [k1, k_1, k2, k_2, k3, k_3, k5, k_5, k6, k_6, k7, k_7, k9, k_9, k10, k_10, k11, k_11, k12, k_12, k25, k_25, V4, V8]

# --- Training Loop Setup ---
results = []
num_epochs = 700
dt = ts[1]-ts[0]
repeats = 20

# --- Parameter scaling (log-rounded) ---
scale = []
for i in true_params:
    val = 10**round(np.log(i.item()))
    scale.append(1 if val < 10 else (10 if val < 100 else 100))
scale = torch.tensor(scale)

# --- Neural ODE Training Loop ---
for rep in range(repeats):
    torch.manual_seed(10 + rep)
    start = time.time()

    NN = neural_net.NeuralNet(
        input_size = data_linear.shape[1],
        output_size = len(param_names),
        num_layers = 14,
        nodes_per_layer = {"default": 15},
        activation_funcs = {"default": "hardsigmoid"},
        biases = {"default": None},
        learning_rate = 0.001
    )

    loss_ts, parameters_ts = [], []
    batch_size = n_steps - 1
    batches = np.arange(0, data_linear.shape[0], batch_size)

    for _ in range(num_epochs):
        for idx in range(len(batches) - 1):
            init_state = data_linear[idx]
            predicted_parameters = scale * NN(init_state.squeeze())
            params_copy = {k: v for k, v in zip(params.keys(), predicted_parameters)}

            x_ss_flux = compute_flux(f, x, params_copy, x_ss)
            flux_jacobian = compute_flux_jacobian(compute_flux, S, f, x, params_copy, x_ss)

            ode = GenericODELinearised(S, flux_jacobian)
            predicted_time_series = odeint(ode, init_state, ts, method='dopri5', rtol=1e-4, atol=1e-6)

            loss = torch.nn.functional.mse_loss(data_linear[1:], predicted_time_series[1:])
            loss.backward()
            NN.optimizer.step()
            NN.optimizer.zero_grad()

            loss_ts.append(loss.clone().detach())
            parameters_ts.append(predicted_parameters.clone().detach())

    end = time.time()
    training_t = end - start

    error = weighted_mse(t_ps, predicted_parameters.detach().numpy())
    b_params = np.array([predicted_parameters[k].detach() for k in ind_2])
    b_true_params = np.array([t_ps[k] for k in ind_2])
    b_error = weighted_mse(b_true_params, b_params)

    results.append({
        'training_time': training_t,
        'loss': loss.detach().numpy(),
        'param_dict': [p.detach() for p in predicted_parameters],
        'error': error,
        'boundaries': [predicted_parameters[k].detach().numpy() for k in ind_2],
        'b_error': b_error
    })

# --- Save Results ---
df = pd.DataFrame(results)
df.to_csv(f"EGFR_MM_subnetwork.csv")