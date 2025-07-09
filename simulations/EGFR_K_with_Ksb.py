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


class MyODE(torch.nn.Module):
    def __init__(self, Kss, Ksb, Kbs, Kbb):
        """
        s_matrix: Stoichiometry matrix S (torch.Tensor).
        flux_jacobian: Precomputed Jacobian J_f (torch.Tensor).
        steady_state_flux: Flux values at steady state f(y*) (torch.Tensor).
        steady_state_y: Steady-state concentrations y* (torch.Tensor).
        epsilon: Scaling factor for diffusion.
        """
        super().__init__()
        self.Kss = Kss  # Jacobian of flux (num_reactions x num_species)
        self.Ksb = Ksb
        self.Kbs = Kbs
        self.Kbb = Kbb

    def forward(self, t, y):
        """
        Drift term: y @ J_f.T)
        y shape: (..., num_species)
        """
        x = y[:,:15]
        z = y[:,15:]
        #print(x.shape,z.shape)
        # Compute the linear flux as the Jacobian times the state
        drift_x = (self.Kss @ x.T + self.Ksb @ z.T).T  # J_f is the linearized Jacobian, y is the state
        drift_z = (self.Kbs @ x.T + self.Kbb @ z.T).T # J_f is the linearized Jacobian, y is the state
        drift = torch.cat([drift_x, drift_z], dim=1)
        #print(drift.shape)
        return drift
        

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
# --- Load Data ---
data_linear = torch.tensor(
    pd.read_csv('data_linear_MM.csv').values, dtype=torch.float32
)[:, [i for i in range(23) if i not in [15, 20, 14, 18, 21, 16, 19, 17]]]
data_linear = data_linear.unsqueeze(1)
n_steps = 300
ts = torch.linspace(0, 3, n_steps)

# Initial conditions
ics_state = torch.tensor(pd.read_csv("ics_EGFR_MM_15_20_14_18_21_16_19_17.csv", header=None).to_numpy(dtype=float), dtype=torch.float32).squeeze()
ics_state_full = torch.tensor(pd.read_csv("ics_EGFR.csv", header=None).to_numpy(dtype=float), dtype=torch.float32).squeeze()

# Parameters
df = pd.read_csv("params_EGFR_MM_15_20_14_18_21_16_19_17.csv")
param_names = list(df.keys())
true_params = torch.tensor(df.to_numpy(dtype=float)[0], dtype=torch.float32)
params = dict(zip(param_names, true_params))

# System matrices and steady states
S = torch.tensor(pd.read_csv("S_EGFR_MM_15_20_14_18_21_16_19_17.csv", header=None).to_numpy(dtype=float), dtype=torch.float32)
f = pd.read_csv("f_EGFR_MM_15_20_14_18_21_16_19_17.csv", header=None).values.T[0].tolist()
x = pd.read_csv("x_EGFR_MM_15_20_14_18_21_16_19_17.csv", header=None).values.T[0].tolist()
x_full = pd.read_csv("x_EGFR.csv", header=None).values.T[0].tolist()
ss_x = torch.tensor(pd.read_csv("steady_states_EGFR_MM_15_20_14_18_21_16_19_17.csv", header=None).values.T[0].tolist())
ss_x_full = torch.tensor(pd.read_csv("steady_states_EGFR.csv", header=None).values.T[0].tolist())


# --- Construct Index Sets ---
n = 23
idx = torch.tensor(list(range(14)) + [n - 1])
idx_b = torch.tensor(list(range(14, n - 1)))


def construct_Ks(pp):
    device = pp.device  # ensure all constants live on same device
    zero = lambda: pp[0] * 0  # gradient-safe zero
    
    K_ss = torch.stack([
        torch.stack([-1.0*pp[0]*ss_xEGF, -1.0*pp[0]*ss_xR, 1.0*pp[1], zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero()]),
        
        torch.stack([-1.0*pp[0]*ss_xEGF, -1.0*pp[0]*ss_xR, 1.0*pp[1], zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero()]),
        
        torch.stack([1.0*pp[0]*ss_xEGF, 1.0*pp[0]*ss_xR, -4.0*pp[2]*ss_xRa - 1.0*pp[1], 2.0*pp[3], zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero()]),
        
        torch.stack([zero(), zero(), 2.0*pp[2]*ss_xRa, -1.0*pp[4] - 1.0*pp[3], 1.0*pp[5], zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero()]),
        
        torch.stack([zero(), zero(), zero(), 1.0*pp[4], pp[22]*ss_xRP/(ss_xRP + K4)**2 - pp[22]/(ss_xRP + K4), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero()]),
        
        torch.stack([zero(), zero(), zero(), zero(), -1.0*pp[6]*ss_xPLCg, -1.0*pp[6]*ss_xRP, 1.0*pp[7], zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero()]),
        
        torch.stack([zero(), zero(), zero(), zero(), 1.0*pp[6]*ss_xPLCg, 1.0*pp[6]*ss_xRP, -1.0*pp[8] - 1.0*pp[7], 1.0*pp[9], zero(), zero(), zero(), zero(), zero(), zero(), zero()]),
        
        torch.stack([zero(), zero(), zero(), zero(), 1.0*pp[11]*ss_xPLCgP, zero(), 1.0*pp[8], -1.0*pp[10] - 1.0*pp[9], 1.0*pp[11]*ss_xRP, zero(), zero(), zero(), zero(), zero(), zero()]),
        
        torch.stack([zero(), zero(), zero(), zero(), -1.0*pp[11]*ss_xPLCgP, zero(), zero(), 1.0*pp[10], -1.0*pp[20] - 1.0*pp[11]*ss_xRP + pp[23]*ss_xPLCgP/(ss_xPLCgP + K8)**2 - pp[23]/(ss_xPLCgP + K8), zero(), zero(), zero(), zero(), zero(), 1.0*pp[19]]),
        
        torch.stack([zero(), zero(), zero(), zero(), -1.0*pp[12]*ss_xGrb, zero(), zero(), zero(), zero(), -1.0*pp[24]*ss_xRShP - 1.0*pp[26]*ss_xShP - 1.0*pp[12]*ss_xRP - 1.0*pp[19]*ss_xSOS, 1.0*pp[13], -1.0*pp[19]*ss_xGrb, zero(), 1.0*pp[18], zero()]),
        
        torch.stack([zero(), zero(), zero(), zero(), 1.0*pp[12]*ss_xGrb, zero(), zero(), zero(), zero(), 1.0*pp[12]*ss_xRP, -1.0*pp[14]*ss_xSOS - 1.0*pp[13], -1.0*pp[14]*ss_xRG, 1.0*pp[15], zero(), zero()]),
        
        torch.stack([zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), -1.0*pp[19]*ss_xSOS, -1.0*pp[14]*ss_xSOS, -1.0*pp[14]*ss_xRG - 1.0*pp[25]*ss_xRShG - 1.0*pp[27]*ss_xShG - 1.0*pp[19]*ss_xGrb, 1.0*pp[15], 1.0*pp[18], zero()]),
        
        torch.stack([zero(), zero(), zero(), zero(), 1.0*pp[17]*ss_xGS, zero(), zero(), zero(), zero(), zero(), 1.0*pp[14]*ss_xSOS, 1.0*pp[14]*ss_xRG, -1.0*pp[16] - 1.0*pp[15], 1.0*pp[17]*ss_xRP, zero()]),
        
        torch.stack([zero(), zero(), zero(), zero(), -1.0*pp[17]*ss_xGS, zero(), zero(), zero(), zero(), 1.0*pp[19]*ss_xSOS, zero(), 1.0*pp[19]*ss_xGrb, 1.0*pp[16], -1.0*pp[18] - 1.0*pp[29]*ss_xRShP - 1.0*pp[17]*ss_xRP - 1.0*pp[28]*ss_xShP, zero()]),   
        
        torch.stack([zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), 1.0*pp[20], zero(), zero(), zero(), zero(), zero(), -1.0*pp[21]]),
    ])
    
    K_sb = pp[30:].reshape(15,8)
    K_bs = K_sb.T

    return K_ss, K_sb, K_bs


# --- Training Loop ---
init_state = torch.tensor(ss_x_full[:-6]*ics_state_full[:-6]).reshape(1,-1)
x_s0 = init_state.index_select(1, idx)
z0 = init_state.index_select(1, idx_b)
state = torch.cat((x_s0, z0), 1)

scale = torch.tensor([1]*214); scale[22] = 10
results = []
repeats = 10
num_epochs = 600
loss_ts, parameters_ts = [], []
init_state = torch.tensor(ss_x_full[:-6]*ics_state_full[:-6]).reshape(1,-1);


for r in range(repeats):
    torch.manual_seed(r*100 + 7)
    
    NN = neural_net.NeuralNet(
        input_size = data_linear.shape[2], 
        output_size = 150,
        num_layers=10,
        nodes_per_layer={"default": 45}, 
        activation_funcs={"default": "hardsigmoid", "layer_specific": {-1: "relu"}}, 
        biases={"default": None},
        learning_rate=0.001
    )

    for _ in range(num_epochs):
        predicted_parameters = scale*NN(x_s0.squeeze())
        K_ss,K_sb,K_bs = construct_Ks(predicted_parameters)
        ode = MyODE(K_ss, K_sb, K_bs)
        predicted_time_series = odeint(ode, state, ts, method='dopri5')[:,:,:15]
        
        loss = torch.nn.functional.mse_loss(
            data_linear[1:], predicted_time_series[1:]
        )
        
        # Perform a gradient descent step
        loss.backward()
        NN.optimizer.step()
        NN.optimizer.zero_grad()
        
        # Store prediction and loss
        loss_ts.append(loss.clone().detach())
        parameters_ts.append(predicted_parameters.clone().detach())
        
    end = time.time()
    
    # Convert both lists to plain Python floats
    values1 = [t.item() for t in loss_ts]
    values2 = [t.item() for t in parameters_ts[-1][:30]]
    values3 = [t.item() for t in parameters_ts[-1][30:]]
    
    # Use zip_longest to pad the shorter list with None
    zipped = list(zip_longest(values1, values2, values3, fillvalue=None))
    
    # Convert to DataFrame
    df = pd.DataFrame(zipped, columns=["List 1", "List 2", "List 3"])
        
    df.to_csv(f"_ksb_{scale[22]}_{num_epochs}_{r}.csv")