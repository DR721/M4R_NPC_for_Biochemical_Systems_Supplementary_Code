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


# --- Define steady-state concentrations (ss_x*) ---
# Example: ss_xR = [R] at steady state
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
# bulk variables
ss_xShc = torch.tensor(1.151684029)
ss_xRSh = torch.tensor(0.073300997)
ss_xRShP = torch.tensor(1.836249351)
ss_xShP = torch.tensor(81.78421278)
ss_xRShG = torch.tensor(0.476292741)
ss_xShG = torch.tensor(35.06713727)
ss_xRShGS = torch.tensor(0.335544075)
ss_xShGS = torch.tensor(29.27560748)
# enzymes
ss_xe1 = torch.tensor(0.9327575856796135)
ss_xerp = torch.tensor(0.06724241595039111)
ss_xe2 = torch.tensor(0.9704735836856779)
ss_xeplcg = torch.tensor(0.029526416314323383)
ss_xe3 = torch.tensor(0.40304969898755355)
ss_xeshp = torch.tensor(0.09695030101244738)

# --- Define true parameters (k1, k_1, ..., V4, K4, etc.) ---
k1 = torch.tensor(0.003)
k_1 = torch.tensor(0.06)
k2 = torch.tensor(0.01)
k_2 = torch.tensor(0.1)
k3 = torch.tensor(1)
k_3 = torch.tensor(0.01)
k5 = torch.tensor(0.06)
k_5 = torch.tensor(0.2)
k6 = torch.tensor(1)
k_6 = torch.tensor(0.05)
k7 = torch.tensor(0.3)
k_7 = torch.tensor(0.006)
k9 = torch.tensor(0.003)
k_9 = torch.tensor(0.05)
k10 = torch.tensor(0.01)
k_10 = torch.tensor(0.06)
k11 = torch.tensor(0.03)
k_11 = torch.tensor(0.0045)
k12 = torch.tensor(0.0015)
k_12 = torch.tensor(0.0001)
k13 = torch.tensor(0.09)
k_13 = torch.tensor(0.6)
k14 = torch.tensor(6)
k_14 = torch.tensor(0.06)
k15 = torch.tensor(0.3)
k_15 = torch.tensor(0.0009)
k17 = torch.tensor(0.003)
k_17 = torch.tensor(0.1)
k18 = torch.tensor(0.3)
k_18 = torch.tensor(0.0009)
k19 = torch.tensor(0.01)
k_19 = torch.tensor(0.0214)
k20 = torch.tensor(0.12)
k_20 = torch.tensor(0.00024)
k21 = torch.tensor(0.003)
k_21 = torch.tensor(0.1)
k22 = torch.tensor(0.03)
k_22 = torch.tensor(0.064)
k23 = torch.tensor(0.1)
k_23 = torch.tensor(0.021)
k24 = torch.tensor(0.009)
k_24 = torch.tensor(0.0429)
k25 = torch.tensor(1)
k_25 = torch.tensor(0.03)
V4 = torch.tensor(450)
K4 = torch.tensor(50)
V8 = torch.tensor(1)
K8 = torch.tensor(100)
V16 = torch.tensor(1.7)
K16 = torch.tensor(1)
# enzyme parameters
ke11 = torch.tensor(809)
ke12 = torch.tensor(40000)	
ke13 = torch.tensor(450)	
ke21 = torch.tensor(3.01)	
ke22 = torch.tensor(300)	
ke23 = torch.tensor(1)
ke31 = torch.tensor(20)
ke32 = torch.tensor(6796.6)	
ke33 = torch.tensor(3.4)


# --- Load Data ---
# Load observed trajectory data for the subnetwork
df = pd.read_csv('data_linear_MM.csv')
data_linear = torch.tensor(df.values, dtype=torch.float32)[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,22]] 

# Load initial conditions
ics_state = torch.tensor(pd.read_csv("ics_EGFR_MM.csv", header=None).values, dtype=torch.float32).squeeze()

# Load parameter values
df = pd.read_csv("params_EGFR_MM.csv")
param_names = list(df.keys())
true_params = torch.tensor(df.to_numpy(dtype=float)[0], dtype=torch.float32)
params = dict(zip(param_names, true_params))

# Load stoichiometry matrix
S = torch.tensor(pd.read_csv("S_EGFR_MM.csv", header=None).values, dtype=torch.float32)

# Load f, x and steady state vectors
f = pd.read_csv("f_EGFR_MM.csv", header=None).values.T[0].tolist()
x = pd.read_csv("x_EGFR_MM.csv", header=None).values.T[0].tolist()
ss_x = torch.tensor(pd.read_csv("steady_states_EGFR_MM.csv", header=None).values.T[0].tolist())


# --- Indexing for state partitioning ---
n = 23  # total number of species
idx = torch.tensor(list(range(14)) + [n - 1])        # observed nodes
idx_b = torch.tensor(list(range(14, n - 1)))         # hidden/boundary nodes


# --- Define K-matrix constructor ---
def construct_Ks(pp):
    device = pp.device
    zero = lambda: pp[0] * 0  # gradient-safe zero
    one = lambda: pp[0] / pp[0]  # gradient-safe one

    # Build full system matrix K_full of size (n x n)
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

    # Partition K_full into submatrices
    K_ss = K_full.index_select(0, idx).index_select(1, idx)
    K_sb = K_full.index_select(0, idx).index_select(1, idx_b)

    return K_ss, K_sb


# --- Trajectory Generator ---
def traj_gen(x_t, z_t, Kss, Ksb, ts, state):
    trajectory = []
    z = z_t
    for t in ts:
        dxs_dt = Kss @ x_t + Ksb @ z
        dstate_dt = torch.cat((dxs_dt, dz_dt))
        state = state + dstate_dt * dt
        z = state[n_s:]
        trajectory.append(state.unsqueeze(0))
    return torch.cat(trajectory)[::1000, :n_s]  # Downsample to reduce memory usage


# --- Initial Setup ---
n_s, n_b = 15, 8
noise_level = 0.0
init_state = torch.tensor(ss_x * ics_state).reshape(1, -1)
x_s0 = init_state.index_select(1, idx).squeeze()
z0 = init_state.index_select(1, idx_b).squeeze(0)
state = torch.cat((x_s0, z0))

# Time grid for integration
t0, t1 = 0.0, 3.0
dt = 0.00001
ts_K = torch.arange(t0, t1, dt)
dz_dt = torch.zeros(n_b)

# Create lists of variable names for plotting/debugging
names = ['{R}', '{EGF}', '{Ra}', '{R2}', '{RP}', '{PLCg}', '{RPL}', '{RPLP}', '{PLCgP}', '{Grb}', '{RG}', '{SOS}', '{RGS}', '{GS}', 
         '{Shc}', '{RSh}', '{RShP}', '{ShP}', '{RShG}', '{ShG}', '{RShGS}', '{ShGS}', '{PLCgPI}']
x_names = [names[i] for i in idx]
z_names = [names[i] for i in idx_b]


# --- Neural Network Setup ---
torch.manual_seed(1)
num_epochs = 400

# Define priors over parameters for Bayesian regularisation (optional)
priors = [{"distribution": "uniform", "parameters": {"lower": 0.0001, "upper":0.5}} for i in range(len(params))]
priors[22] = {"distribution": "uniform", "parameters": {"lower": 0, "upper":0.00001}}
priors[23] = {"distribution": "uniform", "parameters": {"lower": 0, "upper":0.00001}}
priors[44] = {"distribution": "uniform", "parameters": {"lower": 300, "upper":500}}
priors[46] = {"distribution": "uniform", "parameters": {"lower": 0, "upper":0.00001}}
priors[4] = {"distribution": "uniform", "parameters": {"lower": 0.75, "upper":1.25}}
priors[8] = {"distribution": "uniform", "parameters": {"lower": 0.75, "upper":1.25}}
priors[42] = {"distribution": "uniform", "parameters": {"lower": 0.75, "upper":1.25}}
priors[45] = {"distribution": "uniform", "parameters": {"lower": 0.75, "upper":1.25}}

# Initialise NN model for parameter inference
NN = neural_net.NeuralNet(
    input_size=data_linear.shape[1],
    output_size=len(params),
    num_layers=14,
    nodes_per_layer={"default": 15},
    activation_funcs={"default": "hardsigmoid", "layer_specific": {-1: "abs"}},
    biases={"default": None},
    learning_rate=0.001
)

# Scale vector for parameter transformation
scale = torch.tensor([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                      1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                      1,1,1,1,10,1,0])  # 1s for trainable params, 0 for frozen ones, 10 for the very large parameter


# --- Training Loop ---
loss_ts, parameters_ts = [], []
start = time.time()

for _ in range(num_epochs):
    predicted_parameters = scale * NN(x_s0)
    K_ss, K_sb = construct_Ks(predicted_parameters)
    predicted_time_series = traj_gen(x_s0, z0, K_ss, K_sb, ts_K, state)

    loss = torch.nn.functional.mse_loss(data_linear[1:], predicted_time_series[1:])

    # Backpropagation
    loss.backward()
    NN.optimizer.step()
    NN.optimizer.zero_grad()

    # Logging
    loss_ts.append(loss.clone().detach())
    parameters_ts.append(predicted_parameters.clone().detach())

end = time.time()
print(f"Training time: {end-start:.2f} seconds")


# --- Save Results ---
values1 = [t.item() for t in loss_ts]
values2 = [t.item() for t in parameters_ts[-1]]
zipped = list(zip_longest(values1, values2, fillvalue=None))
df_out = pd.DataFrame(zipped, columns=["Loss", "Final Parameters"])
df_out.to_csv("EGFR_boundary_nodes.csv", index=False)