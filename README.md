# Neural Parameter Calibration for Biochemical Systems
This repository supports the paper "Neural Parameter Calibration for Biochemical Systems".

## Requirements
This repository requires the following libraries and packages to run the scripts:

- Python
- PyTorch: Install:
```
pip install torch
```
- NumPy: Install:
```
pip install numpy
```
- SciPy: Install:
```
pip install scipy
```
- Scikit-learn: Install:
```
pip install scikit-learn
```
- Matplotlib: Install:
```
pip install matplotlib
```
- Torchsde: Install:
```
pip install torchsde
```
- Torchdiffeq: Install:
```
pip install torchdiffeq
```
To install all necessary packages, run:
```
pip install numpy scipy scikit-learn torch matplotlib torchsde torchdiffeq
```

## Code Overview
- `simulations/`: This directory includes egfr.ipynb, four Python scripts used for the analysis of the neural calibration methods explored in the paper, and numerous Excel files necessary to run the scripts.

  * `egfr.ipynb`: The primary script for simulating the EGFR signalling network dynamics, explored through the approaches proposed in the paper.
  * `EGFR_MM_subnetwork.py`: The Python script containing repeated simulations of the EGFR signalling pathways subnetwork with Michaelis-Menten dynamics. Approach 1 on page 47.
  * `EGFR_K_with_Ksb.py`: The Python script containing repeated simulations of the EGFR signalling pathways with Michaelis-Menten dynamics, based on approach 1 on page 50.
  * `EGFR_K_with_Kbb.py`: The Python script containing repeated simulations of the EGFR signalling pathways with Michaelis-Menten dynamics, based on approach 2 on page 50.
  * `EGFR_boundary_nodes.py`: The Python script containing repeated simulations of the EGFR signalling pathways subnetwork with Michaelis-Menten dynamics and auxiliary bulk variables used to analyse the calibration of boundary parameters. Approach 3 on page 47.
  * `utils.py`

## Implementation Details
This code is dependent on the NeuralABM repository by Gaskin et al.: https://github.com/ThGaskin/NeuralABM. Ensure that this is operational prior to running the scripts given here, and move the Excel files above into the NeuralABM folder. The neural network architectures, random seeds, repeats, scaling, priors, etc. can be changed by adjusting the corresponding variables.
