import numpy as np

# Setup Config #
# Number of iterations for the three cav runs
number_of_three_cav_runs = 20
# Laser Drive
epsilon = 0.1
# This is the damping coefficient for cavity 4
damping_gamma = 5.77
# Laser Amplitude.
# Used to initialise cavity 4 coherent with an alpha value, as well as calculate theta
alpha = np.sqrt(2)
# Config Theta for 3 Cavity Hardcoded Theta
theta = 0.3
# Initial Photons States
photons_cav1, photons_cav2, photons_cav3 = 2,1,0
# Basis States or Max Dimension
dimensions = {"n_cav": 6, "n_drive": 20}
# Kerr Non-Linear Strength for Self and Cross Kerr
chi = {"s_chi": 1, "c_chi": 0.5}
# Coupling Constants
coupling = {"g1": 1.0, "g2": 1.0}
# Cavity Frequencies
omegas ={"r1": 0.0, "r2": 0.0, "r3": 0.5, "r4": 0.0}


## Simulation Config ##
## **** This needs to be read in full before running any simulation ****
## If you want to run the four cavity simulation at all
run_four_cavity_sim = True
## If you want to use the output of the four cavity simulation and feed into the third
join_simulations = False

## PHI SWEEP CONFIG
## Do you want to run a full calibration sweep of phi to find optimum amplitude and phase adjustments in update_state().
## Remember to change the state_update to True below. Otherwise, it will be incorrect.
run_phi_sweep = False
## Phi Sweep Config
phi_start = -1.0
phi_stop = 1.1
phi_step = 0.1
phi_filename = "./calibration/sweep-results/phi-sweep/phi-parameter-sweep-data/sensitivity_phi_sweep_results.csv"

## DETUNING SWEEP CONFIG
## Do you want to run a full calibration sweep of Detuning.
run_detuning_sweep = False
num_detune_repeats = 100
detune_range_start = -3.0
detune_range_stop = 3.0
detune_range_steps = 31
detuning_file = "./calibration/sweep-results/detune-sweep/Strobe-1.5/generic-data/detuning_sweep_raw.npz"

## RUN THREE CAVITY SIMULATION CONFIG
## Simulation Start Stop and Steps For Three Cavity Simulations
## NB this is the same for Four Cavity Simulation too
times_start = 0
times_stop = 1.5
times_steps = 1000
## Max space for p_m array
m_max = 50
## Do we want to update the state and apply the control pulse according to the below values
state_update = False
## Calibration Sweep to Find Optimum Alpha and Detuning Based on CFI
phi_phase = 1
phi_amp = 1
## Trainable deltas for the update operator. This will be the base configuration. This is controlled with phi_phase
delta1, delta2 = 1, 1
## Trainable alphas for the update operator. This will be the base configuration. This is controlled with phi_amplitude.
alpha1, alpha2 = 1, 1
## Time interval for control pulse in update operator
T = 1.0
## Calibration Machine Learning (CML)
cml_mlp_options = {"solver" :  "lbfgs", "alpha": 1e-4, "hidden_layer_sizes": (32,16), "max_iter" : 5000, "random_state": 1}
## Probability Distribution Machine Learning (PDML)
pdml_mlp_options = {"solver" :  "lbfgs", "alpha": 1e-4, "hidden_layer_sizes": (32,16), "max_iter" : 5000, "random_state": 1}
pdml_ridge_options = {"alpha" :  1.0}


### Fisher Config ###
### Delta Detune for Fisher Run
detune_delta = 1e-6
### Number of runs for Cramer Rao
nu = 1
###Transducer Gain value (movement of mirror)
transducer_gain = 1.0


