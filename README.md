# Machine Learning Simulation of Fock State Lattices with Quantum Non-Demolition Photon Number Detection

---

## Project Summary

Quantum optical sensors offer far greater sensitivity than their classical counterparts, by leveraging quantum entanglement to overcome the Standard Quantum Limit. 
This research presents a computational simulation designed with Python QuTiP to model and validate the construction of a physical Fock state lattice sensor. 
The sensor consists of a three cavity optical lattice initialised in a specific Fock state and coupled with evanescent coupling. 
As photons migrate between the cavities they are entangled due to the coupling, and interact with a Kerr non-linear material within each cavity. 
The physical experiment also has a mechanical mirror within the third cavity, which when moved due to perturbations, changes the resonant frequency of the cavity, detuning it from resonance.
In order to measure this without destroying the quantum state, a fourth cavity is arranged perpendicular to the third, and connected with a cross-Kerr non-linear material. 
When photons are present within the third cavity, they interact with the cross-Kerr, causing a phase shift within the fourth cavity that is then measured. 
This is a Quantum Non-Demolition measurement output ($m$), which allows us to infer the number of photons in cavity three ($n$) by measuring the phase change of the field transmitted in cavity four. 
For the simulation the full end-to-end four cavity physical setup was modelled, and the phase information simulated and calibrated. 
A reduced three cavity setup was also simulated, to validate the measurement calculations, where both the simulated measurement statistics $P(m)$ and the corresponding conditional cavity photon number distribution $P(n)$ were computed using the stochastic master equation.

---

## Aims & Objectives 

### Four Cavity Simulator

The aim of the Four Cavity Simulation is the full end-to-end simulation of the physical setup as described in previous sections. 
It models the connected three cavity system where the third cavity is connected to a fourth via a cross-Kerr interaction. 
This is to validate the end-to-end concept including the ability to extract the quadrature components and find the optimal operating conditions i.e. the driving pulse, $\theta$, the decay rate and the effective Kraus operator $\Upsilon(m)$. 
This enables the simulated measurement outcome $m$ to be obtained from the heterodyne readout. 
Cavity Four can then be disabled allowing the simulation to use the effective Kraus operator description together with the measurement outcome $(m)$.

### Three Cavity Simulator

The aim of the Three Cavity Simulation is to validate the probabilities of $P(m)$ through the effective measurement model. 
The measurement probability distribution $P(m)$ is used as the machine learning input, while the cavity three photon number distribution <i>P(n)</i><sub><i>pre</i></sub> and <i>P(n)</i><sub><i>post</i></sub> extracted from the simulated density matrix are used as the supervised prediction targets. 
The models are therefore trained to infer the underlying photon number distribution $P(n)$ from the measurement distribution $P(m)$, providing a predictive framework for future use. 
In addition, the simulation will validate the effective Kraus operator description of the QND measurement. 


---

## Current Requirements (Tested)

python >= 3.13.14

qutip >= 5.3.1

numpy >= 2.51

scipy >= 1.18

scipy-stubs >= 1.18.0.1

scikit-learn >= 1.9.0

matplotlib >= 3.11.1

pandas >= 3.0.5

tqdm >= 4.70.0

---

## How to Run
As can be seen from the method map everything is run via main.py.

git clone https://github.com/adamsuniverse/fock-state-lattice-mlsim

cd fsl_qnd_ml_sim

Create a virtual environment

python3 -m venv cav-env

\#Activate it

source cav-env/bin/activate

\# Create the log directory as I forgot to code it.

mkdir ./logs/

\# Install the required dependencies 

pip install -r requirements.txt

\# Adjust the config as appropriate

python main.py

---

## Code Preamble
The research program will be centred around the programmatic implementation of the physical experiment involving coding Python to model all aspects of the system. The physical implementation has many facets that will require modelling. 
The code did not follow a more formal object oriented programming or test driven development, but is modular in style and encapsulates functions, so they can be replaced or tested in isolation. Furthermore, various project variables have been captured within the configuration file,
so the whole project can be driven by the configuration, to produce various outcomes. Where possible it has been written according to the PEP 8 specification. 

All configuration is loaded within main.py where the config variables are extracted here and passed to sub modules and their associated functions, which means each function is config independent and can be tested according to the variables passed and any applicable return values. 
With the exception of three sub-module functions, which call other local functions to help achieve a task, all other functions are called from main.py. To help facilitate any future operator or coder, a method map has been created below, which helps to visualize the various functions and how the simulation is called and runs.

---

# Release Notes

v1.0.0  - This Production Release.

New Development Features:
* Fix issues in README.md for the text corrections and installs - DONE. 
* Refactor and make more efficient
* Cut the detune sweep for positive values for Calibration ML
* Look at PyTorch and GRU/LSTM for measurement M
* Look at unconditional update implementation to go faster.

---

# Config Considerations For v1.0.0

#### For the Main (main.py):

The first configuration item the user should be aware of (this default configuration):

run_four_cavity_sim = True 

join_simulations = False

The run_four_cavity_sim configures whether to run the four cavity simulation at all. Because of the collapse operators it runs significantly slower than the three cavity simulation.
The join_simulations toggles whether the output state of the four cavity simulation gets trimmed and used as the input to the third cavity simulation. If set to false the third cavity simulation will start from an initialised state.

Also, you will want to make sure that:

run_phi_sweep = False

run_detuning_sweep = False

Otherwise, you could be sitting there for a very very long time. Use these only to carry out a parameter sweep across the system.


#### For the Setup (setup.py): 

The cavities have an initial dimension setup cavities 1-3 (6) and cavity 4 (20). Where any increase to cavity 1-3 shifts the dimensions for all three (multiplying matrices) and even just moving this to 10 significantly increases the computational overhead. 6 has been deemed an appropriate number.

The cavities have the initial photon state (2,1,0,0) for cavity 1,2,3,4 respectively. For the physical experiment these would need to be manually injected into the cavities on initialisation.

The cavities have been setup with a theta of 0.3 and the larger this gets the greater the abruptness of photon shift when m appears. It is part of the hardware configuration.

At the moment frequencies (omega) for cavity 1 and 2 are set to 0 and cavity 3 has been set to 0.5. Where the coupling constants between each cavity is 1.0 meaning they will feel a coupling effect. In addition, the self-Kerr is set to 1.0 and the cross-Kerr to 0.5, meaning both interactions are enabled. 

#### For the Simulation (simulation.py): 

The damping gamma for the photonic loss in cavity 4 has been set to 5.77.

#### For the Fisher Information (sensitivity.py): 

The detune-delta used to measure the difference between cavity 3 detuning is 1e-6. Small but enough to get a reasonable Quantum Fisher result. As part of this the transducer gain is actually the parameter for how far the optomechanical mirror moves. This is just set to a 1 multiplier for the moment.

#### For the Machine Learning (learn.py): 

The number of iterations that run within cavity three is set to 20. This was only for demonstration and development purposes. With this small amount of runs the features will greatly outnumber the samples causing standard models like linear regression to overfit. So Ridge Regression will give a slightly more appropriate answer. For the physical experiment it is expected that the Multi-Layered Perceptron will provide the most accurate result. However, to run effectively there needs to be a significantly large sample size of 5000+ iterations. This will then start to provide the model enough training and test data to start to become effective.

### General Points
- Note in main the graphing of the large n and p_m plots. I've set a limit of 50 iterations before this stops graphing those. Once it gets suitably large it does throw errors and 50 has been deemed sufficient for these initial runs. We have also defaulted to graph per iteration rather than per time interval

- Also note that as theta is increased it affects the probability distribution of m, which in turn may get wider than m_max. Therefore a larger m_max is required, summing the probabilities calculating them will indicate if m_max=50 is too small.

---


## Method Map

### Main (`main.py`)

* **`main()`**
   * `run_four_cavity_simulation()` [if enabled]
   * `define_initial_cavity_state()` [if not join_simulations] 
   *  `use four_cavity_state` [if join_simulations] 
   * **[Module: setup]** `define_operators()`
   * **[Module: setup]** `define_hamiltonian()`
   * `run_phi_parameter_sweep()` [if enabled]
   * `run_sensor_detuning_sweep()` [elif enabled]
   * `run_three_cavity_simulation()` [else]
     * **[Module: plot_results]** `plot_histograms()`
     * **[Module: plot_results]** `plot_probabilities()`
     * **[Module: plot_results]** `plot_m_and_avg_n3()`
     * **[Module: plot_results]** `plot_fisher()` x 2
     * **[Module: learn]** `learn_probabilities()` x 6
     * **[Module: plot_results]** `plot_ml()` x 6


* **`run_four_cavity_simulation(simulation_timers, photons)`**
   * **[Module: setup]** `calculate_theta()`
   * **[Module: setup]** `define_initial_cavity_state()`
   * **[Module: setup]** `define_operators()`
   * **[Module: setup]** `define_hamiltonian()`
   * **[Module: simulation]** `run_simulation()`
   * **[Module: simulation]** `calculate_m_from_cavity_four()`
   * **[Module: plot_results]** `plot_continuous()`
   * **[Module: plot_results]** `plot_phase()`
   * **[Module: plot_results]** `plot_histogram()`


* **`run_three_cavity_simulation(simulation_timers,state,operators,hamiltonian,omegas,phi_phase,phi_amp)`**
  * **[LOOP]**
   * **[Module: simulation]** `run_simulation()`
   * **[Module: simulation]** `get_n_list()`
   * **[Module: simulation]** `calculate_m_from_cavity_three()`
   * **[Module: sensitivity]** `calculate_fisher()`
   * **[Module: simulation]** `calculate_um()`
   * **[Module: simulation]** `measure_state()`
   * **[CONDITIONAL IF update_state]** [Module: simulation]** `update_state()`
   * **[Module: simulation]** `calculate_average_n3()`
  * **[LOOP END]**


* **`run_phi_parameter_sweep()`**
  * `run_three_cavity_simulation()`


* **`run_sensor_detuning_sweep()`**
  * `run_three_cavity_simulation()`
  
---

### Setup (`setup.py`)

* **`define_initial_cavity_state(dimensions, photons, number_of, alpha=None)`**
* *(No method calls)*


* **`define_operators(dimensions, number_of)`**
* *(No method calls)*


* **`define_hamiltonian(operators, omegas, coupling, chi, epsilon=0.1)`**
* *(No method calls)*


* **`calculate_theta(chi, damping_gamma, alpha)`**
* *(No method calls)*

---

### Simulation (`simulation.py`)

* **`run_simulation(H_TOT, state, operators, simulation_timers, damping_gamma=None)`**
* *(No method calls)*


* **`calculate_m_from_cavity_four(result, damping_gamma)`**
* *(No method calls)*


* **`calculate_m_from_cavity_three(rho_three, operators, theta, m_max)`**
   * `calculate_upsilon()`
   * `calculate_povm()`


* **`measure_state(rho_three, upsilon)`**
* *(No method calls)*


* **`update_state(rho_three, um)`**
* *(No method calls)*


* **`calculate_upsilon(m, operators, theta)`**
* *(No method calls)*


* **`calculate_povm(upsilon)`**
* *(No method calls)*


* **`calculate_phase_feedback_control(m, phi_phase, delta01, delta02)`**
* *(No method calls)*


* **`calculate_amplitude_feedback_control(m, phi_amp, alpha01, alpha02)`**
* *(No method calls)*


* **`calculate_um(operators, T, delta1, delta2, alpha1, alpha2)`**
* *(No method calls)*


* **`get_n_list(rho_three)`**
* *(No method calls)*


* **`calculate_average_n3(operators_three, state)`**
* *(No method calls)*
---

### Sensitivity (`sensitivity.py`)

* **`calculate_semiclassical_fisher(rho_three, state, operators, simulation_timers, omegas, coupling, chi, transduction_gain, detune_delta, nu)`**
   * **[Module: setup]** `define_hamiltonian()`
   * **[Module: simulation]** `run_simulation()`
  

* **`calculate_fisher(rho_three, state, operators, simulation_timers, omegas, coupling, chi, transduction_gain, detune_delta, nu , theta, m_max, p_m)`**
   * **[Module: setup]** `define_hamiltonian()`
   * **[Module: simulation]** `run_simulation()`
   * `calculate_quantum_fisher()`

  
* **`calculate_quantum_fisher(rho_three, rho_derived)`**
---

### Learn (`learn.py`)

* **`learn_probabilities(p_m, p_n_list, learning_mode, pdml_mlp_options, pdml_ridge_options)`**
* *(No method calls)*

---

### Plot Results (`plot_results.py`)

* **`plot_continuous(result)`**
* *(No method calls)*


* **`plot_continuous_full(result)`**
* *(No method calls)*


* **`plot_phase(quad_X, quad_P)`**
* *(No method calls)*


* **`plot_histogram(dimensions, the_state)`**
* *(No method calls)*


* **`plot_histograms(dimensions, list_of_states)`**
* *(No method calls)*


* **`plot_m_and_avg_n3(list_of_m, iterations, list_of_avg_n3)`**
* *(No method calls)*


* **`plot_probabilities(list_of_p_m, max_bins=20)`**
* *(No method calls)*


* **`plot_fisher(iterations, fisher_type, fisher_list, crb_list)`**
* *(No method calls)*


* **`plot_ml(ml_method, y_test, y_pred, num_samples=3)`**
* *(No method calls)*

---
## REFERENCES
### `Setup.py`
* **[Basic Operations on Quantum Objects](https://qutip.org/docs/4.7/guide/guide-basics.html)** – QuTiP developers and contributors
* **[Using Tensor Products and Partial Traces](https://qutip.org/docs/4.7/guide/guide-tensor.html)** – QuTiP developers and contributors

### `Simulation.py`
* **[Time Evolution and Quantum System Dynamics](https://qutip.org/docs/4.7/guide/guide-dynamics.html)** – QuTiP developers and contributors
* **[Lindblad Master Equation Solver](https://qutip.org/docs/4.7/guide/dynamics/dynamics-master.html#unitary-evolution)** – QuTiP developers and contributors
* **[Lindblad Master Equation Solver (API)](https://qutip.org/docs/4.0.2/modules/qutip/mesolve.html)** – QuTiP developers and contributors
* **[Monte Carlo Solver](https://qutip.org/docs/4.7/guide/dynamics/dynamics-monte.html)** – QuTiP developers and contributors
* **[All Equations](https://a.co/d/00uzOXR2)** - D. Walls,  Gerard J. Milburn - "Quantum Optics"

### `Sensitivity.py`
* **[Quantum Fisher Information](https://qutip.org/qutip-julia-tutorials/QuantumToolbox.jl/time_evolution/Quantum_Fisher_Information_with_automatic_differentiation.html)** – Orjan Ameye
* **[scipy.linalg.solve_sylvester](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_sylvester.html#scipy.linalg.solve_sylvester)** – The SciPy community
* **[numpy.eye](https://numpy.org/devdocs/reference/generated/numpy.eye.html)** – NumPy Developers
* **[All Equations](https://amzn.eu/d/0g1Lo3Ie)** - Howard Wiseman, Gerard J. Milburn - "Quantum Measurement and Control"

### `plot_results.py`
* **[Visualization of quantum states and processes (v4.2)](https://qutip.org/docs/4.2/guide/guide-visualization.html)** – QuTiP developers and contributors
* **[Visualization of quantum states and processes (v4.7)](https://qutip.org/docs/4.7/guide/guide-visualization.html)** – QuTiP developers and contributors
* **[matplotlib.axes.Axes.bar](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.bar.html)** – Matplotlib development team
* **[Create Overlapping Graphs Using Python](https://medium.com/@nazishjaveed164/create-overlapping-graphs-using-python-04b86dabacd9)** – N. Javeed
* **[Matplotlib Gallery Examples](https://matplotlib.org/stable/gallery/index.html)** – Matplotlib development team
* **[Scikit-learn Visualizations](https://scikit-learn.org/stable/visualizations.html)** – scikit-learn developers

### `Learn.py`
* **[Varying regularization in Multi-layer Perceptron](https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mlp_alpha.html)** – scikit-learn developers
* **[Neural network models (supervised)](https://scikit-learn.org/stable/modules/neural_networks_supervised.html)** – scikit-learn developers
* **[MLPClassifier Reference](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)** – scikit-learn developers
* **[MLPRegressor Reference](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor)** – scikit-learn developers
* **[Neural Networks with Scikit](https://python-course.eu/machine-learning/neural-networks-with-scikit.php)** – B. Klein
* **[Metrics and scoring](https://scikit-learn.org/stable/modules/model_evaluation.html)** – scikit-learn developers
* **[Supervised Learning: Multilayer Perceptron (Video)](https://www.youtube.com/watch?v=Z_wUdhW7fEI)** – learndataa
* **[Simple Machine Learning Code Tutorial with Sklearn (Video)](https://www.youtube.com/watch?v=-IvNzmrcyUM)** – Python Simplified
* **[How-To-Choose-The-Number-Of-Hidden-Layers-And_Node](https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw)** - Stack Exchange User Forum
* **[A Novel Approach in Determining Neural Networks Architecture](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9252936)** - M. Rachmatullah, J. Santoso, K. Surendro
* **[Building RNN, LSTM, and GRU for time series using PyTorch](https://towardsdatascience.com/building-rnn-lstm-and-gru-for-time-series-using-pytorch-a46e5b094e7b/)** - K. Kuguoglu
* **[Deep Learning (Adaptive Computation and Machine Learning series)](https://www.deeplearningbook.org/contents/mlp.html)** - I. Goodfellow, Y. Bengio, A. Courville
* **[Geometric Pyramid Rule](https://machinelearningtheory.org/docs/Deep-Learning/designing-the-architecture/)** - Professor Yi He (references T. Masters below)
* **[Geometric Pyramid Rule](https://www.sciencedirect.com/book/monograph/9780080514338/practical-neural-network-recipies-in-c-and-and)** - T. Masters
* **[Introduction to Neural Networks with Java]([https://books.google.co.uk/books?id=Swlcw7M4uD8C&printsec=frontcover&redir_esc=y#v=onepage&q&f=false](https://books.google.co.uk/books?id=Swlcw7M4uD8C&pg=PA39))** - J. Heaton
* **[Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)** - M. Neilson
