import numpy as np
import qutip
import config
import logging

# Configure the global settings ONCE here
logging.basicConfig(
    filename='./logs/app.log',
    filemode='a',
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

from setup import (
    calculate_theta,
    define_hamiltonian,
    define_initial_cavity_state,
    define_operators,
)
from simulation import (
    calculate_m_from_cavity_three,
    calculate_m_from_cavity_four,
    get_n_list,
    run_simulation,
    measure_state,
    update_state,
    calculate_phase_feedback_control,
    calculate_amplitude_feedback_control,
    calculate_um,
    calculate_average_n3,
)
from learn import (
    learn_detuning,
    learn_probabilities
)

from plot_results import (
    plot_continuous,
    plot_continuous_full,
    plot_phase,
    plot_histogram,
    plot_histograms,
    plot_probabilities,
    plot_ml,
    plot_m_and_avg_n3,
    plot_fisher,
    plot_detuning_calibration_ml,
    plot_detuning_cfi,
    plot_avg_m_vs_detuning,
)
from sensitivity import (
    calculate_semiclassical_fisher,
    calculate_fisher,
)

def main():
    np.random.seed(100)
    photons = (config.photons_cav1, config.photons_cav2, config.photons_cav3)
    simulation_timers = (config.times_start, config.times_stop, config.times_steps)

    if config.run_four_cavity_sim:
        four_cavity_state = run_four_cavity_simulation(simulation_timers, photons)
    else:
        four_cavity_state = None

    if config.join_simulations & config.run_four_cavity_sim:
        state = four_cavity_state
    else:
        # The Three Cavity System has been decided to construct completely independently here from the 4 cavity system.
        logging.debug("Define the initial cavity state for three cavity simulation")
        state_three = define_initial_cavity_state(config.dimensions, photons, 3,)
        # As the tensor product out of the initial cavity state will be (216 x 1) this needs conversion to a density matrix 216 x 216
        state = qutip.ket2dm(state_three)

    logging.debug("Define the Three Cavity Operators")
    operators_three = define_operators(config.dimensions, 3)
    logging.debug("Define the Three Cavity Hamiltonian")
    h_tot_three = define_hamiltonian(operators_three, config.omegas, config.coupling, config.chi)

    # Run calibration sweeps of phi or m->detuning or just run the three cavity simulation with default configuration.
    if config.run_phi_sweep:
        run_phi_parameter_sweep(simulation_timers, state, operators_three, h_tot_three)

    elif config.run_detuning_sweep:
        # Run the detuning sweep please refer to config for settings.
        detune_results = run_sensor_detuning_sweep(simulation_timers, state, operators_three)

        # Flatten the PM Values
        pm_numpy_array = np.array(detune_results["lists_of_all_pm"])
        flat_pm = pm_numpy_array.reshape(pm_numpy_array.shape[0], -1)

        # Run the Calibration Detuning ML and Plots for Predicting Detune vs AVG (m) and Raw PM Need Average PM
        m_y_test, m_y_pred, m_mse, r2 = learn_detuning(detune_results["lists_of_all_m"], detune_results["lists_of_detuned"], "mlp", config.cml_mlp_options)
        pm_y_test, pm_y_pred, pm_mse, r2 = learn_detuning(flat_pm, detune_results["lists_of_detuned"], "mlp",config.cml_mlp_options)
        plot_detuning_calibration_ml(m_y_test, m_y_pred, "(m)")
        plot_detuning_calibration_ml(pm_y_test, pm_y_pred, "(pm)")
        plot_avg_m_vs_detuning(detune_results["detuning_range"], detune_results["list_of_avg_m"], "last")
        plot_avg_m_vs_detuning(detune_results["detuning_range"], detune_results["list_of_avg_all_m"], "all")
        plot_detuning_cfi(detune_results["detuning_range"], detune_results["list_of_avg_fisher"], "last")
        plot_detuning_cfi(detune_results["detuning_range"], detune_results["list_of_avg_all_fisher"], "all")
    else:
        # Run the Three Cavity Simulation
        sim_three_results = run_three_cavity_simulation(simulation_timers ,state ,operators_three ,h_tot_three ,config.omegas ,config.phi_phase ,config.phi_amp)

        # This restriction here because any more than this & it gets silly on the graphs for n and p_m
        if len(sim_three_results["iterations"]) <= 50:
            logging.info("Plotting Histograms")
            plot_histograms(config.dimensions, sim_three_results["list_of_states"])
            logging.info("Plotting Probabilities")
            plot_probabilities(sim_three_results["lists_of_pm"])

        # Plot the continuous, avg n with m then two Fisher Information Plots
        logging.info("Plotting Fisher Information")
        plot_continuous_full(sim_three_results["list_of_results"])
        plot_m_and_avg_n3(sim_three_results["lists_of_m"], sim_three_results["iterations"], sim_three_results["list_of_avg_n3"])
        plot_fisher(sim_three_results["iterations"], "Classical", sim_three_results["fisher_classical_results"], sim_three_results["crb_classical_results"])
        plot_fisher(sim_three_results["iterations"], "QFI", sim_three_results["fisher_quantum_results"], sim_three_results["crb_quantum_results"])

        # Run the Machine Learning (PDML) and Plots for Learning P(m) -> P(n)_pre & P(n)_post measurements
        models = {"mlp": "MLP", "linear": "Linear Regression", "ridge": "Ridge Regression"}
        source = {"pre": "lists_of_n_premeasure", "post": "lists_of_n_postmeasure"}
        ml_results={}
        for models_key, models_value in models.items():
            for source_key, source_value in source.items():
                y_test, y_pred, mse, r2 = learn_probabilities(sim_three_results["lists_of_pm"], sim_three_results[source_value], models_key, config.pdml_mlp_options, config.pdml_ridge_options)
                ml_results[f"{models_key}_{source_key}"] = {"mse": mse, "r2": r2}
                plot_ml(f"{models_value} {source_key.capitalize()} Measurement", y_test, y_pred, num_samples=3)
        print (ml_results)

# This is the main Four Cavity Simulation Method
def run_four_cavity_simulation(simulation_timers, photons):
    logging.debug("Calculating Theta")
    theta = calculate_theta(config.chi, config.damping_gamma, config.alpha)
    logging.debug("Define Initial Four Cavity State")
    state = define_initial_cavity_state(config.dimensions, photons, 4, config.alpha)
    logging.debug("Define Four Cavity Operators")
    operators_four = define_operators(config.dimensions, 4)
    logging.debug("Define Four Cavity Hamiltonian")
    h_tot_four = define_hamiltonian(operators_four, config.omegas, config.coupling, config.chi, config.epsilon)

    logging.info("Run the Four Cavity Simulation")
    result_four, rho_four = run_simulation(h_tot_four, state, operators_four, simulation_timers, config.damping_gamma)

    logging.debug("Get Four Cavity M and the Quadratures using the Equations")
    m, quad_X, quad_P = calculate_m_from_cavity_four(result_four, config.damping_gamma)

    logging.info("Plot the Four Cavity Results & Phase")
    plot_continuous(result_four)
    plot_phase(quad_X, quad_P)

    # Update the state to be just three cavities used to pass as input to three cavity simulation
    logging.debug("Update State for 3 Cavities")
    state = rho_four.ptrace([0, 1, 2])

    logging.info("Plot the Four Cavity Histogram")
    plot_histogram(config.dimensions, state)
    return state

# This is the main Three Cavity Simulation Method
def run_three_cavity_simulation(simulation_timers, state, operators_three, h_tot_three, omegas, phi_phase, phi_amp):
    lists_of_pm = []
    lists_of_m = []
    lists_of_n_premeasure = []
    lists_of_n_postmeasure = []
    list_of_states = []
    list_of_results = []
    fisher_classical_results = []
    fisher_quantum_results = []
    crb_classical_results = []
    crb_quantum_results = []
    iterations = []
    list_of_avg_n3 = []

    # Run the loop for the Three Cavity Simulations.
    for i in range (0,config.number_of_three_cav_runs):
        # Run the simulation for the Three Cavity System.
        logging.info(f"Run the Three Cavity Simulation Iteration: {i}")
        result_three, rho_three = run_simulation(h_tot_three, state, operators_three, simulation_timers)

        # Get the list of n photons and probabilities pre-measurement
        logging.debug("Get the list of pre measured n values")
        n_list_premeasure = get_n_list(rho_three)
        logging.debug("Add the list of pre measured n values to a larger list")
        lists_of_n_premeasure.append(n_list_premeasure)

        # Calculate M
        logging.debug("Calculate the list of p_m values and get m and the corresponding upsilon")
        m, p_m, upsilon = calculate_m_from_cavity_three(rho_three,operators_three, config.theta, config.m_max)
        logging.debug("Add the list of m values to a larger list")
        lists_of_pm.append(p_m)
        lists_of_m.append(m)

        logging.debug("Calculate the Fisher Information")
        fisher_classical, cramer_rao_classical, qfi, cramer_qfi = calculate_fisher(rho_three, state, operators_three, simulation_timers, omegas, config.coupling, config.chi, config.transducer_gain, config.detune_delta, config.nu, config.theta, config.m_max, p_m)

        logging.debug("Measure the state with Upsilon")
        #state = rho_three
        state = measure_state(rho_three, upsilon)

        # Get the list of n photons and probabilities Post Measurement
        logging.debug("Get the list of post measure n values")
        n_list_postmeasure = get_n_list(state)
        logging.debug("Add the list of post measure n values to a larger list")
        lists_of_n_postmeasure.append(n_list_postmeasure)

        # Prepare the update state variables with the phi parameters
        delta1,delta2 = calculate_phase_feedback_control(m, phi_phase, config.delta1, config.delta2)
        alpha1, alpha2 = calculate_amplitude_feedback_control(m, phi_amp, config.alpha1, config.alpha2)

        logging.debug("Calculate Update Operator")
        um = calculate_um(operators_three, config.T, delta1, delta2, alpha1, alpha2)

        if config.state_update:
            logging.debug("Apply the update to the state")
            state = update_state(state, um)

        # Calculate expectation value of n
        avg_n3 = calculate_average_n3(operators_three, state)
        list_of_avg_n3.append(avg_n3)

        logging.debug("Add all the results to the larger lists for plotting and ML")
        fisher_classical_results.append(fisher_classical)
        fisher_quantum_results.append(qfi)
        crb_classical_results.append(cramer_rao_classical)
        crb_quantum_results.append(cramer_qfi)
        iterations.append(i)
        list_of_states.append(state)
        list_of_results.append(result_three)

    #OUTSIDE THE LOOP# Prepare the Return of the Results
    results = { "iterations": iterations,
                "list_of_results": list_of_results,
                "list_of_states": list_of_states,
                "lists_of_m": lists_of_m,
                "lists_of_n_premeasure": lists_of_n_premeasure,
                "lists_of_n_postmeasure": lists_of_n_postmeasure,
                "list_of_avg_n3": list_of_avg_n3,
                "lists_of_pm": lists_of_pm,
                "fisher_classical_results": fisher_classical_results,
                "crb_classical_results": crb_classical_results,
                "fisher_quantum_results": fisher_quantum_results,
                "crb_quantum_results": crb_quantum_results,
               }

    return  results

# This is a Three Cavity calibration method. To sweep through all expect values for phi and then carry out update_state
# So this needs to be set as True within the config. It will then sweep making adjustments and write the results to a file.
# Phi is brute force the simulation will just accept these local values not the ones in the configuration.
def run_phi_parameter_sweep(simulation_timers, state,operators_three,h_tot_three):
    with open(config.phi_filename, "w") as f:
        f.write("phi_phase,phi_amp,mean_fisher\n")
        for phi_phase in np.arange(config.phi_start, config.phi_stop, config.phi_step):
            phi_phase = np.round(phi_phase, 2)
            for phi_amp in np.arange(config.phi_start, config.phi_stop, config.phi_step):
                phi_amp = np.round(phi_amp, 2)
                sim_three_results = run_three_cavity_simulation(simulation_timers, state.copy(), operators_three, h_tot_three, config.omegas, phi_phase, phi_amp)
                average_fishy = np.mean(sim_three_results["fisher_classical_results"])
                f.write(f"{phi_phase:.2f},{phi_amp:.2f},{average_fishy:.6f}\n")
                f.flush()

# This is also a Three Cavity calibration method. To sweep through all expected values of detune and then collect, save and graph the results
# The Three Cavity Simulation internally has an inner iteration loop.
# This method provides a further two loop over the top of the Three Cavity Simulation
# The first loop is for the values for the detune. Then another loop within it (defined by num_repeats) provides multiple results per detune step.
def run_sensor_detuning_sweep(simulation_timers, state, operators_three, num_repeats= config.num_detune_repeats):
    detuning_range = np.linspace(config.detune_range_start, config.detune_range_stop, config.detune_range_steps)
    lists_of_fishers = []
    lists_of_cramers = []
    lists_of_all_m = []
    lists_of_detuned = []
    list_of_avg_m = []
    list_of_avg_all_m = []
    list_of_avg_fisher = []
    list_of_avg_all_fisher = []
    lists_of_all_pm = []

    for delta_3 in detuning_range:
        # Modify omega for cavity three
        detuned_omegas = config.omegas.copy()
        detuned_omegas["r3"] = config.omegas["r3"] + delta_3
        h_tot_detuned = define_hamiltonian(operators_three, detuned_omegas, config.coupling, config.chi)
        final_m_per_detune = []
        final_fisher_per_detune = []
        all_m_per_detune =[]
        all_fisher_per_detune = []

        for rep in range(num_repeats):
            logging.info(f"Repetition: {rep}")
            # Run for this detuned Hamiltonian
            logging.info(f"Starting run with delta_3 = {detuned_omegas['r3']}")
            sim_three_results = run_three_cavity_simulation(simulation_timers, state.copy(), operators_three, h_tot_detuned, detuned_omegas, config.phi_phase, config.phi_amp)
            all_m_per_detune.append(sim_three_results["lists_of_m"])
            final_m_per_detune.append(sim_three_results["lists_of_m"][-1])
            all_fisher_per_detune.append(sim_three_results["fisher_classical_results"])
            final_fisher_per_detune.append(sim_three_results["fisher_classical_results"][-1])
            lists_of_fishers.append(sim_three_results["fisher_classical_results"])
            lists_of_all_m.append(sim_three_results["lists_of_m"])
            lists_of_all_pm.append(sim_three_results["lists_of_pm"])
            lists_of_detuned.append(delta_3)
            lists_of_cramers.append(sim_three_results["crb_classical_results"])

        #Outside the transaction loop
        list_of_avg_m.append(np.mean(final_m_per_detune))
        list_of_avg_all_m.append(np.mean(all_m_per_detune))
        list_of_avg_fisher.append(np.mean(final_fisher_per_detune))
        list_of_avg_all_fisher.append(np.mean(all_fisher_per_detune))

    # OUTSIDE THE LOOP # Prepare the Return of the Results
    detune_results = {"lists_of_all_m": lists_of_all_m,
                      "lists_of_all_pm": lists_of_all_pm,
                      "lists_of_fishers": lists_of_fishers,
                      "lists_of_cramers": lists_of_cramers,
                      "lists_of_detuned": lists_of_detuned,
                      "list_of_avg_m": list_of_avg_m,
                      "list_of_avg_all_m": list_of_avg_all_m,
                      "list_of_avg_fisher": list_of_avg_fisher,
                      "list_of_avg_all_fisher": list_of_avg_all_fisher,
                      "detuning_range": detuning_range,
    }
    # Save all the information from the run as it can take 36hrs to complete.
    np.savez(config.detuning_file,
             p_m = np.array(lists_of_all_pm),
             detuning = np.array(lists_of_detuned),
             m = np.array(lists_of_all_m),
             fisher = np.array(lists_of_fishers),
             avg_m = np.array(list_of_avg_m),
             avg_all_m=np.array(list_of_avg_all_m),
             avg_fisher = np.array(list_of_avg_fisher),
             avg_all_fisher = np.array(list_of_avg_all_fisher))

    return  detune_results

if __name__ == "__main__":
    main()