import numpy as np
import math
import qutip

# The main run_simulation method called by both Three and Four Cavity Simulators.
def run_simulation (h_tot, state, operators,simulation_timers, damping_gamma=None):
    times = np.linspace(simulation_timers[0], simulation_timers[1], simulation_timers[2])
    solver_options = {"progress_bar": "tqdm", "store_final_state": True}
    #solver_options = {"store_final_state": True}
    e_ops = [operators["a"].dag() * operators["a"],
             operators["b"].dag() * operators["b"],
             operators["c"].dag() * operators["c"]]

    if "d" in operators:
        if damping_gamma is None:
            raise ValueError("damping_gamma must be provided for the four cavity simulation.")
        c_ops = [np.sqrt(damping_gamma) * operators["d"]]
        e_ops.append(operators["d"])
        e_ops.append(operators["d"].dag() * operators["d"])
        result = qutip.mesolve (h_tot, state, times, c_ops=c_ops, e_ops=e_ops, options=solver_options)
    else:
        result = qutip.mesolve(h_tot, state, times, c_ops=[], e_ops=e_ops, options=solver_options)
    rho = result.final_state
    return result, rho

# The calculation of m from the Four Cavity Simulation
def calculate_m_from_cavity_four(result, damping_gamma):
    if len(result.expect) < 4:
        raise ValueError(
            "Expectation values for cavity 'd' not found in result. This is a four cavity method"
        )
    # Split into quadratures
    quad_X = np.real(result.expect[3])
    quad_P = np.imag(result.expect[3])

    #Integrate the squared quadrature amplitudes over the pulse to obtain m
    m = np.trapezoid(quad_X ** 2 + quad_P ** 2, result.times)

    # return measured outcome m and the quadratures
    return m, quad_X, quad_P

# The calculation of m from the Three Cavity Simulation
def calculate_m_from_cavity_three(rho_three, operators, theta, m_max):
    p_m = np.zeros(m_max, dtype=float)
    upsilon_list = []

    for m in range(m_max):
        upsilon = calculate_upsilon(m, operators, theta)
        upsilon_list.append(upsilon)
        em = calculate_povm(upsilon)
        p_m[m] = (em * rho_three).tr().real

    #Prevent negative values
    p_m = np.clip(p_m, 0, None)
    # Normalise
    p_m = p_m / np.sum(p_m)
    # Sample it
    calculated_m = np.random.choice(m_max, p=p_m)
    return calculated_m, p_m, upsilon_list[calculated_m]

# Measuring the state of the Three Cavity Simulation
def measure_state(rho_three, upsilon):
    numerator = upsilon * rho_three * upsilon.dag()
    final_state = numerator / numerator.tr()
    return final_state

# Updating the state of the Three Cavity Simulation
def update_state(rho_three, um):
    numerator = um * rho_three * um.dag()
    final_state = numerator / numerator.tr()
    return final_state

# Calculating the Upsilon used for the measurement of the Three Cavity Simulation state
def calculate_upsilon(m, operators, theta):
    n_op = operators["c"].dag() * operators["c"]
    exponent = ((-theta ** 2) * (n_op)**2 /2).expm()
    # If m_max is above 170 then math.factorial will have issues.
    upsilon = ((1j * theta * n_op) ** m / math.sqrt(math.factorial(m))) * exponent
    return upsilon

# Calculating the POVM used for the measurement of the Three Cavity Simulation state
def calculate_povm (upsilon):
    em = upsilon.dag() * upsilon
    return em

# Calculating the new resonance deltas used for the update_state according to the phi_phase parameter and default config deltas passed
def calculate_phase_feedback_control(m, phi_phase, delta01,delta02):
    delta1 = delta01 * np.exp(-phi_phase * m)
    delta2 = delta02 * np.exp(-phi_phase * m)
    return delta1,delta2

# Calculating the new coupling alphas used for the update_state according to the phi_amp parameter and default config alphas passed
def calculate_amplitude_feedback_control(m, phi_amp, alpha01,alpha02):
    alpha1 = alpha01 * np.exp(-phi_amp * m)
    alpha2 = alpha02 * np.exp(-phi_amp * m)
    return alpha1, alpha2

# Calculating the Update operator used for the update of the Three Cavity Simulation state
def calculate_um (operators, T, delta1, delta2,alpha1, alpha2):
    # Split into each cavity terms i.e. cavity 1/2 then cavity2/3
    cavity12 = delta1 * operators["a"].dag() * operators["a"] + alpha1 * operators["a"] * operators["b"].dag() + np.conj(alpha1) * operators["a"].dag() * operators["b"]
    cavity23 = delta2 * operators["b"].dag() * operators["b"] + alpha2 * operators["b"] * operators["c"].dag() + np.conj(alpha2) * operators["b"].dag() * operators["c"]
    update = cavity12 + cavity23
    um = (-1j * T * update).expm()
    return um

# Extract the list of n from the density matrix this forms P(n)_pre and P(n)_post
def get_n_list(rho_three):
    probability_n_list = rho_three.ptrace(2).diag().real
    return probability_n_list

# Calculates the expectation value of n
def calculate_average_n3(operators_three, state):
    n3_op = operators_three["c"].dag() * operators_three["c"]
    avg_n3 = qutip.expect(n3_op, state)
    return avg_n3

