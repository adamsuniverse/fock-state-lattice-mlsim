import qutip
import numpy as np
from scipy import linalg

from setup import define_hamiltonian
from simulation import run_simulation, calculate_m_from_cavity_three

# This calculates the semiclassical fisher information according the values of N against detuning
# This method is currently unused as only the classical and quantum fisher information was required in the end.
def calculate_semiclassical_fisher(rho_three,state, operators, simulation_timers, omegas, coupling, chi, transduction_gain, detune_delta, nu):
    n = (operators["c"].dag() * operators["c"])
    n3_op = (operators["c"].dag() **2) * (operators["c"] ** 2)

    # Calculate the expectation values using n specified above and the copied rho.
    mean_photon_n3_expect = qutip.expect(n, rho_three)
    mean_n3_op_expect = qutip.expect(n3_op, rho_three)
    mean_sq_photon_n3_expect =  mean_n3_op_expect + mean_photon_n3_expect

    # Calculate the variance of n3
    variance_n3 = mean_sq_photon_n3_expect - (mean_photon_n3_expect ** 2)

    # Copy the detune operators.
    omegas_detuned = omegas.copy()
    omegas_detuned["r3"] += detune_delta

    # Define the Hamiltonian we want to use with the detune difference
    h_detuned = define_hamiltonian(operators, omegas_detuned, coupling, chi)

    # Run the second simulation with the new omega detuned. and get the results
    results_detuned, rho_detuned = run_simulation(h_detuned, state, operators, simulation_timers)

    # Extract <N3> at the end of the shadow step
    # (results_detuned.expect[2] corresponds to Cavity 3 then [-1] is the last time slice in that array.
    mean_photon_n3_expectation_detuned = results_detuned.expect[2][-1]

    # F = (∂ϕ⟨N1⟩)2/Var(N3), with ∂ϕ = ∆′(ϕ) ∂∆
    # Work out derivative with respect to the expectation value i.e how this changes with respect to the detune.
    # Take the detuned expectation value - the previous expected value / change in detune.
    derived_delta  = (mean_photon_n3_expectation_detuned - mean_photon_n3_expect) / detune_delta

    # ∂ϕ = ∆′(ϕ) ∂∆
    d_phi_n3 = transduction_gain * derived_delta

    # F = (∂ϕ⟨N1⟩)2/Var(N3)
    fisher = (d_phi_n3 ** 2) /variance_n3

    #∆ϕ= \frac{1}{\nu F(\phi)
    fisher_trimmed = np.maximum(fisher, 1e-6)
    cramer_rao = 1.0 / np.sqrt(nu * fisher_trimmed)

    return fisher, cramer_rao

# This is the main calculate fisher method. It's used to calculate both the classical fisher information and the quantum fisher information.
# calculate_quantum_fisher is called from this method and both values are returned.
def calculate_fisher(rho_three, state, operators, simulation_timers, omegas, coupling, chi, transduction_gain, detune_delta, nu, theta, m_max, p_m):
    # Copy the detune operators.
    omegas_detuned = omegas.copy()
    omegas_detuned["r3"] += detune_delta

    # Define the Hamiltonian we want to use with the detune difference
    H_detuned = define_hamiltonian(operators, omegas_detuned, coupling, chi)

    # Run the second simulation with the new omega detuned. and get the results then get pm_detuned
    results_detuned, rho_detuned = run_simulation(H_detuned, state, operators, simulation_timers)
    m, pm_detuned, upsilon = calculate_m_from_cavity_three(rho_detuned, operators, theta, m_max)

    # Work out derivative with respect to P(m) i.e how this changes with respect to the detune.
    # Take the detuned P(m) - the original P(m) value / change in detune.
    dp_derived_delta  = (pm_detuned - p_m) / detune_delta

    # ∂ϕ = ∆′(ϕ) ∂∆
    d_phi_pm = transduction_gain * dp_derived_delta

    # Prevent division by 0 when p_m is 0
    filter = 1e-12
    submask = p_m > filter
    # Fϕ = (∂_{ϕ} P(m|ϕ))^2/P(m|ϕ)
    fisher = np.sum((d_phi_pm[submask] ** 2) /p_m[submask])

    # Trim to prevent zero/small vals
    fisher_trimmed = np.maximum(fisher, 1e-6)
    #∆ϕ= \frac{1}{\nu F_{c}
    cramer_rao = 1.0 / np.sqrt(nu * fisher_trimmed)

    # Work out Quantum Fisher Information. PLEASE NOTE it doesn't have the transduction gain in there!!
    rho_derived = (rho_detuned - rho_three) / detune_delta
    qfi = calculate_quantum_fisher(rho_three, rho_derived)

    # Do same trimming to try and prevent zero/small vals
    qfi_trimmed = np.maximum(qfi, 1e-6)
    # ∆ϕ= \frac{1}{\nu F_{c}
    cramer_qfi = 1.0 / np.sqrt(nu * qfi_trimmed)

    return fisher, cramer_rao, qfi, cramer_qfi

# This calculates the Quantum Fisher Information
# This is based and referenced on the Julia implementation. See references in README.
def calculate_quantum_fisher(rho_three, rho_derived):
    # Add small regularization to avoid numerical issues with zero eigenvalues
    rho_final_converted = rho_three.full()
    rho_derived_converted = rho_derived.full()

    shape = rho_final_converted.shape[0]
    reg = 1e-12 * np.eye(shape)
    rho_reg = rho_final_converted + reg

    # Solve for the symmetric logarithmic derivative L
    # dρ = (1/2)(ρL + Lρ)
    # This is a Sylvester equation: ρL + Lρ = 2*dρ
    L = linalg.solve_sylvester(rho_reg, rho_reg, 2 * rho_derived_converted)

    # Fisher information F = Tr(dρ * L)
    qfi = np.real(np.trace(np.matmul(rho_derived_converted, L)))

    return qfi


