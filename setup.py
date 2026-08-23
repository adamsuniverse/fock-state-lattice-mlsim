import qutip
import numpy as np

# This defines the initial cavity state for both Four and Three Cavity Simulations
def define_initial_cavity_state(dimensions, photons, number_of, alpha=None):
    if number_of not in (3,4):
        raise ValueError(f"number_of must be 3 or 4, got {number_of}")
    psi1 = qutip.fock(dimensions["n_cav"], photons[0])
    psi2 = qutip.fock(dimensions["n_cav"], photons[1])
    psi3 = qutip.fock(dimensions["n_cav"], photons[2])

    if number_of == 3:
        psi0 = qutip.tensor(psi1, psi2, psi3)
    elif number_of == 4:
        psi4 = qutip.coherent(dimensions["n_drive"], alpha)
        psi0 = qutip.tensor(psi1, psi2, psi3, psi4)
    return psi0

# This defines the operators for both Four and Three Cavity Simulations
def define_operators(dimensions, number_of):
    if number_of not in (3,4):
        raise ValueError(f"numberof must be 3 or 4, got {number_of}")

    # Define Cavity Ops and Identity parameters
    c_o_cav = qutip.destroy(dimensions["n_cav"])
    i_o_cav = qutip.qeye(dimensions["n_cav"])
    c_o_drv = qutip.destroy(dimensions["n_drive"])
    i_o_drv = qutip.qeye(dimensions["n_drive"])

    if number_of == 3:
        # Set for three cavities and what a,b and c are in terms of destruction operators per cavity
        a = qutip.tensor(c_o_cav, i_o_cav, i_o_cav)
        b = qutip.tensor(i_o_cav, c_o_cav, i_o_cav)
        c = qutip.tensor(i_o_cav, i_o_cav, c_o_cav)
        operators = {"a": a, "b": b, "c": c}
    elif number_of == 4:
        # Set for four cavities and what a, b, c and d are in terms of destruction operators per cavity
        a = qutip.tensor (c_o_cav, i_o_cav, i_o_cav, i_o_drv )
        b = qutip.tensor (i_o_cav, c_o_cav, i_o_cav, i_o_drv )
        c = qutip.tensor (i_o_cav, i_o_cav, c_o_cav, i_o_drv )
        d = qutip.tensor (i_o_cav, i_o_cav, i_o_cav, c_o_drv )
        operators = {"a": a, "b": b, "c": c, "d": d}

    return operators

# This defines the Hamiltonians for both Four and Three Cavity Simulations
def define_hamiltonian(operators, omegas, coupling, chi, epsilon=0.1):
    # h_bar constant
    hbar = 1.0

    a,b,c = operators["a"], operators["b"], operators["c"]

    # Set the terms of the Hamiltonian and include the resonant frequency of each of the cavities.
    H = omegas["r1"] * (a.dag() * a) + omegas["r2"]  * (b.dag() * b) + omegas["r3"]  * (c.dag() * c)
    H_AB = coupling["g1"] * (a * b.dag() + b * a.dag())
    H_BC = coupling["g2"] * (b * c.dag() + c * b.dag())

    # Add Self-Kerr Nonlinearity
    H_A_Kerr = 0.5 * chi["s_chi"] * (a.dag()**2) * (a**2)
    H_B_Kerr = 0.5 * chi["s_chi"] * (b.dag()**2) * (b**2)
    H_C_Kerr = 0.5 * chi["s_chi"] * (c.dag()**2) * (c**2)

    # Add it all together.
    H_TOT = H +  H_AB + H_BC + H_A_Kerr + H_B_Kerr + H_C_Kerr

    # This is for the Four Cavity Simulation additonal parameters
    # Including base 4, cross-Kerr and the laser drive
    if "d" in operators:
        d = operators["d"]
        H4 = omegas["r4"] * d.dag() * d
        H_Kerr_Interaction = hbar * chi["c_chi"] * (c.dag() * c) * (d.dag() * d)
        H_D_Drive = 1j * epsilon * (d - d.dag())
        H_TOT += H4 + H_Kerr_Interaction + H_D_Drive
    return H_TOT

# This defines the theta used in the Four Cavity Simulation.
# The Three Cavity Simulation currently has its one hardcoded.
def calculate_theta(chi, damping_gamma, alpha):
    N = alpha **2
    theta = np.sqrt(6 *N) * chi["c_chi"]/damping_gamma
    return theta
