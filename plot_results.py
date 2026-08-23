import matplotlib.pyplot as plt
import numpy as np

# This plots the continuous evolution of photon within the Four Cavity Simulation
# It shows photon numbers across all four cavities. Useful to see the impacts on the Fourth Cavity.
def plot_continuous(result):
    fig, ax = plt.subplots()
    ax.plot(result.times, result.expect[0])
    ax.plot(result.times, result.expect[1])
    ax.plot(result.times, result.expect[2])
    ax.plot(result.times, result.expect[4])
    ax.set_xlabel("Time")
    ax.set_ylabel("Average Number of Photons")
    ax.legend(("Cavity 1", "Cavity 2", "Cavity 3", "Cavity 4"))
    plt.show()
    plt.close(fig)

# Plots the large continuous plot for the Three Cavity Simulation
# As displayed in the calibration document.
def plot_continuous_full(list_of_results):
    fig, ax = plt.subplots(figsize=(10,5))
    for idx, result in enumerate(list_of_results):
        times = np.array(result.times)
        time = idx + (times / times[-1])
        # Plot each cavity (only add label on idx==0 for clean legend)
        ax.plot(time, result.expect[0], color="tab:blue", label="Cavity 1" if idx == 0 else "")
        ax.plot(time, result.expect[1], color="tab:orange", label="Cavity 2" if idx == 0 else "")
        ax.plot(time, result.expect[2], color="tab:green", label="Cavity 3" if idx == 0 else "")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Photon Population")
    ax.set_title("3-Cavity Photon Numbers")
    ax.legend(loc="upper right")
    ax.grid(True)
    plt.tight_layout()
    plt.show()
    plt.close(fig)

# This plots the phase across Cavity 4 and demonstrated the cross-Kerr interaction
def plot_phase(quad_X, quad_P):
    fig,ax = plt.subplots()
    ax.plot(quad_X, quad_P, label="Cavity 4 Phase")
    ax.set_xlabel('X')
    ax.set_ylabel('P')
    ax.legend()
    plt.show()
    plt.close(fig)

# This was used for the Four Cavity Simulation. Not that useful and can probably be removed.
def plot_histogram(dimensions, the_state):
    fig, axes = plt.subplots(1, 3, figsize= (12,3))
    x_bin = np.arange(dimensions["n_cav"])
    for idx, ax in enumerate(axes):
        rho = the_state.ptrace(idx).diag().real
        ax.bar(x_bin, rho)
        ax.set_title(f"Cavity {idx +1}")
        ax.set_ylim([0,1])
        ax.set_ylabel("Probability")
        ax.set_xlabel("Photon Number")
    plt.tight_layout()
    plt.show()
    plt.close(fig)

# This plots the big histogram of probabilities of P(n) but showing individual per cavity
# This will only run if the number of Three Cavity iterations is less than 50.
def plot_histograms(dimensions, list_of_states):
    num_iterations = len(list_of_states)
    fig, axes = plt.subplots(num_iterations, 3, figsize= (12,3* num_iterations))
    x_bin = np.arange(dimensions["n_cav"])
    for idx, state in enumerate(list_of_states):
        for cav_idx in range(3):
            ax = axes[idx, cav_idx]
            rho = state.ptrace(cav_idx).diag().real
            ax.bar(x_bin, rho)
            if idx == 0:
                ax.set_title(f"Cavity {cav_idx +1}")
            ax.set_ylim([0,1])
            ax.set_ylabel("Probability")
            ax.set_xlabel("Photon Number")
    plt.tight_layout()
    plt.show()
    plt.close(fig)

# This plots the average of the measured m against the expectation value of n
def plot_m_and_avg_n3(list_of_m, iterations, list_of_avg_n3):
    fig, ax1 = plt.subplots(figsize=(9,5))
    #print (list_of_m)
    color = "tab:blue"
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Measured m", color=color)
    ax1.plot(iterations, list_of_m, color=color, marker="o", linestyle="-", label="Measured m")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True)

    ax2 = ax1.twinx()
    color = "tab:green"
    ax2.set_ylabel(r"Average Photon Number $\langle n_3 \rangle$", color=color)
    ax2.plot(iterations, list_of_avg_n3, color=color, marker="s", linestyle="--", label=r"$\langle n_3 \rangle$")
    ax2.tick_params(axis='y', labelcolor=color)
    plt.title(r"Sampled m and Cavity 3 Average Photon Number $\langle n_3 \rangle$ vs. Iterations")
    fig.tight_layout()
    plt.show()

# This plots the big histogram of probabilities of P(m)
# This will only run if the number of Three Cavity iterations is less than 50.
def plot_probabilities(list_of_p_m, max_bins=20):
    num_iterations = len(list_of_p_m)
    cols = 3
    rows = int(np.ceil(num_iterations / cols))
    fig, axes = plt.subplots(rows, cols, figsize= (12,3 * rows))
    axes = axes.flatten()
    for idx, p_m in enumerate(list_of_p_m):
        # Plot the full distribution as bars
        ax = axes[idx]
        bins = np.arange(min(len(p_m), max_bins))
        ax.bar(bins, p_m[:max_bins], label="Probability $p(m)$")
        if idx == 0:
            ax.set_title("Probability M")
        ax.set_xlabel("Photon Number")
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1.0)
        ax.set_xlim(0, 5.0)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend()
    plt.tight_layout()
    plt.show()
    plt.close(fig)

# This is what plots any fisher i.e. both Classical and Quantum Fisher Information
def plot_fisher(iterations, fishertype, fisher_list, crb_list):
    if fishertype == None:
        fishertype = "fishertype_none"
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize= (8,6))
    ax1.plot(iterations, fisher_list, label=f"{fishertype} Fisher Information $F(\\phi)$")
    ax1.set_title("Quantum Sensing")
    ax1.set_ylabel("Fisher Information")
    ax1.grid(True)
    ax1.legend()

    ax2.plot(iterations, crb_list, label=f"{fishertype} Cramer Rao $(\\Delta\\phi)$")
    ax2.set_title("Cramer Rao Plot")
    ax2.set_ylabel("Cramer Rao")
    ax2.grid(True)
    ax2.legend()
    plt.show()
    plt.close(fig)

# This i the main Probability Distribution Machine Learning Plot method. Sampling and the providing Predictive vs Actual lines
def plot_ml(ml_method, y_test, y_pred, num_samples=5):
    # Select a few random samples from the test set
    indices = np.random.choice(len(y_test), size=min(num_samples, len(y_test)), replace=False)
    plt.figure(figsize=(12, 4 * len(indices)))
    for idx, sample in enumerate(indices, 1):
        plt.subplot(len(indices), 1, idx)
        photon_numbers = np.arange(y_test.shape[1])
        # Plot true vs predicted probabilities
        plt.plot(photon_numbers, y_test[sample], "o-", label="True Measure $P(n)$", color="black", linewidth=2)
        plt.plot(photon_numbers, y_pred[sample], "s--", label="Predicted Measure $P(n)$", color="red", alpha=0.8)
        plt.title(f"Test Sample #{sample} from {ml_method}")
        plt.xlabel("Photon Number $n$")
        plt.ylabel("Probability $P(n)$")
        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.show()

# Called for the detuning calibration sweep. This calculates an m averages (shot and continuous) against detuning
def plot_avg_m_vs_detuning(detunings, avg_m, mode):
    plt.figure(figsize=(6,4))
    plt.plot(detunings, avg_m, "r-o", markersize=4, label=r"Mean Clicks $\langle m \rangle$")
    plt.xlabel(r"Detuning ($\Delta_3$)")
    plt.ylabel(fr"Average measured across {mode} strobe ($m$)")
    plt.title("Detector Response vs. Detuning")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./calibration/sweep-results/detune-sweep/Strobe-1.5/generic-data/avg_{mode}_m_vs_detuning.png", dpi=300)
    #plt.show()
    plt.close()

# This plots the results from the Calibration Machine Leaning used in detuning sweeps
def plot_detuning_calibration_ml(y_test, y_pred, detune_type):
    plt.figure(figsize=(6,5))
    plt.scatter(y_test, y_pred, alpha=0.6, edgecolors="k", label=f"ML Predictions Based on {detune_type}")
    min_val = min(np.min(y_test), np.min(y_pred))
    max_val = max(np.max(y_test), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label=r"Ideal ($y=x$)")
    plt.xlabel(r"Actual Detuning ($\Delta_3$)")
    plt.ylabel(r"Predicted Detuning (Pred $\Delta_3$)")
    plt.title("Sensor Detuning Calibration")
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.savefig(f"./calibration/sweep-results/detune-sweep/Strobe-1.5/generic-data/detuning_calibration_ml_{detune_type}.png", dpi=300)
    #plt.show()
    plt.close()

# Called for the detuning calibration sweep. This calculates the classical fisher information averages (shot and continuous) against detuning
def plot_detuning_cfi(detunings, cfi_values, mode):
    plt.figure(figsize=(6,4))
    plt.plot(detunings, cfi_values, "b-o", markersize=2, label="Mean Classical Fisher")
    plt.xlabel(r"Detuning ($\Delta_3$)")
    plt.ylabel(fr"CFI average across {mode} strobe $F(\Delta_3)$")
    plt.title("Sensor Sensitivity vs. Detuning")
    plt.grid(True, linestyle=':')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./calibration/sweep-results/detune-sweep/Strobe-1.5/generic-data/detuning_{mode}_cfi.png", dpi=300)
    #plt.show()
    plt.close()

