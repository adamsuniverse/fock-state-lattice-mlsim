import numpy as np
import matplotlib.pyplot as plt
import os

strobe_numbers = [1,3,6,12]
filename = "detuning_sweep_raw.npz"
relative_folder = f"../sweep-results/detune-sweep/Strobe-1.5/"
results = {}
for n in strobe_numbers:
    filepath = os.path.join(relative_folder,str(n), filename)
    print(f"Loading n = {n}: {filepath}")
    data = np.load(filepath)
    detuning = data["detuning"]
    # Extract P(m) and sampled m at strobe n
    p_m = data["p_m"][:, n - 1, :]
    sampled_m = data["m"][:, n - 1]
    # Possible m values: 0, 1, 2, ..., m_max - 1
    m_values = np.arange(p_m.shape[1])
    # <m> calculated directly from P(m)
    expected_m = np.sum(p_m * m_values, axis=1)
    # Unique detuning values
    detuning_points = np.unique(detuning)
    # Average m across repeats at each detuning
    mean_expected_m = np.array([expected_m[detuning == delta].mean()for delta in detuning_points])
    # Also calculate sampled mean
    mean_sampled_m = np.array([sampled_m[detuning == delta].mean() for delta in detuning_points])

    # Store everything
    results[n] = {"detuning": detuning_points,"mean_pm": mean_expected_m,"mean_sampled": mean_sampled_m }

# Plot Everything
plt.figure(figsize=(10, 6))
for n in strobe_numbers:
    plt.plot(results[n]["detuning"], results[n]["mean_pm"], marker="o", label=f"n = {n}")

plt.xlabel(r"Detuning $\Delta_3$")
plt.ylabel(r"$\langle m \rangle$")
plt.title(r"Mean Measurement Outcome P(m) vs Detuning")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
