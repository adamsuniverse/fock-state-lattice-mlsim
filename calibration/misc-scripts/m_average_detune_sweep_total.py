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
    # Extract sampled m values at strobe n
    sampled_m = data["m"][:, n - 1]
    # Unique detuning values
    detuning_points = np.unique(detuning)
    # Mean sampled m across repeats for each detuning
    mean_sampled_m = np.array([sampled_m[detuning == delta].mean() for delta in detuning_points])

    # Store results
    results[n] = {"detuning": detuning_points,"mean_m": mean_sampled_m}

plt.figure(figsize=(10, 6))
for n in strobe_numbers:
    plt.plot(results[n]["detuning"], results[n]["mean_m"], marker="o", label=f"n = {n}")

plt.xlabel(r"Detuning $\Delta_3$")
plt.ylabel(r"$\langle m \rangle$")
plt.title(r"Mean Sampled Measurement Outcome (m) vs Detuning")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
