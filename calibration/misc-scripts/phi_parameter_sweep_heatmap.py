import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter

# Read data from the CV
csv_fh = pd.read_csv("../sweep-results/phi-sweep/phi-parameter-sweep-data/sensitivity_phi_sweep_results.csv", names=["phi_phase", "phi_amp", "cfi"])

# Force columns to float, turning any corrupt text into nulls
csv_fh["phi_phase"] = pd.to_numeric(csv_fh["phi_phase"], errors="coerce")
csv_fh["phi_amp"] = pd.to_numeric(csv_fh["phi_amp"], errors="coerce")
csv_fh["cfi"] = pd.to_numeric(csv_fh["cfi"], errors="coerce")

# Drop any non-numeric header rows that might have been loaded otherwise it won't load
csv_fh = csv_fh.dropna(subset=["phi_phase", "phi_amp"])

# Fills holes in the data to be the minimum cfi i.e. rather than nan in the export its fills it in with the minimum cfi  
csv_fh["cfi"] = csv_fh["cfi"].fillna(csv_fh["cfi"].min())

# Get the info and create a pivot table to draw from
pivot_table = csv_fh.pivot(index="phi_phase", columns="phi_amp", values="cfi")
X = pivot_table.columns.values.astype(float)
Y = pivot_table.index.values.astype(float)
Z = pivot_table.values

# Apply 2D Gaussian smoothing 
Z_smooth = gaussian_filter(Z, sigma=1.2)

# Draw the plot size colours etc...
fig, ax = plt.subplots(figsize=(8, 6))
cp = ax.contourf(X, Y, Z_smooth, levels=350 )
cbar = fig.colorbar(cp, ax=ax)

# Set axis label right hand side
cbar.set_label("Classical Fisher Information (CFI)", fontsize=10)

# Pinpoint the peak of the smoothed surface
max_idx = np.unravel_index(np.nanargmax(Z_smooth), Z_smooth.shape)
phi_phase, phi_amp = Y[max_idx[0]], X[max_idx[1]]

# Plot the max marker 
ax.plot(phi_amp, phi_phase, "r*", markersize=8, label=f"Max CFI ({phi_phase:.2f}, {phi_amp:.2f})")

# Plot the 0,0 marker dot
ax.plot(0.0, 0.0, "wo", markersize=4, label="Baseline (0.0, 0.0)")

#Set Axis Labels
ax.set_xlabel(f"Amplitude Feedback Gain (phi_amp)", fontsize=10)
ax.set_ylabel(r"Phase Feedback Gain (phi_phase)", fontsize=10)

# Set Title
ax.set_title("Classical Fisher Information", fontsize=10)
ax.legend(loc="upper left")
plt.tight_layout()
plt.savefig("../sweep-results/phi-sweep/graphs/phi_parameter_sweep_cfi_heatmap.png", dpi=300)
plt.show()
