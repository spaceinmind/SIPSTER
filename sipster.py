#!/usr/bin/env python3
"""
Flux Density Analysis for Pulsar Archive Files

This script processes PSRCHIVE `.ar` files to extract pulse profile information,
compute the signal-to-noise ratio (SNR), estimate flux densities (S), and
fit a power-law to the flux as a function of frequency.

It also incorporates sky temperature corrections using the Haslam 408 MHz sky map.

Author: Simon C.-C. Ho
Email: simon.ho@anu.edu.au
"""

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psrchive
import sys
from scipy.optimize import curve_fit

np.set_printoptions(threshold=sys.maxsize)

# --------------------------------------------------------------------------------
# Functions for Tsky
# --------------------------------------------------------------------------------

def tsky_init():
    """
    Load the Haslam 408 MHz sky temperature map and reshape it.
    Returns a 2D array scaled to 831 MHz.
    """
    table = "tsky1.ascii"
    data = np.loadtxt(fname=table, dtype="float")
    data = np.reshape(data, (180, 360))
    return data * 0.1  # Approximate scaling to 831 MHz

def get_tsky(l, b, nsky, freq):
    """
    Retrieve sky temperature at a specific galactic (l, b) and frequency.
    """
    i = min(int(l + 0.5), 359)
    j = min(int(b + 90.5), 179)
    tsky = nsky[j, i] * (408. / freq) ** 2.6
    return tsky

# --------------------------------------------------------------------------------
# Fitting function
# --------------------------------------------------------------------------------

def linearFunc(x, intercept, slope):
    return intercept + slope * x

# --------------------------------------------------------------------------------
# Main Program
# --------------------------------------------------------------------------------

# Read command line argument for input filename
nargs = len(sys.argv) - 1
if nargs != 1:
    print("Usage: python flux_density_analysis.py <archive_file.ar>")
    sys.exit(1)

filename = sys.argv[1]

# Sky temperature map
nsky = tsky_init()

# Galactic coordinates of pulsar (example: J1823-3021A)
l = 2.788
b = -7.913

# Set up plot
plt.figure(figsize=(18, 12))
fs = 20  # font size for plots

# Load pulsar archive and preprocess
ar = psrchive.Archive_load(filename)
ar.dedisperse()
ar.remove_baseline()
data = ar.get_data()
print("data.shape =", data.shape)

# Metadata
Nsubint = ar.get_nsubint()
Npol = ar.get_npol()
Nfreq = ar.get_nchan()
Nphase = ar.get_nbin()

cent_freq = ar.get_centre_frequency()
freq_lo = cent_freq - ar.get_bandwidth() / 2.0
freq_hi = cent_freq + ar.get_bandwidth() / 2.0
print(f"Frequency range: {freq_lo:.2f} MHz to {freq_hi:.2f} MHz | Centre: {cent_freq:.2f} MHz")

# Apply zapping mask
mask = ar.get_weights()
print("Mask shape:", mask.shape)

for i in range(Nfreq):
    data[:, :, i, :] *= mask[0, i]

# Scrunch data
time_phase_freq = np.mean(data, axis=1)
time_phase = np.mean(time_phase_freq, axis=1)
freq_phase = np.mean(time_phase_freq, axis=0)
pulse = np.mean(time_phase, axis=0)
frequency = np.linspace(freq_lo, freq_hi, Nfreq)

# Frequency selection range (adjust if needed)
start_idx, end_idx = 70, 930
pulse_idx = np.argmax(pulse)

# Extract signal and noise windows
signal_window = freq_phase[start_idx:end_idx, pulse_idx - 75:pulse_idx + 75]
noise_window = freq_phase[start_idx:end_idx, pulse_idx - 250:pulse_idx - 150]

A = frequency[start_idx:end_idx]
B = np.mean(signal_window, axis=1)
A1 = A - cent_freq
B1 = np.mean(noise_window, axis=1)

# Averaging data
n = 1
df1 = pd.DataFrame({
    'avgResult_A': np.average(A.reshape(-1, n), axis=1),
    'avgResult_B': np.average(B.reshape(-1, n), axis=1)
})

# Compute noise, signal, SNR
noise = np.std(noise_window, axis=1)
signal = np.sum(signal_window, axis=1)
SNR = signal / noise

# Compute Tsky and flux density S
Tsky = np.array([get_tsky(l, b, nsky, freq) for freq in A])
S = SNR * (18 + Tsky) / (3.0 * np.sqrt(2 * 544e6 * 0.05e-3))

# Remove invalid flux values
valid = ~(np.isnan(S) | np.isinf(S))
df_log = pd.DataFrame({
    'avgResult_A_log': np.log10(df1['avgResult_A'][valid]) - np.log10(cent_freq),
    'avgResult_B_log': np.log10(S[valid])
})

# Fit power-law model
a_fit, cov = curve_fit(linearFunc, df_log['avgResult_A_log'], df_log['avgResult_B_log'])
inter, slope = a_fit
d_inter, d_slope = np.sqrt(np.diag(cov))

# Plotting
x_fit = df_log['avgResult_A_log']
y_fit = linearFunc(x_fit, inter, slope)
y_upper = linearFunc(x_fit, inter, slope + d_slope)
y_lower = linearFunc(x_fit, inter, slope - d_slope)

plt.plot(x_fit, y_fit, label='Power-law fit')
plt.plot(x_fit, y_upper, color='green', label='Upper bound')
plt.plot(x_fit, y_lower, color='green', label='Lower bound')
plt.scatter(x_fit, df_log['avgResult_B_log'], label='Pulse signal')

plt.xlabel('log(frequency - centre frequency)', fontsize=fs)
plt.ylabel('log(S)', fontsize=fs)
plt.title(f"{filename}\nSlope: {slope:.3f} ± {d_slope:.3f}", fontsize=fs)
plt.legend(fontsize=fs, loc='upper right')
plt.tick_params(axis='both', which='major', labelsize=fs)
plt.tight_layout()
plt.show()

# Output fit result
print(f"slope = {slope:.4f} ± {d_slope:.4f}, intercept = {inter:.4f}")
