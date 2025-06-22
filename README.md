# SIPSTER

**Spectral Index of Pulses with Sky Temperature Estimation Routine**

SIPSTER is a Python tool for estimating the spectral index of a single radio pulse using pulsar data from PSRCHIVE `.ar` files. It applies sky temperature correction based on the Haslam 408 MHz sky map and fits a power-law to the pulse flux across frequency.

---

## Features

- Loads and processes `.ar` files via `psrchive`
- Applies sky temperature corrections using `tsky1.ascii`
- Computes signal-to-noise ratios and flux densities
- Fits a spectral index using log-log regression
- Outputs slope and intercept of the spectral fit
- Visualizes data and model fits interactively with `matplotlib`

---

## Requirements

Install the required Python packages via:

```bash
pip install -r requirements.txt  
```

## Usage
python flux_density_analysis.py <archive_file.ar>

## Example:  
python flux_density_analysis.py J1823-3021A.ar  

## Output
Terminal: Prints spectral index slope ± error and intercept  

Plot: Displays log-log fit of flux vs. frequency offset
