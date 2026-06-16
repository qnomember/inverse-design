# FDTD Simulation and Data Export for Planar Chiral Flower Structure

This repository contains a Python-based **MEEP FDTD simulation workflow** for a planar chiral flower-shaped nanostructure in a silver thin film.  
The structure is illuminated by **right circularly polarized (RCP)** and **left circularly polarized (LCP)** light to analyze its optical response and circular dichroism.

The main purpose of this project is to simulate the optical behavior of a chiral nanostructure and systematically save important simulation outputs for later plotting, analysis, and thesis/report preparation.

---

## Project Goals

This project is designed to:

- simulate the optical response of a planar chiral flower-shaped nanostructure
- calculate transmission spectra under RCP and LCP illumination
- evaluate circular dichroism in transmission
- save simulation results in a structured format
- export data for post-processing and reproducibility

Instead of only saving final figures, this workflow also stores:

- simulation parameters
- normalized spectra
- dielectric maps
- frequency-domain electric fields
- field visualization data

This makes it possible to re-plot or analyze the results without rerunning the full FDTD simulation.

---

---

## Physical Structure

The simulated structure is a **silver thin film with a flower-shaped air hole**.

The flower geometry is defined in polar coordinates using the following parameters:

- `rmin`: minimum radius of the flower pattern
- `rmax`: maximum radius of the flower pattern
- `alpha`: angular modulation parameter
- `beta`: phase or shape-control parameter
- `s`: sharpness / deformation parameter
- `Nf`: number of flower petals

The structure is periodic in the transverse plane and is defined by the simulation cell size.

The current implementation uses:

- **Material:** Silver (`Ag`) from `meep.materials`
- **Hole region:** Air (`epsilon = 1`)
- **Illumination:** Circularly polarized Gaussian source
- **Output quantities:**
  - transmission spectra
  - dielectric distribution
  - complex frequency-domain electric fields

---

## Features

### 1. Reference Simulation

The workflow first runs an empty-cell simulation to obtain the incident flux for normalization.

This step is necessary because the raw transmitted flux from the structure simulation must be normalized by the incident source spectrum.

---

### 2. Structure Simulation

The structure simulation is performed under both circular polarization states:

- **RCP:** right circularly polarized light
- **LCP:** left circularly polarized light

The normalized transmission spectra are then calculated as:

```text
T_RCP = transmitted_flux_RCP / incident_flux
T_LCP = transmitted_flux_LCP / incident_flux
