# Trajectory-Monte-Carlo

A compact, reproducible quasi-6-DoF projectile simulation framework built for large-scale Monte Carlo analysis and machine-learning dataset generation. The model couples simplified rigid-body dynamics with proportional-navigation steering and 
pitch-trim control.

Aerodynamic forces are computed from tabulated Cl/Cd values using 2-D interpolation, and pitching moments include static Cm(α), control-induced moment, and rate damping. 
Quaternion attitude propagation, fixed-step RK4 integration, and event-based ground impact detection ensure stable and consistent trajectory evolution.

This tool enables running thousands (or tens of thousands) of randomized trajectories (targets, winds, etc.) to evaluate miss-distance statistics or generate ML-ready datasets. Users may replace the placeholder/dummy aerodynamic tables with high-fidelity data from CFD or experiments.

The main simulation script (`6DOF_Light_ML_Training.py`) is an updated version emphasizing speed, reproducibility, and scalable batch execution using a hand-coded RK4 integrator. It can be extended to run 100,000+ trajectories for training neural-network surrogate models.

#EDIT: I have been playing around with the Aero Tables; current design space trim AoA is ~13° from the given Cm,alpha table. If you edit the tables and want to see what happens at higher Angles of Attack, the design space is capped at 16°, you will need to update the aero tables accordingly.

Companion repo found here:  https://github.com/Ultravis66/Guided-Trajectory-Neural-Net

Use `Plotting.py` to visualize the dataset and generate performance plots after running the simulation. Aerodynamic inputs in this repository are placeholder values; in practice these are typically replaced with CFD- or experimentally-derived tables.

## Features

- **Quasi-6-DoF rigid-body dynamics**
  - 13-state model (position, velocity, quaternion attitude, angular rates)
- **Guidance & control**
  - Proportional-navigation-style lateral steering  
  - Pitch-trim logic for altitude control  
- **Aerodynamics**
  - Tabulated Cl/Cd lookups with 2-D interpolation  
  - Cm(α) curve with control + damping contributions  
  - Optional CP→CG moment arm
- **Numerical integration**
  - Hand-written fixed-step RK4  
  - Stable quaternion normalization  
- **Monte Carlo engine**
  - Randomized target locations & wind vectors  
  - Batched execution for 1,000–100,000+ trajectories  
  - Progress reporting, statistics, timing  
- **Data export**
  - Summary CSV for machine learning  
  - Individual trajectory `.npz` files (states + time histories)
- **Plotting tools**
  - Miss-distance histograms  
  - CEP statistics  
  - Target hit/miss maps  
  - Trajectory visualizations  

---
## Author
if you use this work, please cite:
```
@misc{trajectoryMC2025,
author    = {Stolk, Mitchell},
title     = {Trajectory-Monte-Carlo: Quasi-6DoF Projectile Simulator and ML Dataset Generator},
year      = {2025},
publisher = {GitHub},
url       = {https://github.com/ultravis66/Trajectory-Monte-Carlo}
}
