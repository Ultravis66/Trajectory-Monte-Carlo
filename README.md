# Trajectory-Monte-Carlo

A compact, reproducible quasi-6-DoF projectile simulation framework built for 
large-scale Monte Carlo analysis and machine-learning dataset generation. The model 
couples simplified rigid-body dynamics with proportional-navigation steering and 
pitch-trim control.

Aerodynamic forces are computed from tabulated Cl/Cd values using 2-D interpolation, 
and pitching moments include static Cm(α), control-induced moment, and rate damping. 
Quaternion attitude propagation, fixed-step RK4 integration, and event-based ground 
impact detection ensure stable and consistent trajectory evolution.

This tool enables running thousands (or tens of thousands) of randomized trajectories 
(targets, winds, etc.) to evaluate miss-distance statistics or generate ML-ready 
datasets. Users may replace the placeholder aerodynamic tables with high-fidelity 
data from CFD or experiments.

The main simulation script (`6DOF_Light_ML_Training.py`) is an updated version 
emphasizing speed, reproducibility, and scalable batch execution using a 
hand-coded RK4 integrator. It can be extended to run 100,000+ trajectories for 
training neural-network surrogate models.

Use `Plotting.py` to visualize the dataset and generate performance plots after 
running the simulation. Aerodynamic inputs in this repository are placeholder values; 
in practice these are typically replaced with CFD- or experimentally-derived tables.
