# Trajectory-Monte-Carlo

A compact, reproducible quasi-6-DoF projectile simulation framework built for large-scale Monte Carlo analysis and machine-learning dataset generation. The model couples simplified rigid-body dynamics with proportional-navigation-style steering and pitch-trim control.

Aerodynamic forces are computed from tabulated Cl/Cd data via 2-D interpolation, and pitching moments include static Cm(α), control-induced moment, and rate damping. Quaternion attitude propagation, RK4 numerical integration, and event-based ground-impact termination ensure stable trajectory evolution.

This tool enables running thousands of randomized trajectories (targets, winds, etc.) to evaluate miss distance statistics or generate ML-ready datasets. Users can replace the placeholder aerodynamic tables with real aero data from CFD or experiments.

`6DOF_Light_ML_Training.py` is a new version emphasizing speed, reproducibility, and scalable batch simulation using a hand-coded RK4 integrator, ran 100,000 times.

use Plotting.py to plot the data after running the simulation.
