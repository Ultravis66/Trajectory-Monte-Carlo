# Trajectory-Monte-Carlo

A small, reproducible simulation that couples guidance (PN + pitch trim) with rigid-body dynamics. Aerodynamics are from tabulated Cl/Cd with interpolation; pitching moments use 
Cm(α) + control + rate damping. Includes event-based ground impact, Monte Carlo, targeting, and runtime reporting.

this is designed to run multiple cases and let the user know if you can hit the target within a given radius and within 2 meters.
