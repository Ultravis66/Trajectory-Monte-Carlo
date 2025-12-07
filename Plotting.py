# plot_results.py
# Visualization for 6-DoF Monte Carlo trajectory results
#
# Author: Mitchell R. Stolk
# Date: December 2025

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# ==========================
# Load data & basic stats
# ==========================
df = pd.read_csv('trajectory_ml_dataset.csv')

print("="*50)
print("DATASET SUMMARY")
print("="*50)
print(f"Total runs: {len(df)}")
mean_miss = df['miss_distance'].mean()
std_miss  = df['miss_distance'].std()
print(f"Mean miss: {mean_miss:.2f} m")
print(f"Std miss:  {std_miss:.2f} m")

# CEP-style stats
cep50 = np.percentile(df['miss_distance'], 50)
cep90 = np.percentile(df['miss_distance'], 90)
print(f"CEP50 (50% miss radius): {cep50:.2f} m")
print(f"CEP90 (90% miss radius): {cep90:.2f} m")

if 'hit' in df.columns:
    print(f"Hit rate (<2m flag): {100*df['hit'].mean():.2f}%")
print("="*50)

# ==========================
# PLOT 1: Miss Distance Histogram
# ==========================
plt.figure(figsize=(10, 6))
plt.hist(df['miss_distance'], bins=50, edgecolor='black', alpha=0.7)
plt.axvline(mean_miss, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_miss:.1f} m')
plt.axvline(cep50, color='orange', linestyle='--', linewidth=2, label=f'CEP50: {cep50:.1f} m')
plt.axvline(cep90, color='green', linestyle='--', linewidth=2, label=f'CEP90: {cep90:.1f} m')
plt.xlabel('Miss Distance (m)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Miss Distance Distribution', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('01_miss_histogram.png', dpi=150)
plt.show()

# ==========================
# PLOT 2: Impact Footprint (World Frame) + Convex Hull
# ==========================
fig, ax = plt.subplots(figsize=(12, 10))

impacts_x = df['x_impact'].values
impacts_z = df['z_impact'].values
impact_points = np.column_stack([impacts_x, impacts_z])

# Convex hull for overall impact envelope (if enough points)
if len(impact_points) >= 3:
    hull_all = ConvexHull(impact_points)
    hull_pts = impact_points[hull_all.vertices]
    ax.fill(hull_pts[:,0], hull_pts[:,1], alpha=0.2, color='steelblue', label='Impact Envelope')

# Scatter impacts colored by miss
sc = ax.scatter(impacts_x, impacts_z, c=df['miss_distance'],
                cmap='RdYlGn_r', alpha=0.5, s=10)
cbar = plt.colorbar(sc, label='Miss Distance (m)')

# Launch point
ax.scatter(0, 0, marker='^', s=200, c='black', zorder=5, label='Launch')

ax.set_xlabel('X (m) - Downrange', fontsize=12)
ax.set_ylabel('Z (m) - Crossrange', fontsize=12)
ax.set_title('Impact Footprint (World Frame)', fontsize=14)
ax.legend(loc='upper left')
ax.set_aspect('equal', adjustable='box')
ax.grid(True, alpha=0.4)
# ---- Add minor ticks + minor grid ----
ax.minorticks_on()
ax.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.savefig('02_impact_footprint.png', dpi=150)
plt.show()

# ==========================
# PLOT 3: Target-Relative Error Cloud (CEP View)
# ==========================
fig, ax = plt.subplots(figsize=(10, 10))

dx = df['x_impact'] - df['x_target']
dz = df['z_impact'] - df['z_target']

ax.scatter(dx, dz, alpha=0.3, s=8)

# Draw CEP50 and CEP90 circles
r_50 = cep50
r_90 = cep90
theta = np.linspace(0, 2*np.pi, 200)
ax.plot(r_50 * np.cos(theta), r_50 * np.sin(theta), 'orange', linestyle='--', label=f'CEP50: {r_50:.1f} m')
ax.plot(r_90 * np.cos(theta), r_90 * np.sin(theta), 'green', linestyle='--', label=f'CEP90: {r_90:.1f} m')

ax.axhline(0, color='k', linewidth=0.8)
ax.axvline(0, color='k', linewidth=0.8)

ax.set_xlabel('ΔX = X_impact - X_target (m)', fontsize=12)
ax.set_ylabel('ΔZ = Z_impact - Z_target (m)', fontsize=12)
ax.set_title('Target-Relative Error Cloud (Impact - Target)', fontsize=14)
ax.legend(loc='upper right')
ax.set_aspect('equal', adjustable='box')
ax.grid(True, alpha=0.4)
# ---- Add minor ticks + minor grid ----
ax.minorticks_on()
ax.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.savefig('03_error_cloud_target_relative.png', dpi=150)
plt.show()

# ==========================
# PLOT 4: Miss Distance vs Wind Speed
# ==========================
if 'wind_speed' in df.columns:
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(df['wind_speed'], df['miss_distance'], alpha=0.3, s=8)

    # Binned trend line
    bins = np.linspace(df['wind_speed'].min(), df['wind_speed'].max(), 15)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_means = []
    for i in range(len(bins)-1):
        mask = (df['wind_speed'] >= bins[i]) & (df['wind_speed'] < bins[i+1])
        if mask.sum() > 0:
            bin_means.append(df.loc[mask, 'miss_distance'].mean())
        else:
            bin_means.append(np.nan)

    ax.plot(bin_centers, bin_means, 'r-', linewidth=2.5, label='Average Miss')

    ax.set_xlabel('Wind Speed (m/s)', fontsize=12)
    ax.set_ylabel('Miss Distance (m)', fontsize=12)
    ax.set_title('Miss Distance vs Wind Speed', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    # ---- Add minor ticks + minor grid ----
    ax.minorticks_on()
    ax.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig('04_miss_vs_wind.png', dpi=150)
    plt.show()

# ==========================
# PLOT 5: Target Locations Colored by Hit/Miss
# ==========================
fig, ax = plt.subplots(figsize=(12, 10))

# Extract arrays
xt = df['x_target']
zt = df['z_target']
hit = df['hit']

# Colors
colors = np.where(hit == 1, 'red', 'blue')
labels = {0: 'Miss', 1: 'Hit'}

# Scatter plot
sc = ax.scatter(xt, zt, c=colors, alpha=0.6, s=40, edgecolors='k', linewidth=0.5)

# Legend
for value, label in labels.items():
    ax.scatter([], [], c=('red' if value == 1 else 'blue'),
               s=60, edgecolor='k', linewidth=0.5, label=label)

ax.set_xlabel('X Target (m) - Downrange', fontsize=12)
ax.set_ylabel('Z Target (m) - Crossrange', fontsize=12)
ax.set_title('Target Locations Colored by Hit/Miss', fontsize=14)
ax.legend(loc='upper right')

# Major grid
ax.grid(True, alpha=0.3)

# ---- Add minor ticks + minor grid ----
ax.minorticks_on()
ax.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.5)

ax.set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.savefig('05_target_hit_map.png', dpi=150)
plt.show()

print("Total hits:", df['hit'].sum())
print("Any hits?", (df['hit'] == 1).any())
print("Min miss:", df['miss_distance'].min())
print("\n" + "="*50)
print("All plots saved!")
print("="*50)