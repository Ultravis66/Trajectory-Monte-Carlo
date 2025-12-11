# 6-DoF-Lite Guided Projectile - Monte Carlo for ML Dataset
# ML-READY VERSION: Fixed-size trajectory output for neural network training
#
# Author: Mitchell R. Stolk
# License: MIT
# Date: December 2025
#
# Output files:
#   - trajectory_ml_dataset.csv: Summary statistics for each run
#   - trajectories.npz: All trajectories in single file (inputs, trajectories, times)
#
# CSV columns:
#   INPUTS:  x_target, z_target, range_to_target, bearing_deg, wx, wz, wind_speed
#   OUTPUTS: x_impact, z_impact, tof, miss_distance, hit (bool)

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import multiprocessing as mp
import time
import csv
import os

# === CONFIGURATION ===
OUTPUT_CSV = "trajectory_ml_dataset.csv"
OUTPUT_TRAJ = "trajectories.npz"

# Monte Carlo settings
NUM_RUNS = 100000
RADIUS = 50
BATCH_SIZE = 500

# ML trajectory settings - FIXED SIZE OUTPUT
TRAJ_OUTPUT_DT = 0.05    # Save state every 50ms (20 Hz)
TRAJ_MAX_TIME = 10.0     # Max trajectory duration for ML
TRAJ_MAX_STEPS = int(TRAJ_MAX_TIME / TRAJ_OUTPUT_DT)  # 200 steps

# Initial conditions
x0, y0_alt, z0 = 0.0, 100.0, 0.0

# Target initial starting location
X_init, Y_init, Z_init = 85.0, 0.0, 0.0

# Initial Velocity
vx0, vy0, vz0 = 20.0, 0.0, 0.0
Wind_Speed = 5.0  # max wind speed m/s

# Guidance
K_nav = 20
baseline_trim = 12  # deg

# Aero/moment params
Cm_per_deg = 0.10
Cm_q = -20.0
Cl_p = -10.0
Cn_r = -20.0

# Geometry
D_ref = 0.05
S_ref = 0.25 * 3.141592653589793 * (D_ref**2)
c_ref = 0.50
b_ref = 0.15
I_body = (8.5e-4, 0.03, 0.03)
r_cp_cg = (0.03, 0.0, 0.0)
use_cp_lever_arm = True

# Integration
dt = 0.005        # 5ms integration timestep
t_max = 10.0      # Max simulation time (matches TRAJ_MAX_TIME)

HIT_THRESH = 5.0
EPS = 1e-12

# === AERO TABLES ===
AoA = np.array([0, 4, 6, 8, 12, 16], dtype=float)
canP = np.array([0, 4, 6, 8, 12, 16], dtype=float)

CdP = np.array([[0.35,0.40,0.50,0.65,1.10,1.80],
                [0.40,0.50,0.60,0.80,1.30,2.00],
                [0.45,0.55,0.65,0.85,1.40,2.10],
                [0.50,0.65,0.80,1.00,1.60,2.30],
                [0.60,0.75,0.90,1.10,1.70,2.35],
                [0.70,0.85,1.00,1.20,1.80,2.45]], dtype=float)

ClP = np.array([[0.00,1.10,1.75,2.50,3.90,5.50],
                [0.05,1.20,1.90,2.60,4.00,5.40],
                [0.20,1.25,1.95,2.70,4.10,5.30],
                [0.40,1.35,2.00,2.80,4.20,5.20],
                [0.50,1.40,2.05,2.70,4.10,5.10],
                [0.60,1.45,2.10,2.60,4.00,5.00]], dtype=float)

_cl_interp = RegularGridInterpolator((AoA, canP), ClP, bounds_error=False, fill_value=None)
_cd_interp = RegularGridInterpolator((AoA, canP), CdP, bounds_error=False, fill_value=None)

# Pre-compute inverse inertia matrix
I_mat = np.diag(I_body)
I_inv = np.linalg.inv(I_mat)

def lookup_cl_cd(alpha_deg, pitch_deg):
    a = np.clip(abs(alpha_deg), 0.0, 16.0)
    p = np.clip(abs(pitch_deg), 0.0, 16.0)
    pt = np.array([[a, p]])
    return _cl_interp(pt).item(), _cd_interp(pt).item()

alpha_table_full = np.array([0,4,6,8,12,16,25,45,60,90,180], dtype=float)
Cm_alpha_values = np.array([0,-0.10,-0.15,-0.30,-1,-2,-4,-8,-10,-8,-6], dtype=float)

def cm_alpha_interp(alpha_deg):
    return float(np.interp(abs(alpha_deg), alpha_table_full, Cm_alpha_values))

def quat_to_rotm(qw, qx, qy, qz):
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1-2*(qx*qx+qy*qy)]
    ], dtype=float)


def deriv(y, p):
    """Compute state derivative."""
    x, y_pos, z = y[0], y[1], y[2]
    vx, vy, vz = y[3], y[4], y[5]
    qw, qx, qy, qz = y[6], y[7], y[8], y[9]
    pr, qr, rr = y[10], y[11], y[12]
    
    rho, S, m, g = p['rho'], p['S'], p['m'], p['g']
    wx, wy, wz = p['wx'], p['wy'], p['wz']
    x_t, y_t, z_t = p['x_target'], p['y_target'], p['z_target']
    
    # Normalize quaternion
    qnorm = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    if qnorm < EPS:
        qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
    else:
        qw, qx, qy, qz = qw/qnorm, qx/qnorm, qy/qnorm, qz/qnorm
    
    Rbw = quat_to_rotm(qw, qx, qy, qz)
    Rwb = Rbw.T
    
    # Relative velocity
    vrel = np.array([vx-wx, vy-wy, vz-wz])
    V = np.sqrt(vrel[0]**2 + vrel[1]**2 + vrel[2]**2) + EPS
    v_hat = vrel / V
    
    # AoA
    Vb = Rwb @ vrel
    alpha_deg = np.degrees(np.arctan2(Vb[2], Vb[0]))
    
    # Guidance
    to_tgt = np.array([x_t - x, y_t - y_pos, z_t - z])
    dist = np.sqrt(to_tgt[0]**2 + to_tgt[1]**2 + to_tgt[2]**2)
    
    Cl_nav = 0.0
    lift_dir_nav = np.zeros(3)
    if dist > 0.1:
        u_tgt = to_tgt / dist
        dot_ut_vh = u_tgt[0]*v_hat[0] + u_tgt[1]*v_hat[1] + u_tgt[2]*v_hat[2]
        u_tgt_perp = u_tgt - dot_ut_vh * v_hat
        u_perp_norm = np.sqrt(u_tgt_perp[0]**2 + u_tgt_perp[1]**2 + u_tgt_perp[2]**2)
        if u_perp_norm > 1e-12:
            a_req = K_nav * V * (u_perp_norm / max(1.0, dist))
            Cl_req = 2.0 * m * a_req / (rho * V*V * S + EPS)
            Cl_nav = min(max(Cl_req, 0.0), 6.0)
            lift_dir_nav = u_tgt_perp / u_perp_norm
    
    # Pitch control
    horiz_dist = np.sqrt(to_tgt[0]**2 + to_tgt[2]**2) + EPS
    gamma_desired = np.arctan2(to_tgt[1], horiz_dist)
    gamma_actual = np.arctan2(vy, np.sqrt(vx*vx + vz*vz) + EPS)
    pitch_error = gamma_desired - gamma_actual
    pitch_cmd = baseline_trim + max(-8.0, min(8.0, np.degrees(pitch_error)*2.0))
    pitch_cmd = max(-16.0, min(16.0, pitch_cmd))
    
    # Aero coefficients
    Cl_basic, Cd_val = lookup_cl_cd(alpha_deg, pitch_cmd)
    
    # Drag
    F_drag = -0.5 * rho * S * Cd_val * V * vrel
    
    # Basic lift direction
    e_y = np.array([0.0, 1.0, 0.0])
    cross1 = np.cross(vrel, e_y)
    lift_basic_dir = np.cross(cross1, vrel)
    lb_norm = np.sqrt(lift_basic_dir[0]**2 + lift_basic_dir[1]**2 + lift_basic_dir[2]**2)
    if lb_norm > 1e-12:
        lift_basic_dir /= lb_norm
    else:
        lift_basic_dir = np.zeros(3)
    
    sign_basic = 1.0 if pitch_cmd >= 0 else -1.0
    F_lift_basic = sign_basic * 0.5 * rho * S * Cl_basic * V*V * lift_basic_dir
    F_lift_nav = 0.5 * rho * S * Cl_nav * V*V * lift_dir_nav
    
    # Total force
    F_aero_W = F_drag + F_lift_basic + F_lift_nav
    F_total = F_aero_W + np.array([0.0, -m*g, 0.0])
    ax, ay, az = F_total / m
    
    # Moments
    q_dyn = 0.5 * rho * V*V
    Cm_static = cm_alpha_interp(alpha_deg)
    Cm_ctrl = Cm_per_deg * pitch_cmd
    q_hat = (qr * c_ref) / (2.0*V + EPS)
    Cm_total = Cm_static + Cm_ctrl + Cm_q * q_hat
    My = q_dyn * S * c_ref * Cm_total
    
    p_hat = (pr * b_ref) / (2.0*V + EPS)
    r_hat = (rr * b_ref) / (2.0*V + EPS)
    Mx = q_dyn * S * b_ref * (Cl_p * p_hat)
    Mz = q_dyn * S * b_ref * (Cn_r * r_hat)
    
    M_body = np.array([Mx, My, Mz])
    
    if use_cp_lever_arm:
        F_aero_B = Rwb @ F_aero_W
        M_arm = np.cross(np.array(r_cp_cg), F_aero_B)
        M_body = M_body + M_arm
    
    # Rotational dynamics
    omega = np.array([pr, qr, rr])
    Iomega = I_mat @ omega
    domega = I_inv @ (M_body - np.cross(omega, Iomega))
    
    # Quaternion kinematics
    qdot = np.array([
        0.5 * (-pr*qx - qr*qy - rr*qz),
        0.5 * ( pr*qw + rr*qy - qr*qz),
        0.5 * ( qr*qw - rr*qx + pr*qz),
        0.5 * ( rr*qw + qr*qx - pr*qy)
    ])
    
    return np.array([vx, vy, vz, ax, ay, az, 
                     qdot[0], qdot[1], qdot[2], qdot[3],
                     domega[0], domega[1], domega[2]])


def rk4_integrate_fixed_output(y0, params, dt, t_max, output_dt, max_steps):
    """
    RK4 integration with FIXED-SIZE output for ML.
    
    Returns:
        tof: actual time of flight (when ground hit)
        y_final: final state at impact
        traj_fixed: (max_steps, 13) array - fixed size, padded with final state if needed
        times_fixed: (max_steps,) array - fixed time grid
    """
    y = y0.copy()
    t = 0.0
    
    # Fixed output time grid
    times_fixed = np.linspace(0, (max_steps - 1) * output_dt, max_steps)
    traj_fixed = np.zeros((max_steps, 13), dtype=np.float64)
    
    # Store initial state
    traj_fixed[0] = y.copy()
    
    next_output_idx = 1
    next_output_time = times_fixed[next_output_idx] if next_output_idx < max_steps else np.inf
    
    hit_ground = False
    y_at_impact = y.copy()
    
    while t < t_max and next_output_idx < max_steps:
        # Check ground impact
        if y[1] <= 0.0:
            hit_ground = True
            y_at_impact = y.copy()
            break
        
        # RK4 step
        k1 = deriv(y, params)
        k2 = deriv(y + 0.5*dt*k1, params)
        k3 = deriv(y + 0.5*dt*k2, params)
        k4 = deriv(y + dt*k3, params)
        
        y = y + (dt/6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
        t += dt
        
        # Save to fixed grid when we cross output times
        while next_output_idx < max_steps and t >= times_fixed[next_output_idx]:
            # Linear interpolation to exact output time (optional, can just use current state)
            traj_fixed[next_output_idx] = y.copy()
            next_output_idx += 1
    
    # Record actual time of flight
    tof = t
    y_final = y.copy() if not hit_ground else y_at_impact
    
    # Pad remaining timesteps with final state (for trajectories that ended early)
    if next_output_idx < max_steps:
        for i in range(next_output_idx, max_steps):
            traj_fixed[i] = y_final.copy()
    
    return tof, y_final, traj_fixed, times_fixed


def run_single_sim(args):
    """Run one simulation, return summary + fixed-size trajectory."""
    params, y0, dt, t_max, output_dt, max_steps, x_tgt, z_tgt, wx, wz = args
    
    params = dict(params)
    params['x_target'] = x_tgt
    params['z_target'] = z_tgt
    params['wx'] = wx
    params['wz'] = wz
    
    try:
        tof, y_final, traj_fixed, times_fixed = rk4_integrate_fixed_output(
            y0, params, dt, t_max, output_dt, max_steps
        )
        
        xf, zf = y_final[0], y_final[2]
        miss = np.sqrt((xf - x_tgt)**2 + (zf - z_tgt)**2)
        
        # Summary for CSV
        summary = {
            'x_target': x_tgt,
            'z_target': z_tgt,
            'range_to_target': np.sqrt(x_tgt**2 + z_tgt**2),
            'bearing_deg': np.degrees(np.arctan2(z_tgt, x_tgt)),
            'wx': wx,
            'wz': wz,
            'wind_speed': np.sqrt(wx**2 + wz**2),
            'x_impact': xf,
            'z_impact': zf,
            'tof': tof,
            'miss_distance': miss,
            'hit': 1 if miss < HIT_THRESH else 0
        }
        
        # ML input vector (7 features)
        input_vec = np.array([
            x_tgt,
            z_tgt,
            np.sqrt(x_tgt**2 + z_tgt**2),
            np.degrees(np.arctan2(z_tgt, x_tgt)),
            wx,
            wz,
            np.sqrt(wx**2 + wz**2)
        ], dtype=np.float64)
        
        return {
            'summary': summary,
            'inputs': input_vec,
            'trajectory': traj_fixed,  # Shape: (max_steps, 13)
            'tof': tof
        }
        
    except Exception as e:
        return None


def main():
    print("="*60)
    print("6-DoF MONTE CARLO FOR ML TRAINING")
    print("="*60)
    print(f"Runs: {NUM_RUNS:,}")
    print(f"Target radius: {RADIUS} m")
    print(f"Integration dt: {dt*1000:.1f} ms")
    print(f"Output dt: {TRAJ_OUTPUT_DT*1000:.1f} ms")
    print(f"Trajectory steps: {TRAJ_MAX_STEPS} (fixed size)")
    print(f"Output CSV: {OUTPUT_CSV}")
    print(f"Output trajectories: {OUTPUT_TRAJ}")
    print("="*60)
    
    rng = np.random.default_rng(42)
    
    # Generate random targets
    angles = rng.uniform(0.0, 2*np.pi, size=NUM_RUNS)
    rads = rng.uniform(0.0, RADIUS, size=NUM_RUNS)
    x_targets = X_init + rads * np.cos(angles)
    z_targets = Z_init + rads * np.sin(angles)
    
    # Generate random wind
    wind_speeds = rng.uniform(0.0, Wind_Speed, size=NUM_RUNS)
    wind_dirs = rng.uniform(0.0, 2*np.pi, size=NUM_RUNS)
    wx_vals = wind_speeds * np.cos(wind_dirs)
    wz_vals = wind_speeds * np.sin(wind_dirs)
    
    # Base params
    params = {
        'rho': 1.225, 'S': S_ref, 'm': 1.50, 'g': 9.81,
        'wx': 0.0, 'wy': 0.0, 'wz': 0.0,
        'K_nav': K_nav, 'Cl_max': 6.0,
        'x_target': X_init, 'y_target': Y_init, 'z_target': Z_init,
        'c_ref': c_ref, 'b_ref': b_ref, 'I_body': I_body,
        'Cm_per_deg': Cm_per_deg, 'Cm_q': Cm_q,
        'Cl_p': Cl_p, 'Cn_r': Cn_r,
        'r_cp_cg': r_cp_cg, 'use_cp_lever_arm': use_cp_lever_arm,
    }
    
    y0 = np.zeros(13)
    y0[0:6] = [x0, y0_alt, z0, vx0, vy0, vz0]
    y0[6:10] = [1.0, 0.0, 0.0, 0.0]
    y0[10:13] = [0.0, 0.0, 0.0]
    
    # Build args list
    args_list = [
        (params, y0.copy(), dt, t_max, TRAJ_OUTPUT_DT, TRAJ_MAX_STEPS,
         x_targets[i], z_targets[i], wx_vals[i], wz_vals[i])
        for i in range(NUM_RUNS)
    ]
    
    fieldnames = ['x_target', 'z_target', 'range_to_target', 'bearing_deg',
                  'wx', 'wz', 'wind_speed',
                  'x_impact', 'z_impact', 'tof', 'miss_distance', 'hit']
    
    # Initialize CSV
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
    
    # Pre-allocate arrays for trajectory data
    all_inputs = np.zeros((NUM_RUNS, 7), dtype=np.float64)
    all_trajectories = np.zeros((NUM_RUNS, TRAJ_MAX_STEPS, 13), dtype=np.float64)
    all_tofs = np.zeros(NUM_RUNS, dtype=np.float64)
    
    n_workers = max(1, mp.cpu_count() - 1)
    print(f"Using {n_workers} CPU cores")
    print("-"*60)
    
    t_start = time.perf_counter()
    completed = 0
    failed = 0
    all_misses = []
    
    with mp.Pool(processes=n_workers) as pool:
        for batch_start in range(0, NUM_RUNS, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, NUM_RUNS)
            batch_args = args_list[batch_start:batch_end]
            
            results = pool.map(run_single_sim, batch_args)
            
            valid_results = [r for r in results if r is not None]
            failed += len(results) - len(valid_results)
            
            # Append summaries to CSV
            with open(OUTPUT_CSV, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerows([r['summary'] for r in valid_results])
            
            # Store trajectory data in pre-allocated arrays
            for i, r in enumerate(valid_results):
                idx = batch_start + i
                if idx < NUM_RUNS:
                    all_inputs[idx] = r['inputs']
                    all_trajectories[idx] = r['trajectory']
                    all_tofs[idx] = r['tof']
            
            # Progress tracking
            completed += len(valid_results)
            all_misses.extend([r['summary']['miss_distance'] for r in valid_results])
            
            elapsed = time.perf_counter() - t_start
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (NUM_RUNS - completed) / rate if rate > 0 else 0
            
            print(f"Progress: {completed:,}/{NUM_RUNS:,} "
                  f"({100*completed/NUM_RUNS:.1f}%) | "
                  f"Rate: {rate:.1f} runs/s | ETA: {eta:.0f}s | "
                  f"Avg miss: {np.mean(all_misses):.2f}m")
    
    # Save all trajectories to single .npz file
    print("\nSaving trajectory data...")
    times_grid = np.linspace(0, (TRAJ_MAX_STEPS - 1) * TRAJ_OUTPUT_DT, TRAJ_MAX_STEPS)
    
    np.savez_compressed(
        OUTPUT_TRAJ,
        inputs=all_inputs,              # (NUM_RUNS, 7)
        trajectories=all_trajectories,  # (NUM_RUNS, TRAJ_MAX_STEPS, 13)
        tofs=all_tofs,                  # (NUM_RUNS,)
        times=times_grid,               # (TRAJ_MAX_STEPS,) - same for all
        # Metadata
        state_labels=np.array(['x', 'y', 'z', 'vx', 'vy', 'vz', 
                               'qw', 'qx', 'qy', 'qz', 'p', 'q', 'r']),
        input_labels=np.array(['x_target', 'z_target', 'range_to_target', 
                               'bearing_deg', 'wx', 'wz', 'wind_speed'])
    )
    
    total_time = time.perf_counter() - t_start
    misses = np.array(all_misses)
    
    # File sizes
    csv_size = os.path.getsize(OUTPUT_CSV) / (1024*1024)  # MB
    npz_size = os.path.getsize(OUTPUT_TRAJ) / (1024*1024)  # MB
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Successful runs: {completed:,}")
    print(f"Failed runs: {failed}")
    print(f"Rate: {completed/total_time:.1f} runs/s")
    print("-"*60)
    print("MISS DISTANCE STATISTICS:")
    print(f"  Mean:   {np.mean(misses):.2f} m")
    print(f"  Std:    {np.std(misses):.2f} m")
    print(f"  Min:    {np.min(misses):.2f} m")
    print(f"  Max:    {np.max(misses):.2f} m")
    print(f"  Hit rate (<{HIT_THRESH}m): {100*np.sum(misses < HIT_THRESH)/len(misses):.1f}%")
    print("-"*60)
    print("OUTPUT FILES:")
    print(f"  {OUTPUT_CSV}: {csv_size:.1f} MB")
    print(f"  {OUTPUT_TRAJ}: {npz_size:.1f} MB")
    print("-"*60)
    print("TRAJECTORY DATA SHAPES:")
    print(f"  inputs:       ({NUM_RUNS}, 7)")
    print(f"  trajectories: ({NUM_RUNS}, {TRAJ_MAX_STEPS}, 13)")
    print(f"  tofs:         ({NUM_RUNS},)")
    print(f"  times:        ({TRAJ_MAX_STEPS},)")
    print("="*60)
    print("\nReady for ML training!")
    print("Load with: data = np.load('trajectories.npz')")
    print("           X = data['inputs']")
    print("           Y = data['trajectories']")


if __name__ == '__main__':
    mp.freeze_support()

    main()

