# 6-DoF-Lite Guided Projectile Demo
# Translation forces remain in WORLD frame (drag + altitude lift + PN steering lift).
# Rotation is added (quaternion + body rates) with a minimal moment model:
#   M_body = [Cl_p * p_hat, Cm(alpha, delta, q_hat), Cn_r * r_hat] * q_dyn * S * ref
# plus optional CP->CG lever-arm moment from aero force rotated to body frame.
#
# State (13): [x, y, z, vx, vy, vz, qw, qx, qy, qz, p, q, r]
#
# Notes:
# - Positive pitch_cmd produces moment sign via Cm_per_deg. If pitch goes the wrong way,
#   flip Cm_per_deg sign.
# - Quaternion is normalized inside RHS for robustness.
# - Event-based ground impact termination; miss computed at event state.

# === 0) User-configurable settings ===
x_target     = 85         # target X [m]
y_target     = 0          # target altitude [m]
z_target     = 0          # target Z [m]

vx0, vy0, vz0 = 20, 0, 0   # initial velocity [m/s]
y0_alt        = 100        # initial altitude [m]

K_nav         = 20         # navigation gain
baseline_trim = 12         # constant pitch trim [deg]

HIT_THRESH = 2.0  # meters

# Aerodynamic/moment params
Cm_per_deg    = 0.10       # control slope [per deg]; flip sign if pitch response inverted
Cm_q          = -20.0      # pitch-rate damping coefficient (dimensionless)
Cl_p          = -10.0       # roll-rate damping coefficient
Cn_r          = -20.0      # yaw-rate damping coefficient

# Geometry / inertia
D_ref   = 0.05             # reference diameter [m]
S_ref   = 0.25 * 3.141592653589793 * (D_ref**2)   # keep consistent with prior scripts
c_ref   = 0.50             # ref chord [m] (for pitching moment)
b_ref   = 0.15             # ref span [m] (for roll/yaw moments)
I_body  = (8.5e-4, 0.03, 0.03)   # (Ix, Iy, Iz) [kg m^2]
r_cp_cg = (0.03, 0.0, 0.0)       # CP ahead of CG by 3 cm in body-x (optional)
use_cp_lever_arm = True

# Integration settings
dt    = 0.001              # time step [s]
t_max = 8.0                # max sim time [s]


# === 1) Imports & surrogate setup ===
import numpy as np
from scipy.integrate import solve_ivp
import multiprocessing as mp
import matplotlib.pyplot as plt
import time
from scipy.interpolate import RegularGridInterpolator

EPS = 1e-12

# AoA/deflection grids (nonnegative AoA table; we mirror by |alpha|)
AoA  = np.array([0, 4, 6, 8, 12, 16], dtype=float)     # deg
canP = np.array([0, 4, 6, 8, 12, 16], dtype=float)     # deg (canard/trim deflection)

# 6x6 tables from MATLAB (deterministic surf)
CdP = np.array([[0.35,0.40,0.50,0.65,1.10,1.80],
                [0.35,0.45,0.60,0.80,1.30,2.00],
                [0.45,0.55,0.65,0.90,1.40,2.10],
                [0.50,0.70,0.85,1.00,1.60,2.30],
                [0.60,0.75,0.90,1.10,1.70,2.35],
                [0.70,0.85,1.00,1.20,1.80,2.45]], dtype=float)

ClP = np.array([[0.00,1.00,1.80,2.50,4.00,5.50],
                [0.05,1.13,1.80,2.60,4.00,5.40],
                [0.20,1.20,1.95,2.80,4.10,5.35],
                [0.40,1.20,1.90,2.70,4.10,5.36],
                [0.50,1.25,1.90,2.70,4.15,5.37],
                [0.60,1.20,1.80,2.60,4.10,5.30]], dtype=float)

# Deterministic 2D interpolators (faster & cleaner than GP)
_cl_interp = RegularGridInterpolator((AoA, canP), ClP, bounds_error=False, fill_value=None)
_cd_interp = RegularGridInterpolator((AoA, canP), CdP, bounds_error=False, fill_value=None)

def lookup_cl_cd(alpha_deg, pitch_deg):
    a = np.clip(abs(alpha_deg), AoA.min(), AoA.max())
    p = np.clip(abs(pitch_deg), canP.min(), canP.max())
    pt = np.array([[a, p]])
    Cl = _cl_interp(pt).item()   # scalar
    Cd = _cd_interp(pt).item()   # scalar
    return Cl, Cd

# 1D Cm(alpha) table (use your smaller, plausible values)
alpha_table_full = np.array([0,4,6,8,12,16,25,45,60,90,180], dtype=float)
Cm_alpha_values  = np.array([0,-1,-.4,-1.0,-2.0,-5.0,-8.0,-10.0,-12.0,-8.0,-5.0], dtype=float)

def cm_alpha_interp(alpha_deg):
    a = np.clip(abs(alpha_deg), alpha_table_full.min(), alpha_table_full.max())
    # Linear 1D interp is fine here
    return float(np.interp(a, alpha_table_full, Cm_alpha_values))

# Quaternion
def quat_to_rotm(q):
    # q = [qw,qx,qy,qz], unit assumed
    qw, qx, qy, qz = q
    # Rotation matrix body->world
    R = np.array([
        [1-2*(qy*qy+qz*qz),   2*(qx*qy - qz*qw),   2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),   1-2*(qx*qx+qz*qz),   2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),   2*(qy*qz + qx*qw),   1-2*(qx*qx+qy*qy)]
    ], dtype=float)
    return R

def omega_matrix(omega):
    p, q, r = omega
    return np.array([
        [0.0, -p,  -q,  -r],
        [p,   0.0,  r,  -q],
        [q,  -r,   0.0,  p],
        [r,   q,  -p,   0.0]
    ], dtype=float)

# Dynamics derivative (6-DoF-lite)

def deriv(t, y, params):
    # unpack state
    x, y_pos, z = y[0], y[1], y[2]
    vx, vy, vz  = y[3], y[4], y[5]
    qw, qx, qy, qz = y[6], y[7], y[8], y[9]
    p, q_rate, r   = y[10], y[11], y[12]

    # Extract parameters
    rho, S, m, g = params['rho'], params['S'], params['m'], params['g']
    wx, wy, wz   = params['wx'], params['wy'], params['wz']
    x_t, y_t, z_t = params['x_target'], params['y_target'], params['z_target']
    Cl_max       = params['Cl_max']
    K_nav        = params['K_nav']
    c_ref        = params['c_ref']
    b_ref        = params['b_ref']
    I_body       = params['I_body']
    Cm_per_deg   = params['Cm_per_deg']
    Cm_q         = params['Cm_q']
    Cl_p         = params['Cl_p']
    Cn_r         = params['Cn_r']
    r_cp_cg      = params['r_cp_cg']
    use_arm      = params['use_cp_lever_arm']

    # Normalize quaternion for robustness
    q = np.array([qw,qx,qy,qz], dtype=float)
    qn = np.linalg.norm(q)
    if qn < EPS:
        q = np.array([1.0,0.0,0.0,0.0])
    else:
        q = q / qn
    Rbw = quat_to_rotm(q)            # body -> world
    Rwb = Rbw.T                      # world -> body

    # relative velocity (world)
    v_rel_vec = np.array([vx-wx, vy-wy, vz-wz])
    V = np.linalg.norm(v_rel_vec) + EPS
    v_hat = v_rel_vec / V

    # AoA for lookup (using body-frame longitudinal plane)
    Vb = Rwb @ v_rel_vec
    alpha_deg = np.degrees(np.arctan2(Vb[2], Vb[0]))

    # --- Guidance toward target (PN-like lateral) ---
    to_tgt = np.array([x_t - x, y_t - y_pos, z_t - z])
    dist_to_tgt = np.linalg.norm(to_tgt)

    Cl_nav = 0.0
    lift_dir_nav = np.zeros(3)
    if dist_to_tgt > 0.1:
        u_tgt = to_tgt / dist_to_tgt
        u_tgt_perp = u_tgt - np.dot(u_tgt, v_hat) * v_hat
        u_perp_norm = np.linalg.norm(u_tgt_perp)
        if u_perp_norm > 1e-12:
            a_req  = K_nav * V * (u_perp_norm / max(1.0, dist_to_tgt))
            Cl_req = 2.0 * m * a_req / (rho * V**2 * S + EPS)
            Cl_nav = float(np.clip(Cl_req, 0.0, Cl_max))
            lift_dir_nav = u_tgt_perp / u_perp_norm

    # --- Pitch control for altitude (maps to canard deflection) ---
    gamma_desired = np.arctan2(to_tgt[1], np.linalg.norm([to_tgt[0], to_tgt[2]]) + EPS)
    gamma_actual  = np.arctan2(vy, np.sqrt(vx**2 + vz**2) + EPS)
    pitch_error   = gamma_desired - gamma_actual

    pitch_cmd = baseline_trim + np.clip(np.degrees(pitch_error)*2.0, -8.0, 8.0)
    pitch_cmd = float(np.clip(pitch_cmd, -16.0, 16.0))

    # --- Aero coefficients from tables ---
    Cl_basic, Cd_val = lookup_cl_cd(alpha_deg, pitch_cmd)
    # Drag
    F_drag = -0.5 * rho * S * Cd_val * V * v_rel_vec
    # Basic lift direction: exactly ⟂ to velocity and in the (v, +y) plane
    e_y = np.array([0.0, 1.0, 0.0])
    lift_basic_dir = np.cross(np.cross(v_rel_vec, e_y), v_rel_vec)
    lb_norm = np.linalg.norm(lift_basic_dir)
    if lb_norm > 1e-12:
        lift_basic_dir /= lb_norm
    else:
        lift_basic_dir = np.zeros(3)

    # Apply sign from pitch_cmd (allows down-lift)
    sign_basic = 1.0 if pitch_cmd >= 0 else -1.0
    F_lift_basic = sign_basic * 0.5 * rho * S * Cl_basic * V**2 * lift_basic_dir

    # Navigation lift (steering)
    F_lift_nav = 0.5 * rho * S * Cl_nav * V**2 * lift_dir_nav

    # Total aero + gravity (world)
    F_aero_W  = F_drag + F_lift_basic + F_lift_nav
    F_total_W = F_aero_W + np.array([0.0, -m*params['g'], 0.0])

    # Translational accelerations (world)
    ax, ay, az = F_total_W / m

    # --- Moments in BODY frame ---
    q_dyn = 0.5 * rho * V**2

    # Static Cm(alpha) + control + rate damping
    Cm_static = cm_alpha_interp(alpha_deg)
    Cm_ctrl   = Cm_per_deg * pitch_cmd
    q_hat     = (q_rate * c_ref) / (2.0*V + EPS)
    Cm_total  = Cm_static + Cm_ctrl + Cm_q * q_hat

    My_coeff  = Cm_total
    My = q_dyn * S * c_ref * My_coeff

    # Roll/yaw damping moments
    p_hat = (p * b_ref) / (2.0*V + EPS)
    r_hat = (r * b_ref) / (2.0*V + EPS)
    Mx = q_dyn * S * b_ref * (Cl_p * p_hat)
    Mz = q_dyn * S * b_ref * (Cn_r * r_hat)

    M_body = np.array([Mx, My, Mz])

    # Optional CP->CG lever-arm moment from aero forces
    if use_arm:
        F_aero_B = Rwb @ F_aero_W
        M_arm_B  = np.cross(np.array(r_cp_cg), F_aero_B)
        M_body  += M_arm_B

    # Rotational EOM: I * domega = M - omega x (I omega)
    I = np.diag(I_body)
    omega = np.array([p, q_rate, r])
    Iomega = I @ omega
    domega = np.linalg.solve(I, (M_body - np.cross(omega, Iomega)))

    # Quaternion kinematics
    qdot = 0.5 * (omega_matrix(omega) @ q)

    # Return state derivative
    return np.concatenate(([vx, vy, vz], [ax, ay, az], qdot, domega))

# === 2b) Ground-impact event ===

def ground_event(t, y):
    return y[1]     # altitude

ground_event.terminal  = True
ground_event.direction = -1

# Simulation runner

def run_simulation(params, y0, t_span, dt):
    t_eval = np.arange(t_span[0], t_span[1] + dt, dt)
    sol = solve_ivp(
        lambda t, y: deriv(t, y, params),
        t_span, y0, method='RK45',
        max_step=dt, rtol=1e-6, atol=1e-9,
        events=ground_event, t_eval=t_eval
    )

    if sol.status == 1 and sol.t_events and sol.t_events[0].size > 0:
        t_hit = sol.t_events[0][0]
        y_hit = sol.y_events[0][0]
        xf, yf, zf = y_hit[0], y_hit[1], y_hit[2]
    else:
        t_hit = sol.t[-1]
        xf, yf, zf = sol.y[0, -1], sol.y[1, -1], sol.y[2, -1]

    miss = float(np.linalg.norm([xf - params['x_target'],
                                 yf - params['y_target'],
                                 zf - params['z_target']]))

    print(f"Impact @ (x={xf:.2f}, y={yf:.2f}, z={zf:.2f})  TOF={t_hit:.3f}s")
    return miss, sol

# === 4) Main: integrate & plot ===
if __name__ == '__main__':
    mp.freeze_support()  # only needed on Windows

    # Reproducibility
    rng = np.random.default_rng(42)

    # MONTE CARLO CONFIGURATION
    num_runs  = 50        # Number of Monte Carlo simulations to run
    radius    = 10       # Max radius for random targets [m]

    t_span = (0.0, t_max)

    # Build params dict
    params = {
        'rho':   1.2250,                                  # air density
        'S':     S_ref,                                    # ref area
        'm':     1.50,                                     # mass [kg]
        'g':     9.81,
        'wx':    0.0, 'wy': 0.0, 'wz': 0.0,               # wind [m/s]
        'K_nav': K_nav,
        'Cl_max': 6.0,
        'x_target': x_target, 'y_target': y_target, 'z_target': z_target,
        'c_ref': c_ref, 'b_ref': b_ref,
        'I_body': I_body,
        'Cm_per_deg': Cm_per_deg,
        'Cm_q': Cm_q,
        'Cl_p': Cl_p,
        'Cn_r': Cn_r,
        'r_cp_cg': r_cp_cg,
        'use_cp_lever_arm': use_cp_lever_arm,
    }

    # Initial state (13)
    y0 = np.zeros(13)
    y0[0:6]  = [0.0, y0_alt, 0.0, vx0, vy0, vz0]
    y0[6:10] = [1.0, 0.0, 0.0, 0.0]  # quaternion (w,x,y,z)
    y0[10:13]= [0.0, 0.0, 0.0]       # body rates p,q,r

    # Baseline run
    print("Running baseline simulation...")
    miss, sol = run_simulation(params, y0, t_span, dt)
    print(f"Baseline target @ ({x_target}, {y_target}, {z_target}) → Miss: {miss:.2f} m")

    # Plot baseline trajectory
    x, y, z = sol.y[0], sol.y[1], sol.y[2]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, z, y, linewidth=2, label='Trajectory')
    ax.scatter(0, 0, y0_alt, marker='o', s=100, label='Launch')
    hit_color = 'red' if miss < HIT_THRESH else 'grey'
    ax.scatter(x_target, z_target, y_target, marker='X', s=200, c=hit_color, label='Target')
    ax.scatter(x[-1], z[-1], y[-1], marker='*', s=150, label='Impact')
    ax.set_xlim(0, max(150, x_target+25))
    ax.set_ylim(z_target-75, z_target+75)
    ax.set_zlim(0, max(110, y0_alt+10))
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_zlabel('Altitude (m)')
    ax.set_title(f'Baseline Guided Trajectory (Miss: {miss:.2f} m)')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()

    # Monte Carlo: random targets around (x_target, z_target)
    print("\nRunning Monte Carlo simulations...")
    angles = rng.uniform(0.0, 2*np.pi, size=num_runs)
    rads   = rng.uniform(0.0, radius, size=num_runs)
    x_targets = x_target + rads * np.cos(angles)
    z_targets = z_target + rads * np.sin(angles)

    args = []
    for xt, zt in zip(x_targets, z_targets):
        p = dict(params)
        p['x_target'] = float(xt)
        p['z_target'] = float(zt)
        args.append((p, y0.copy(), t_span, dt))
    t0 = time.perf_counter()
    with mp.Pool(processes=min(num_runs, mp.cpu_count())) as pool:
        results = pool.starmap(run_simulation, args)

    mc_secs = time.perf_counter() - t0
    print(f"\nMonte Carlo runtime: {mc_secs:.3f} s ({num_runs/mc_secs:.1f} runs/s)")
    miss_distances = [r[0] for r in results]
    solutions = [r[1] for r in results]

    print("\nMonte Carlo Results:")
    print("-" * 50)
    for i, (xt, zt, m_) in enumerate(zip(x_targets, z_targets, miss_distances)):
        print(f"[{i:02d}] Target @ ({xt:6.1f}, {zt:6.1f}) → Miss: {m_:6.2f} m")
    print("-" * 50)
    print(f"Average miss distance: {np.mean(miss_distances):.2f} m")
    print(f"Std dev: {np.std(miss_distances):.2f} m")
    print(f"Min/Max: {np.min(miss_distances):.2f} / {np.max(miss_distances):.2f} m")

    # Plot all MC trajectories on one figure for fast review
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for sol_i, xt, zt, m_ in zip(solutions, x_targets, z_targets, miss_distances):
        ax.plot(sol_i.y[0], sol_i.y[2], sol_i.y[1], linewidth=1)
        color = 'red' if m_ < HIT_THRESH else 'grey'
        ax.scatter(xt, zt, 0.0, marker='X', s=60, c=color)
        ax.plot([xt, sol_i.y[0,-1]], [zt, sol_i.y[2,-1]], [0, sol_i.y[1,-1]], linestyle='--', alpha=0.4)
    ax.scatter(0, 0, y0_alt, marker='o', s=100, label='Launch')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Z (m)'); ax.set_zlabel('Altitude (m)')
    ax.set_title('Monte Carlo Trajectories (6-DoF-lite)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim3d(0.0, 100.0)   # X (forward)
    ax.set_ylim3d(-50, 50)   # Z (side); use
    ax.set_zlim3d(0, 100.0)   # Altitude
    plt.tight_layout(); plt.show()

    # Summary
    print("\n" + "="*50)
    print("MONTE CARLO SUMMARY")
    print("="*50)
    print(f"Total runs: {num_runs}")
    print(f"Target radius: {radius} m")
    print(f"Average miss distance: {np.mean(miss_distances):.2f} m")
    print(f"Standard deviation: {np.std(miss_distances):.2f} m")
    print(f"Best shot: {np.min(miss_distances):.2f} m")
    print(f"Worst shot: {np.max(miss_distances):.2f} m")
    print(f"Success rate (<1m): {(np.sum(np.array(miss_distances) < 1.0)/num_runs*100):.1f}%")
    print(f"Success rate (<2m): {(np.sum(np.array(miss_distances) < 2.0)/num_runs*100):.1f}%")