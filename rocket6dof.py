"""
6-DoF flight simulator for a model rocket
Francesco Pighetti - Aerospace Engineering, Politecnico di Torino

Rigid-body simulation of the AeroTech F62T-6 rocket from my previous projects:
  - translation: Newton's law in an Earth-fixed frame (x East, y North, z Up)
  - rotation: Euler's equations in the body frame, attitude as a quaternion
  - aerodynamics: drag from OpenRocket, normal force and centre of pressure
    computed with the Barrowman method, aerodynamic + jet damping
  - launch rail, constant wind, parachute descent

Version 1: a single flight, validated against OpenRocket.
Run:  python rocket6dof.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

RESULTS, DOCS = Path("results"), Path("docs")
RESULTS.mkdir(exist_ok=True)
DOCS.mkdir(exist_ok=True)
G0 = 9.80665


# ============================================================================
# 1. Rocket data
# ============================================================================
def load_openrocket(path="razzo2_dati.csv"):
    """Thrust curve, mass properties and drag coefficient exported by OpenRocket."""
    rows = [line for line in open(path, encoding="utf-8") if line.strip() and not line.startswith("#")]
    d = np.array([[float(x) for x in line.split(",")] for line in rows])
    t, h, vz, v = d[:, 0], d[:, 1], d[:, 2], d[:, 4]
    burn = d[:, 8] > 0                                  # mass properties are logged during the burn
    coast = (t > 1.0) & (vz > 0) & ~np.isnan(d[:, 10])  # drag coefficient vs speed, from the coast phase
    order = np.argsort(v[coast])
    return {
        "t": t, "h": h, "vz": vz,
        "thrust_t": np.r_[0.0, t[burn], t[burn][-1] + 1e-3], "thrust_N": np.r_[0.0, d[burn, 8], 0.0],
        "mass_t": t[burn], "mass_kg": d[burn, 5] / 1000,
        "I_kgm2": d[burn, 6], "cg_m": d[burn, 7] / 100,
        "cd_v": v[coast][order], "cd": d[coast, 10][order],
    }


OR = load_openrocket()

ROCKET = {
    "length": 0.97,                 # [m]
    "diameter": 0.054,              # [m]
    "I_roll": 0.0006,               # roll inertia [kg m^2] (thin-wall tube estimate)
    # Barrowman normal-force slopes [1/rad] and their positions from the nose tip [m]
    "cna_nose": 2.0,   "x_nose": 0.056,      # ogive nose, X = 0.466 * L_nose
    "cna_fins": 11.13, "x_fins": 0.906,      # 4 trapezoidal fins with body interference
    "rail_length": 1.0,             # [m]
    "chute_cd": 0.8, "chute_d": 0.70,          # parachute
}
ROCKET["area"] = np.pi * ROCKET["diameter"] ** 2 / 4
ROCKET["cna"] = ROCKET["cna_nose"] + ROCKET["cna_fins"]
ROCKET["x_cp"] = (ROCKET["cna_nose"] * ROCKET["x_nose"] + ROCKET["cna_fins"] * ROCKET["x_fins"]) / ROCKET["cna"]


def thrust(t):
    return np.interp(t, OR["thrust_t"], OR["thrust_N"], left=0.0, right=0.0)


def mass_properties(t):
    """Mass [kg], pitch/yaw inertia [kg m^2], CG from the nose [m], mass flow [kg/s]."""
    m = np.interp(t, OR["mass_t"], OR["mass_kg"])
    I = np.interp(t, OR["mass_t"], OR["I_kgm2"])
    cg = np.interp(t, OR["mass_t"], OR["cg_m"])
    dt = 1e-3
    mdot = (np.interp(t - dt, OR["mass_t"], OR["mass_kg"]) - m) / dt
    return m, I, cg, mdot


def drag_coefficient(speed):
    return np.interp(speed, OR["cd_v"], OR["cd"])


def air_density(z):
    """International Standard Atmosphere, troposphere."""
    T = 288.15 - 0.0065 * z
    p = 101325.0 * (T / 288.15) ** 5.25588
    return p / (287.05 * T)


# ============================================================================
# 2. Quaternion helpers (scalar first, body -> world)
# ============================================================================
def quat_to_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(R):
    w = np.sqrt(max(1e-12, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    return np.array([w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)])


def quat_derivative(q, w_body):
    """dq/dt = 1/2 q x [0, omega]"""
    p, qx, qy, qz = q
    wx, wy, wz = w_body
    return 0.5 * np.array([
        -qx * wx - qy * wy - qz * wz,
        p * wx + qy * wz - qz * wy,
        p * wy - qx * wz + qz * wx,
        p * wz + qx * wy - qy * wx,
    ])


def rail_attitude(elevation_deg, azimuth_deg):
    """Quaternion with the body x axis (the nose) along the launch rail."""
    el, az = np.radians(elevation_deg), np.radians(azimuth_deg)
    x_b = np.array([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)])
    helper = np.array([0.0, 0.0, 1.0]) if abs(x_b[2]) < 0.99 else np.array([0.0, 1.0, 0.0])
    y_b = np.cross(helper, x_b); y_b /= np.linalg.norm(y_b)
    z_b = np.cross(x_b, y_b)
    return matrix_to_quat(np.column_stack([x_b, y_b, z_b]))


# ============================================================================
# 3. Equations of motion
# ============================================================================
def derivatives(t, s, wind):
    """State s = [position(3), velocity(3), quaternion(4), angular rate in body(3)]."""
    r, v, q, w = s[0:3], s[3:6], s[6:10], s[10:13]
    m, I, x_cg, mdot = mass_properties(t)
    R = quat_to_matrix(q)
    rho = air_density(r[2])

    v_air_b = R.T @ (v - wind)                      # air-relative velocity in body axes
    V = np.linalg.norm(v_air_b)
    F_b = np.array([thrust(t), 0.0, 0.0])           # thrust along the rocket axis
    M_b = np.zeros(3)
    alpha = 0.0

    if V > 0.5:
        q_dyn = 0.5 * rho * V**2
        A = ROCKET["area"]
        lateral = np.hypot(v_air_b[1], v_air_b[2])
        alpha = np.arctan2(lateral, v_air_b[0])     # total angle of attack

        F_b[0] -= q_dyn * A * drag_coefficient(V)   # axial drag
        if lateral > 1e-9:                          # normal force, opposite to the sideslip
            N = q_dyn * A * ROCKET["cna"] * alpha
            F_b[1] -= N * v_air_b[1] / lateral
            F_b[2] -= N * v_air_b[2] / lateral
        # restoring moment: normal force applied at the centre of pressure
        r_cp = np.array([-(ROCKET["x_cp"] - x_cg), 0.0, 0.0])
        M_b += np.cross(r_cp, np.array([0.0, F_b[1], F_b[2]]))
        # damping: aerodynamic (fins and nose) + jet damping (exhaust mass flow)
        c_aero = 0.5 * rho * V * A * (ROCKET["cna_nose"] * (ROCKET["x_nose"] - x_cg) ** 2
                                      + ROCKET["cna_fins"] * (ROCKET["x_fins"] - x_cg) ** 2)
        c_jet = mdot * (ROCKET["length"] - x_cg) ** 2
        M_b[1] -= (c_aero + c_jet) * w[1]
        M_b[2] -= (c_aero + c_jet) * w[2]
        M_b[0] -= 1e-5 * w[0]                       # small roll damping

    a = (R @ F_b) / m + np.array([0.0, 0.0, -G0])   # Newton in the Earth frame
    I_vec = np.array([ROCKET["I_roll"], I, I])
    w_dot = (M_b - np.cross(w, I_vec * w)) / I_vec  # Euler's equations
    return np.concatenate([v, a, quat_derivative(q, w), w_dot]), alpha


def rk4_step(t, s, dt, wind):
    k1, alpha = derivatives(t, s, wind)
    k2, _ = derivatives(t + dt / 2, s + dt / 2 * k1, wind)
    k3, _ = derivatives(t + dt / 2, s + dt / 2 * k2, wind)
    k4, _ = derivatives(t + dt, s + dt * k3, wind)
    s_new = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    s_new[6:10] /= np.linalg.norm(s_new[6:10])      # keep the quaternion unit-length
    return s_new, alpha


# ============================================================================
# 4. Flight simulation: rail -> powered/coast (6-DoF) -> parachute
# ============================================================================
def simulate(wind_speed=0.0, wind_from_deg=270.0, rail_elevation=90.0, rail_azimuth=0.0, dt=1e-3):
    # wind blowing FROM wind_from_deg (meteorological convention: 270 = from the West)
    wf = np.radians(wind_from_deg)
    wind = -wind_speed * np.array([np.sin(wf), np.cos(wf), 0.0])

    q0 = rail_attitude(rail_elevation, rail_azimuth)
    d_rail = quat_to_matrix(q0)[:, 0]
    log = {k: [] for k in ("t", "pos", "vel", "q", "alpha", "phase")}
    events = {}

    # --- Launch rail: 1-D motion along the rail ---
    t, s_rail, v_rail = 0.0, 0.0, 0.0
    while s_rail < ROCKET["rail_length"]:
        m, *_ = mass_properties(t)
        F = thrust(t) - m * G0 * d_rail[2] - 0.5 * air_density(0) * v_rail**2 * ROCKET["area"] * drag_coefficient(v_rail)
        a = max(F / m, 0.0) if v_rail == 0 else F / m
        v_rail += a * dt; s_rail += v_rail * dt; t += dt
        log["t"].append(t); log["pos"].append(s_rail * d_rail); log["vel"].append(v_rail * d_rail)
        log["q"].append(q0); log["alpha"].append(0.0); log["phase"].append(0)
    events["rail_exit_time"], events["rail_exit_speed"] = t, v_rail

    # --- Free flight with 6 degrees of freedom until apogee ---
    s = np.concatenate([s_rail * d_rail, v_rail * d_rail, q0, np.zeros(3)])
    while not (t > 1.5 and s[5] <= 0):
        s, alpha = rk4_step(t, s, dt, wind)
        t += dt
        log["t"].append(t); log["pos"].append(s[0:3].copy()); log["vel"].append(s[3:6].copy())
        log["q"].append(s[6:10].copy()); log["alpha"].append(alpha); log["phase"].append(1 if thrust(t) > 0 else 2)
    events["apogee_time"], events["apogee"] = t, s[2]

    # --- Parachute descent: point mass with drag ---
    r, v = s[0:3].copy(), s[3:6].copy()
    m_dry = OR["mass_kg"][-1]
    cda = ROCKET["chute_cd"] * np.pi * ROCKET["chute_d"] ** 2 / 4
    dt_d = 0.01
    while r[2] > 0:
        v_air = v - wind
        a = -0.5 * air_density(r[2]) * np.linalg.norm(v_air) * v_air * cda / m_dry + np.array([0, 0, -G0])
        v = v + a * dt_d; r = r + v * dt_d; t += dt_d
        log["t"].append(t); log["pos"].append(r.copy()); log["vel"].append(v.copy())
        log["q"].append(log["q"][-1]); log["alpha"].append(np.nan); log["phase"].append(3)
    events["landing_time"], events["landing_speed"] = t, np.linalg.norm(v)
    events["landing_point"] = r[:2]

    out = {k: np.array(val) for k, val in log.items()}
    out["events"] = events
    vel = out["vel"]
    events["max_speed"] = np.linalg.norm(vel, axis=1).max()
    return out


def pitch_from_vertical(q_array):
    """Angle between the rocket axis and the vertical [deg]."""
    nose_z = 2 * (q_array[:, 1] * q_array[:, 3] - q_array[:, 0] * q_array[:, 2])   # z of body x axis
    return np.degrees(np.arccos(np.clip(nose_z, -1, 1)))


# ============================================================================
# 5. Validation against OpenRocket
# ============================================================================
def validate():
    sim = simulate(wind_speed=0.0)
    ev = sim["events"]
    ref = {"rail_exit_time": 0.188, "rail_exit_speed": 12.1, "max_speed": 62.1, "apogee": 191.3,
           "apogee_time": 6.48, "landing_speed": 6.31, "landing_time": 37.1}
    labels = {"rail_exit_time": ("Rail exit time", "s"), "rail_exit_speed": ("Velocity off the rail", "m/s"),
              "max_speed": ("Maximum velocity", "m/s"),
              "apogee": ("Apogee", "m"), "apogee_time": ("Time to apogee", "s"),
              "landing_speed": ("Landing velocity", "m/s"), "landing_time": ("Flight time", "s")}

    print(f"\nStatic margin at liftoff: {(ROCKET['x_cp'] - OR['cg_m'][0]) / ROCKET['diameter']:.2f} calibres "
          f"(CP {100 * ROCKET['x_cp']:.1f} cm, CG {100 * OR['cg_m'][0]:.1f} cm from the nose)")
    print("\nValidation against OpenRocket (no wind):")
    print(f"  {'':<24}{'OpenRocket':>12}{'6-DoF':>10}{'diff':>9}")
    rows = []
    for key, (name, unit) in labels.items():
        mine, theirs = ev[key], ref[key]
        diff = 100 * (mine - theirs) / theirs
        rows.append((name, unit, theirs, mine, diff))
        print(f"  {name:<24}{theirs:>9.2f} {unit:<3}{mine:>7.2f} {unit:<3}{diff:>+6.1f} %")

    fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    t = sim["t"]
    ax[0].plot(OR["t"], OR["h"], "o", ms=3, color="#999999", label="OpenRocket")
    ax[0].plot(t, sim["pos"][:, 2], color="#0072B2", lw=2, label="My 6-DoF")
    ax[0].set_ylabel("Altitude [m]"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[0].set_title("Validation against OpenRocket (no wind)")
    ax[1].plot(OR["t"], OR["vz"], "o", ms=3, color="#999999", label="OpenRocket")
    ax[1].plot(t, sim["vel"][:, 2], color="#0072B2", lw=2, label="My 6-DoF")
    ax[1].set_ylabel("Vertical velocity [m/s]"); ax[1].set_xlabel("Time [s]"); ax[1].grid(alpha=0.3)
    ax[1].set_xlim(0, 40)
    fig.tight_layout(); fig.savefig(RESULTS / "validation_openrocket.png", dpi=150); plt.close(fig)
    return rows


# ============================================================================
# 6. Flight in a crosswind: weathercocking and drift
# ============================================================================
def windy_flight(wind_speed=5.0):
    calm = simulate(wind_speed=0.0)
    windy = simulate(wind_speed=wind_speed, wind_from_deg=270.0)
    ev = windy["events"]
    drift = np.linalg.norm(ev["landing_point"])
    print(f"\nFlight with a {wind_speed:.0f} m/s wind from the West:")
    k_exit = np.argmax(windy["phase"] == 1)
    k_burn = np.argmin(np.abs(windy["t"] - 0.999))
    print(f"  apogee {ev['apogee']:.1f} m (calm: {calm['events']['apogee']:.1f} m)")
    print(f"  angle of attack leaving the rail {np.degrees(windy['alpha'][k_exit + 1]):.1f} deg")
    print(f"  tilt into the wind at burnout {pitch_from_vertical(windy['q'][[k_burn]])[0]:.1f} deg")
    print(f"  landing {drift:.0f} m from the pad")

    # Attitude: the rocket turns INTO the wind (weathercocking)
    fig, ax = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    for sim, name, color in ((calm, "No wind", "#999999"), (windy, f"{wind_speed:.0f} m/s wind", "#D55E00")):
        k = sim["phase"] < 3
        ax[0].plot(sim["t"][k], pitch_from_vertical(sim["q"][k]), color=color, lw=2, label=name)
        ax[1].plot(sim["t"][k], np.degrees(sim["alpha"][k]), color=color, lw=2, label=name)
    ax[0].axvline(0.999, color="k", ls=":", lw=1); ax[1].axvline(0.999, color="k", ls=":", lw=1)
    ax[0].text(1.05, ax[0].get_ylim()[1] * 0.9, "burnout", fontsize=9)
    ax[0].set_ylabel("Tilt from vertical [deg]"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[0].set_title("Weathercocking: the rocket turns into the wind")
    ax[1].set_ylabel("Angle of attack [deg]"); ax[1].set_xlabel("Time [s]"); ax[1].grid(alpha=0.3)
    ax[1].set_xlim(0, windy["events"]["apogee_time"] + 0.2)
    fig.tight_layout(); fig.savefig(RESULTS / "weathercocking.png", dpi=150); plt.close(fig)

    # 3-D trajectory (static image)
    fig = plt.figure(figsize=(8, 7))
    ax3 = fig.add_subplot(projection="3d")
    colors = {0: "#000000", 1: "#D55E00", 2: "#0072B2", 3: "#009E73"}
    names = {0: "Rail", 1: "Powered", 2: "Coast", 3: "Parachute"}
    p = windy["pos"]
    for ph in (0, 1, 2, 3):
        k = windy["phase"] == ph
        ax3.plot(p[k, 0], p[k, 1], p[k, 2], color=colors[ph], lw=2.5, label=names[ph])
    ax3.plot(p[:, 0], p[:, 1], 0 * p[:, 2], color="#BBBBBB", lw=1, ls="--", label="Ground track")
    ax3.scatter([p[-1, 0]], [p[-1, 1]], [0], color="#009E73", s=40)
    half = max(np.abs(p[:, 0]).max(), np.abs(p[:, 1]).max(), 20)
    ax3.set_xlim(-half, half); ax3.set_ylim(-half, half)
    ax3.set_xlabel("East [m]"); ax3.set_ylabel("North [m]"); ax3.set_zlabel("Altitude [m]")
    ax3.set_title(f"3-D trajectory with a {wind_speed:.0f} m/s wind from the West")
    ax3.legend(loc="upper left"); ax3.view_init(elev=18, azim=-60)
    fig.tight_layout(); fig.savefig(RESULTS / "trajectory_3d.png", dpi=150); plt.close(fig)

    # 3-D trajectory (interactive, for GitHub Pages)
    figi = go.Figure()
    for ph in (0, 1, 2, 3):
        k = windy["phase"] == ph
        figi.add_trace(go.Scatter3d(x=p[k, 0], y=p[k, 1], z=p[k, 2], mode="lines",
                                    line=dict(color=colors[ph], width=6), name=names[ph]))
    figi.add_trace(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=0 * p[:, 2], mode="lines",
                                line=dict(color="#BBBBBB", width=2, dash="dash"), name="Ground track"))
    figi.update_layout(title=f"6-DoF flight of the F62T rocket, {wind_speed:.0f} m/s wind from the West "
                             "(drag to rotate)",
                       scene=dict(xaxis_title="East [m]", yaxis_title="North [m]", zaxis_title="Altitude [m]",
                                  aspectmode="data"),
                       height=750, margin=dict(l=0, r=0, t=50, b=0))
    figi.write_html(DOCS / "index.html", include_plotlyjs="cdn")


# ============================================================================
if __name__ == "__main__":
    validate()
    windy_flight(wind_speed=5.0)
    print("\nDone. Figures in results/, interactive trajectory in docs/index.html")
