"""Generate the README diagrams, 3D plots and charts (flat PNG files in the repo root).

Everything comes from the project's own solvers in main.py.

Usage:  python make_diagrams.py [name ...]   (no names = all; e.g. python make_diagrams.py multi-re moody)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import main as fl

OUT = Path(__file__).resolve().parent


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


def architecture():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.2); ax.axis("off")

    def box(x, y, w, h, t, c):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12", fc=c, ec="#263238", lw=1.4))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=9.5, weight="bold", color="#102027")

    def arr(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14, lw=1.4, color="#37474F"))

    ax.text(6, 5.85, "FluidFlow CFD - architecture", ha="center", fontsize=14, weight="bold")
    box(4.2, 4.7, 3.6, 0.8, "CLI  (python main.py ...)", "#BBDEFB")
    xs = (0.1, 2.5, 4.9, 7.3, 9.7)
    top = ["cavity + convection\nstream-vorticity, heat\nSSP-RK3 + DST", "taylor-green / shear\npseudo-spectral\nIF-RK4, dealiased",
           "channel + blasius\nCrank-Nicolson / shooting\nstart-up, boundary layer", "pipe\nColebrook, minor losses,\npump power",
           "network\nglobal gradient algorithm\nloops, demands, heads"]
    ref = ["Ghia 1982 Re 100-1000\nde Vahl Davis 1983 Nu", "exact Taylor-Green\nenergy balance", "exact series\nf''(0) = 0.33206",
           "Haaland vs\nColebrook", "mass + loop energy\nbalance, D^4/L split"]
    for x, a, b in zip(xs, top, ref):
        box(x, 3.1, 2.2, 1.0, a, "#C8E6C9")
        box(x, 1.5, 2.2, 0.9, b, "#FFE0B2")
    box(2.0, 0.1, 8.0, 0.75, "Outputs: JSON, NPZ, PNG plots  |  unittest/pytest suite + GitHub Actions CI", "#E1BEE7")
    for x in (1.2, 3.6, 6.0, 8.4, 10.8):
        arr(6.0, 4.7, x, 4.1)
        arr(x, 3.1, x, 2.4)
        arr(x, 1.5, x if 2.0 < x < 10 else (3.0 if x < 6 else 9.0), 0.85)
    save(fig, "diagram_architecture.png")


def cavity_figures():
    cav = fl.LidDrivenCavity(65, 100)
    cav.run()
    x = np.linspace(0, 1, cav.n)
    X, Y = np.meshgrid(x, x)
    u, v = cav.velocity()
    speed = np.hypot(u, v)

    fig = plt.figure(figsize=(9.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    s = ax.plot_surface(X, Y, cav.psi, cmap="viridis", linewidth=0, antialiased=True)
    ax.contourf(X, Y, cav.psi, zdir="z", offset=float(cav.psi.min()) - 0.03, cmap="viridis", alpha=0.85)
    ax.set_zlim(float(cav.psi.min()) - 0.03, 0.01)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("streamfunction")
    ax.set_title("Lid-driven cavity (Re = 100): 3D streamfunction surface", weight="bold")
    ax.view_init(elev=30, azim=-120)
    fig.colorbar(s, shrink=0.6, pad=0.08)
    save(fig, "diagram_3d_cavity_streamfunction.png")

    fig = plt.figure(figsize=(9.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    s = ax.plot_surface(X, Y, cav.omega, cmap="RdBu_r", vmin=-10, vmax=10, linewidth=0, antialiased=True)
    ax.set_zlim(-10, 10)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("vorticity")
    ax.set_title("Cavity vorticity surface (clipped at +/-10; strong at the lid corners)", weight="bold")
    ax.view_init(elev=32, azim=-125)
    fig.colorbar(s, shrink=0.6, pad=0.08)
    save(fig, "diagram_3d_cavity_vorticity.png")

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.contourf(X, Y, speed, 24, cmap="viridis")
    ax.streamplot(x, x, u, v, color="white", density=1.6, linewidth=0.8)
    ax.set_aspect("equal")
    ax.set_title("Cavity Re = 100: speed and streamlines", weight="bold")
    fig.colorbar(im, label="|u|")
    save(fig, "chart_cavity_streamlines.png")

    y, uc = cav.centreline_u()
    ref = np.array(fl.GHIA_RE100_U)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot(uc, y, lw=2, label="FluidFlow (65 x 65)")
    ax.plot(ref[:, 1], ref[:, 0], "o", label="Ghia et al. (1982)")
    ax.set_xlabel("u along x = 0.5"); ax.set_ylabel("y")
    ax.set_title(f"Validation: RMS deviation {fl.ghia_error(cav)['rms']:.4f}", weight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    save(fig, "chart_cavity_validation.png")

    t, r = zip(*cav.history)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(t, r, lw=2)
    ax.set_xlabel("pseudo-time"); ax.set_ylabel("max |d omega / dt|")
    ax.set_title("Cavity convergence to steady state", weight="bold")
    ax.grid(alpha=0.3, which="both")
    save(fig, "chart_cavity_convergence.png")


def cavity_multi_re(n: int = 129):
    """Re = 100, 400, 1000 on one grid with continuation (warm start), each checked against Ghia et al. (1982)."""
    cav = fl.LidDrivenCavity(n, 100)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.2), gridspec_kw={"height_ratios": [1, 1.05]})
    x = np.linspace(0, 1, n)
    stats = {}
    for k, re in enumerate((100, 400, 1000)):
        if k:
            cav.set_reynolds(re)
        info = cav.run(tol=1e-5, max_time=400)
        u, v = cav.velocity()
        cx, cy, pmin = cav.vortex_centre()
        err = fl.ghia_error(cav)
        stats[re] = {**info, "psi_min": pmin, "centre": (cx, cy), **err}
        ax = axes[0, k]
        ax.contourf(x, x, np.hypot(u, v), 24, cmap="viridis")
        ax.streamplot(x, x, u, v, color="white", density=1.7, linewidth=0.7)
        ax.set_aspect("equal")
        ax.set_title(f"Re = {re}: vortex centre ({cx:.3f}, {cy:.3f})", weight="bold")
        y, uc = cav.centreline_u()
        ref = fl.ghia_reference(re)
        ax = axes[1, k]
        ax.plot(uc, y, lw=2, label=f"FluidFlow ({n} x {n})")
        ax.plot(ref[:, 1], ref[:, 0], "o", label="Ghia et al. (1982)")
        ax.set_xlabel("u along x = 0.5"); ax.set_ylabel("y"); ax.grid(alpha=0.3); ax.legend()
        ax.set_title(f"RMS deviation {err['rms']:.4f}, max {err['max']:.4f}", weight="bold")
    fig.suptitle("Lid-driven cavity at three Reynolds numbers (warm-started continuation)", weight="bold", fontsize=14)
    fig.tight_layout()
    save(fig, "chart_cavity_multi_re.png")
    for re, st in stats.items():
        print(f"Re={re}: steps={st['steps']} converged={st['converged']} psi_min={st['psi_min']:.4f} "
              f"centre=({st['centre'][0]:.4f},{st['centre'][1]:.4f}) rms={st['rms']:.4f} max={st['max']:.4f}")


def convergence_chart():
    res = fl.cavity_convergence(100.0, (17, 33, 65, 129))
    h = np.array([1.0 / (r["n"] - 1) for r in res["rows"]])
    err = np.array([abs(r["psi_min"] - fl.GHIA_VORTEX[100][2]) for r in res["rows"]])
    rms = np.array([r["ghia_rms"] for r in res["rows"]])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(h, rms, "o-", lw=2, label="centreline RMS error vs Ghia et al.")
    ax.loglog(h, err, "s-", lw=2, label="|psi_min - Ghia psi_min|")
    ax.loglog(h, rms[0] * (h / h[0]) ** 2, "k--", label="second-order slope")
    ax.set_xlabel("grid spacing h"); ax.set_ylabel("error")
    ax.set_title(f"Cavity grid convergence (Re = 100), observed order {res['observed_order']:.2f}", weight="bold")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    save(fig, "chart_cavity_grid_convergence.png")
    print({r["n"]: (round(r["psi_min"], 5), round(r["ghia_rms"], 5)) for r in res["rows"]}, res["observed_order"])


def pipe_system_chart():
    """Pressure drop and pump power versus flow velocity for a 100 m, 50 mm steel pipe with fittings."""
    v = np.linspace(0.2, 4.0, 80)
    rows = [fl.pipe_pressure_drop(0.05, 100, vv, minor_k=2.5, efficiency=0.7) for vv in v]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].plot(v, [r["pressure_drop_major_pa"] / 1e3 for r in rows], lw=2, label="pipe friction")
    ax[0].plot(v, [r["pressure_drop_minor_pa"] / 1e3 for r in rows], lw=2, label="fittings (K = 2.5)")
    ax[0].plot(v, [r["pressure_drop_pa"] / 1e3 for r in rows], "k--", lw=2, label="total")
    ax[0].set_xlabel("velocity (m/s)"); ax[0].set_ylabel("pressure drop (kPa)"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[0].set_title("Pressure drop, D = 50 mm, L = 100 m", weight="bold")
    ax[1].plot([r["flow_rate_m3_s"] * 1e3 for r in rows], [r["pump_power_w"] / 1e3 for r in rows], lw=2, color="C3")
    ax[1].set_xlabel("flow rate (L/s)"); ax[1].set_ylabel("pump shaft power (kW), eta = 0.7"); ax[1].grid(alpha=0.3)
    ax[1].set_title("Pump power versus flow rate", weight="bold")
    save(fig, "chart_pipe_system.png")


def convection_figures():
    """Differentially heated cavity at Ra = 1e3, 1e4, 1e5 (Pr = 0.71) and Nu against de Vahl Davis (1983)."""
    cases = ((1e3, 41), (1e4, 61), (1e5, 81))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9))
    nus = {}
    for a, (ra, n) in zip(axes, cases):
        cav = fl.NaturalConvectionCavity(n, ra)
        cav.run()
        u, v = cav.velocity()
        im = a.contourf(cav.x, cav.y, cav.T, 24, cmap="inferno")
        a.streamplot(cav.x, cav.y, u, v, color="white", density=1.3, linewidth=0.7)
        a.set_aspect("equal")
        nu = cav.nusselt()
        nus[ra] = nu
        a.set_title(f"Ra = {ra:g}: Nu = {nu:.3f} (ref. {fl.DE_VAHL_DAVIS_NU[ra]})", weight="bold")
    fig.colorbar(im, ax=list(axes), shrink=0.8, label="temperature")
    fig.suptitle("Natural convection in a differentially heated square cavity (Pr = 0.71)", weight="bold")
    save(fig, "chart_convection_temperature.png")

    ras = np.array(sorted(nus))
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.loglog(ras, [nus[r] for r in ras], "o-", lw=2, label="FluidFlow")
    ax.loglog(ras, [fl.DE_VAHL_DAVIS_NU[r] for r in ras], "ks", mfc="none", ms=10, label="de Vahl Davis (1983)")
    ax.set_xlabel("Rayleigh number"); ax.set_ylabel("average Nusselt number"); ax.grid(alpha=0.3, which="both"); ax.legend()
    ax.set_title("Heat transfer vs Rayleigh number", weight="bold")
    save(fig, "chart_convection_nusselt.png")
    print({r: round(v, 4) for r, v in nus.items()})

    cav = fl.NaturalConvectionCavity(61, 1e4)
    cav.run()
    X, Y = np.meshgrid(cav.x, cav.y)
    fig = plt.figure(figsize=(9.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    s = ax.plot_surface(X, Y, cav.T, cmap="inferno", linewidth=0, antialiased=True)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("temperature")
    ax.set_title("3D temperature surface, Ra = 1e4 (thermal boundary layers on the side walls)", weight="bold")
    ax.view_init(elev=30, azim=-125)
    fig.colorbar(s, shrink=0.6, pad=0.08)
    save(fig, "diagram_3d_convection_temperature.png")


def rectangular_figures():
    """Non-square geometry: lid-driven cavity (Re = 100) and heated cavity (Ra = 1e4) with aspect ratios 0.5, 1, 2."""
    aspects = (0.5, 1.0, 2.0)
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.6), gridspec_kw={"width_ratios": [0.5, 1.0, 2.0]})
    for k, a in enumerate(aspects):
        cav = fl.LidDrivenCavity(65, 100, aspect=a)
        cav.run(tol=1e-5)
        u, v = cav.velocity()
        x, y, pmin = cav.vortex_centre()
        ax = axes[0, k]
        ax.contourf(cav.x, cav.y, np.hypot(u, v), 24, cmap="viridis")
        ax.streamplot(cav.x, cav.y, u, v, color="white", density=1.5, linewidth=0.7)
        ax.set_aspect("equal")
        ax.set_title(f"Lid-driven, W/H = {a:g}: psi_min {pmin:.4f}", weight="bold", fontsize=10)
        hc = fl.NaturalConvectionCavity(61, 1e4, aspect=a)
        hc.run(max_time=10)
        uu, vv = hc.velocity()
        ax = axes[1, k]
        ax.contourf(hc.x, hc.y, hc.T, 24, cmap="inferno")
        ax.streamplot(hc.x, hc.y, uu, vv, color="white", density=1.4, linewidth=0.7)
        ax.set_aspect("equal")
        ax.set_title(f"Heated, W/H = {a:g}: Nu = {hc.nusselt():.3f}", weight="bold", fontsize=10)
        print(f"aspect {a}: lid psi_min={pmin:.4f} centre=({x:.3f},{y:.3f}); heated Nu={hc.nusselt():.4f} converged")
    fig.suptitle("Non-square geometry: aspect ratios 0.5, 1 and 2", weight="bold", fontsize=14)
    fig.tight_layout()
    save(fig, "chart_rectangular_cavities.png")


def shear_figures():
    res = fl.shear_layer(n=128, nu=1e-3, t_end=6.0, dt=0.01, snapshots=(0.0, 2.0, 4.0, 6.0))
    snaps = res["snapshots"]
    keys = sorted(snaps)
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))
    for a, k in zip(axes, keys):
        im = a.pcolormesh(res["x"], res["y"], snaps[k], cmap="RdBu_r", vmin=-25, vmax=25, shading="auto")
        a.set_title(f"t = {k:g}")
        a.set_aspect("equal")
    fig.colorbar(im, ax=list(axes), shrink=0.8, label="vorticity")
    fig.suptitle("Double shear layer roll-up (pseudo-spectral, 128 x 128)", weight="bold")
    save(fig, "chart_shear_snapshots.png")

    w = snaps[keys[-1]][::2, ::2]
    X, Y = res["x"][::2, ::2], res["y"][::2, ::2]
    fig = plt.figure(figsize=(10, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    s = ax.plot_surface(X, Y, w, cmap="RdBu_r", vmin=-25, vmax=25, linewidth=0, antialiased=True)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("vorticity")
    ax.set_title("3D vorticity surface at t = 6 (shear layer vortices)", weight="bold")
    ax.view_init(elev=35, azim=-60)
    fig.colorbar(s, shrink=0.6, pad=0.08)
    save(fig, "diagram_3d_shear_vorticity.png")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(res["t"], res["energy"] / res["energy"][0], lw=2, label="kinetic energy E / E0")
    ax.plot(res["t"], res["enstrophy"] / res["enstrophy"][0], lw=2, label="enstrophy Z / Z0")
    ax.set_xlabel("time"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("Shear layer: energy decays, enstrophy grows as vortices form", weight="bold")
    save(fig, "chart_shear_energy_enstrophy.png")


def taylor_green_chart():
    res = fl.taylor_green(32, 0.05, 4.0, 0.01)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(res["t"], res["energy"], lw=3, label="numerical")
    ax.plot(res["t"], res["energy_exact"], "k--", label="exact 0.25 exp(-4 nu t)")
    ax.set_xlabel("time"); ax.set_ylabel("kinetic energy"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title(f"Taylor-Green vortex: max error {res['error_max']:.1e}", weight="bold")
    save(fig, "chart_taylor_green.png")


def channel_figure():
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for t in (0.1, 0.25, 0.5, 1.0, 3.0, 100.0):
        r = fl.poiseuille_transient(101, 0.01, 1.0, 1.0, t, 1e-3 if t <= 3 else 0.05)
        ax.plot(r["u"], r["y"], lw=2, label=f"t = {t:g}")
        ax.plot(r["u_exact"], r["y"], "k:", lw=1)
    ax.set_xlabel("u"); ax.set_ylabel("y"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("Poiseuille start-up (dotted = exact series)", weight="bold")
    save(fig, "chart_channel_profiles.png")

    y = np.linspace(0, 1, 60)
    t = np.linspace(0.0, 2.0, 60)
    U = np.array([fl.poiseuille_exact(y, tt, 0.01, 1.0) for tt in t])
    Y, T = np.meshgrid(y, t)
    fig = plt.figure(figsize=(9.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    s = ax.plot_surface(Y, T, U, cmap="plasma", linewidth=0)
    ax.set_xlabel("y"); ax.set_ylabel("time"); ax.set_zlabel("u")
    ax.set_title("3D: channel velocity profile developing in time", weight="bold")
    ax.view_init(elev=28, azim=-55)
    fig.colorbar(s, shrink=0.6, pad=0.08)
    save(fig, "diagram_3d_channel_startup.png")


def moody_chart():
    re = np.logspace(3, 8, 400)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    lam = re[re < 2300]
    ax.loglog(lam, 64 / lam, "k", lw=2, label="laminar 64/Re")
    tur = re[re > 4000]
    for rr in (0, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2):
        ax.loglog(tur, [fl.friction_factor_colebrook(r, rr) for r in tur], lw=1.6, label=f"e/D = {rr:g}")
    ax.axvspan(2300, 4000, color="gray", alpha=0.2, label="transitional")
    ax.set_xlabel("Reynolds number"); ax.set_ylabel("Darcy friction factor")
    ax.set_title("Moody diagram from the project's own friction-factor code", weight="bold")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    save(fig, "chart_moody.png")


def blasius_chart():
    """Blasius profile and skin friction; the solver's f''(0) against the literature value."""
    bl = fl.blasius()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    ax[0].plot(bl["fp"], bl["eta"], lw=2, label="u / U = f'")
    ax[0].plot(bl["fpp"], bl["eta"], lw=2, label="f''")
    ax[0].axhline(bl["delta99"], color="gray", ls="--", lw=1)
    ax[0].text(0.03, bl["delta99"] + 0.15, f"delta99 = {bl['delta99']:.3f}", fontsize=9)
    ax[0].set_xlabel("similarity profile"); ax[0].set_ylabel("eta = y sqrt(U / nu x)"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[0].set_title("Blasius similarity profile", weight="bold")
    for xi, c in zip((0.1, 0.4, 1.0), ("C0", "C1", "C2")):
        s = np.sqrt(1.5e-5 * xi / 5.0)
        ax[1].plot(bl["fp"], bl["eta"] * s * 1e3, color=c, lw=2, label=f"x = {xi} m")
    ax[1].set_xlabel("u / U"); ax[1].set_ylabel("y (mm)"); ax[1].set_title("Air, U = 5 m/s: boundary layer growth", weight="bold")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    re = np.logspace(3, 5.7, 60)
    ax[2].loglog(re, bl["cf_sqrt_re"] / np.sqrt(re), lw=2, label="Blasius cf = 0.6641 / sqrt(Re_x)")
    ax[2].loglog(re, 1.328 / np.sqrt(re), "--", lw=1.5, label="plate drag C_D = 1.328 / sqrt(Re_L)")
    ax[2].axvline(5e5, color="C3", ls=":", label="typical transition")
    ax[2].set_xlabel("Reynolds number"); ax[2].set_ylabel("friction coefficient"); ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3, which="both")
    ax[2].set_title("Skin friction", weight="bold")
    save(fig, "chart_blasius.png")
    print({"fpp0": round(bl["fpp0"], 6), "delta99": round(bl["delta99"], 4)})


def network_figures():
    """Example looped network: flows/heads, and a demand sweep showing the head at the far node."""
    e = fl.EXAMPLE_NETWORK
    res = fl.pipe_network(e["pipes"], e["fixed_heads"], e["demands"])
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.25, 1]})
    pos = e["positions"]
    nodes = list(res["heads_m"])
    heads = np.array([res["heads_m"][n] for n in nodes])
    qmax = max(abs(p["flow_m3_s"]) for p in res["pipes"])
    for p in res["pipes"]:
        a, b = (p["from"], p["to"]) if p["flow_m3_s"] >= 0 else (p["to"], p["from"])
        (x0, y0), (x1, y1) = pos[a], pos[b]
        ax[0].annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=1 + 5 * abs(p["flow_m3_s"]) / qmax,
                                                                         color="#546E7A", shrinkA=16, shrinkB=16))
        ax[0].text((x0 + x1) / 2, (y0 + y1) / 2, f"{p['name']}\n{abs(p['flow_m3_s']) * 1e3:.2f} L/s", ha="center", va="center",
                   fontsize=8, bbox=dict(fc="white", ec="none", alpha=0.85))
    pts = np.array([pos[n] for n in nodes])
    sc = ax[0].scatter(pts[:, 0], pts[:, 1], c=heads, cmap="viridis_r", s=700, zorder=3, edgecolor="k")
    for n, (x, y) in zip(nodes, pts):
        ax[0].text(x, y, n, ha="center", va="center", color="white", weight="bold", zorder=4)
    fig.colorbar(sc, ax=ax[0], label="head (m)")
    ax[0].margins(0.15); ax[0].axis("off")
    ax[0].set_title("Looped network: flows and heads (P6 reverses)", weight="bold")
    scale = np.linspace(0.4, 2.0, 25)
    far = []
    for k in scale:
        r = fl.pipe_network(e["pipes"], e["fixed_heads"], {n: k * v for n, v in e["demands"].items()})
        far.append(r["heads_m"]["D"])
    ax[1].plot(scale * sum(e["demands"].values()) * 1e3, far, lw=2)
    ax[1].set_xlabel("total demand (L/s)"); ax[1].set_ylabel("head at node D (m)"); ax[1].grid(alpha=0.3)
    ax[1].set_title("Pressure sag at the far node as demand grows", weight="bold")
    save(fig, "chart_pipe_network.png")


def pump_chart():
    """Pump curve vs system curve: operating point, speed change and two pumps in parallel."""
    base = dict(shutoff_head=40.0, design_flow=0.01, design_head=30.0, diameter=0.1, length=200.0, static_head=10.0)
    q = np.linspace(0, 0.032, 120)
    area = np.pi * 0.1 ** 2 / 4
    k = (40.0 - 30.0) / 0.01 ** 2
    sysc = [10.0 + (fl.pipe_pressure_drop(0.1, 200.0, qq / area)["head_loss_m"] if qq > 0 else 0.0) for qq in q]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(q * 1e3, sysc, "k", lw=2.5, label="system curve (10 m lift + friction)")
    for s, n, c, lab in ((1.0, 1, "C0", "rated speed"), (0.85, 1, "C1", "85 % speed"), (1.0, 2, "C2", "2 pumps in parallel")):
        ax.plot(q * 1e3, s ** 2 * 40.0 - k * (q / n) ** 2, color=c, lw=2, label=lab)
        op = fl.pump_operating_point(**base, speed_ratio=s, n_parallel=n)
        ax.plot(op["flow_m3_s"] * 1e3, op["head_m"], "o", color=c, ms=9)
        ax.annotate(f"{op['flow_m3_s'] * 1e3:.1f} L/s", (op["flow_m3_s"] * 1e3, op["head_m"]), textcoords="offset points", xytext=(8, 6), fontsize=9)
    ax.set_ylim(0, 45); ax.set_xlabel("flow (L/s)"); ax.set_ylabel("head (m)"); ax.grid(alpha=0.3); ax.legend()
    ax.set_title("Pump operating points", weight="bold")
    save(fig, "chart_pump_operating_point.png")


def open_channel_chart():
    """Specific-energy diagram and normal/critical depth versus bed slope (rectangular channel, b = 3 m, Q = 5 m3/s)."""
    g, b, qf = 9.80665, 3.0, 5.0
    y = np.linspace(0.15, 2.0, 300)
    e = y + (qf / (b * y)) ** 2 / (2 * g)
    r = fl.open_channel(qf, b, 0.001, 0.013)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    ax[0].plot(e, y, lw=2)
    ax[0].plot([r["specific_energy_m"]], [r["normal_depth_m"]], "o", label="normal depth, S = 0.001")
    ax[0].plot([1.5 * r["critical_depth_m"]], [r["critical_depth_m"]], "s", label="critical depth (minimum energy)")
    ax[0].set_xlabel("specific energy E (m)"); ax[0].set_ylabel("depth y (m)"); ax[0].grid(alpha=0.3); ax[0].legend()
    ax[0].set_title("Specific energy diagram", weight="bold")
    sl = np.logspace(-4, -1.5, 60)
    rows = [fl.open_channel(qf, b, s, 0.013) for s in sl]
    ax[1].loglog(sl, [x["normal_depth_m"] for x in rows], lw=2, label="normal depth")
    ax[1].loglog(sl, [x["critical_depth_m"] for x in rows], "--", lw=2, label="critical depth")
    ax[1].set_xlabel("bed slope S"); ax[1].set_ylabel("depth (m)"); ax[1].grid(alpha=0.3, which="both"); ax[1].legend()
    ax[1].set_title("Mild-to-steep transition", weight="bold")
    save(fig, "chart_open_channel.png")


def backwater_chart():
    """Water-surface profiles M1 / M2 / S2 / S3 and a hydraulic jump (b = 3 m, Q = 5 m3/s, n = 0.013)."""
    q, b, n = 5.0, 3.0, 0.013
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    mild = fl.open_channel(q, b, 0.001, n)
    for y0, c in ((2.5, "C0"), (1.5, "C0"), (0.8, "C1")):
        r = fl.gvf_profile(q, b, 0.001, n, y0, 4000.0)
        ax[0].plot(-r["x_m"], r["depth_m"], color=c, lw=2, label=r["kind"] + f", y0 = {y0} m")
    ax[0].axhline(mild["normal_depth_m"], color="k", ls="--", lw=1, label=f"normal depth {mild['normal_depth_m']:.2f} m")
    ax[0].axhline(mild["critical_depth_m"], color="r", ls=":", lw=1.5, label=f"critical depth {mild['critical_depth_m']:.2f} m")
    ax[0].set_xlabel("distance upstream of the control (m, control at 0)"); ax[0].set_ylabel("depth (m)")
    ax[0].set_title("Mild slope S = 0.001: backwater and drawdown", weight="bold"); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
    steep = fl.open_channel(q, b, 0.02, n)
    for y0, c in ((1.2, "C2"), (0.15, "C3")):
        r = fl.gvf_profile(q, b, 0.02, n, y0, 120.0)
        ax[1].plot(r["x_m"], r["depth_m"], color=c, lw=2, label=r["kind"] + f", y0 = {y0} m")
    ax[1].axhline(steep["normal_depth_m"], color="k", ls="--", lw=1, label=f"normal depth {steep['normal_depth_m']:.2f} m")
    ax[1].axhline(steep["critical_depth_m"], color="r", ls=":", lw=1.5, label=f"critical depth {steep['critical_depth_m']:.2f} m")
    ax[1].set_xlabel("distance downstream of the control (m)"); ax[1].set_ylabel("depth (m)")
    ax[1].set_title("Steep slope S = 0.02 (S1 stops at critical depth)", weight="bold"); ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)
    save(fig, "chart_backwater.png")


ALL = {
    "architecture": architecture, "cavity": cavity_figures, "shear": shear_figures, "taylor-green": taylor_green_chart,
    "channel": channel_figure, "moody": moody_chart, "multi-re": cavity_multi_re, "convergence": convergence_chart,
    "pipe-system": pipe_system_chart, "blasius": blasius_chart, "pump": pump_chart, "open-channel": open_channel_chart, "backwater": backwater_chart, "network": network_figures, "convection": convection_figures, "rectangular": rectangular_figures,
}

if __name__ == "__main__":
    import sys

    names = sys.argv[1:] or list(ALL)
    unknown = [n for n in names if n not in ALL]
    if unknown:
        sys.exit(f"unknown figure(s) {unknown}; choose from {sorted(ALL)}")
    for name in names:
        ALL[name]()
