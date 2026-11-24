"""3D graphs, 3D charts, 3D mesh views, 3D models and 3D comparison tables (flat PNG files + comparison_tables.md).

Everything is computed with the project's own solvers in main.py.

Usage:  python make_3d.py [name ...]   (no names = all)
        names: mesh fields channel-model network-model blasius pump compare
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import main as fl

OUT = Path(__file__).resolve().parent
_CACHE: dict = {}


def save(fig, name):
    fig.savefig(OUT / name, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


def cavity(n: int, re: float = 100.0):
    key = ("cav", n, re)
    if key not in _CACHE:
        cav = fl.LidDrivenCavity(n, re)
        cav.run(tol=1e-5, max_time=200)
        _CACHE[key] = cav
    return _CACHE[key]


def grid(cav):
    x = np.linspace(0, cav.aspect, cav.nx)
    y = np.linspace(0, 1, cav.n)
    return np.meshgrid(x, y)


def colored(z, cmap="viridis"):
    return matplotlib.colormaps[cmap](Normalize(z.min(), z.max())(z))


def ax3(fig, pos, title, elev=28, azim=-60):
    ax = fig.add_subplot(*pos, projection="3d")
    ax.set_title(title, weight="bold", fontsize=11)
    ax.view_init(elev=elev, azim=azim)
    return ax


# ------------------------------------------------------------------ 3D mesh simulation
def mesh_views():
    """The computational mesh itself, drawn as a 3D surface mesh (height = streamfunction, colour = speed), for three grids."""
    fig = plt.figure(figsize=(17, 5.6))
    for k, n in enumerate((17, 33, 65)):
        cav = cavity(n)
        X, Y = grid(cav)
        u, v = cav.velocity()
        sp = np.hypot(u, v)
        ax = ax3(fig, (1, 3, k + 1), f"{n} x {n} mesh ({n * n} nodes), Re = 100")
        step = 1 if n < 40 else 2
        ax.plot_surface(X, Y, cav.psi, facecolors=colored(sp), rstride=step, cstride=step, edgecolor=(0, 0, 0, 0.55),
                        linewidth=0.35 if n < 40 else 0.25, shade=False)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("streamfunction")
        ax.set_box_aspect((1, 1, 0.6))
    fig.suptitle("3D mesh of the lid-driven cavity solution: refinement 17 -> 33 -> 65", weight="bold")
    save(fig, "mesh_3d_cavity_refinement.png")


def field_views():
    """3D surfaces with projected contours: speed, vorticity and streamfunction of the Re = 100 cavity (65 x 65)."""
    cav = cavity(65)
    X, Y = grid(cav)
    u, v = cav.velocity()
    fields = [("speed |u|", np.hypot(u, v), "viridis"), ("vorticity", np.clip(cav.omega, -10, 10), "RdBu_r"),
              ("streamfunction", cav.psi, "plasma")]
    fig = plt.figure(figsize=(17, 5.4))
    for k, (name, f, cmap) in enumerate(fields):
        ax = ax3(fig, (1, 3, k + 1), f"3D {name}")
        ax.plot_surface(X, Y, f, cmap=cmap, linewidth=0, antialiased=True, alpha=0.95)
        ax.contour(X, Y, f, 14, zdir="z", offset=f.min() - 0.15 * (f.max() - f.min()), cmap=cmap, linewidths=0.8)
        ax.set_zlim(f.min() - 0.15 * (f.max() - f.min()), f.max())
        ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.suptitle("Lid-driven cavity, Re = 100: 3D field surfaces with projected contours", weight="bold")
    save(fig, "chart_3d_cavity_fields.png")

    # natural convection: temperature surface + 3D streamfunction, two Rayleigh numbers
    fig = plt.figure(figsize=(14, 5.6))
    for k, ra in enumerate((1e4, 1e5)):
        key = ("conv", ra)
        if key not in _CACHE:
            c = fl.NaturalConvectionCavity(49, ra)
            c.run(tol=1e-6, max_time=20)
            _CACHE[key] = c
        c = _CACHE[key]
        X, Y = np.meshgrid(c.x, c.y)
        ax = ax3(fig, (1, 2, k + 1), f"Temperature surface, Ra = {ra:.0e}  (Nu = {c.nusselt('hot'):.3f})")
        ax.plot_surface(X, Y, c.T, cmap="inferno", linewidth=0, antialiased=True)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("T")
    save(fig, "chart_3d_convection_surfaces.png")


# ------------------------------------------------------------------ 3D models
def _box(ax, verts, color, alpha=1.0):
    ax.add_collection3d(Poly3DCollection(verts, facecolor=color, edgecolor="none", alpha=alpha))


def channel_model():
    """3D model of a trapezoidal channel carrying an M1 backwater curve (width and height exaggerated for display)."""
    q, b, s0, n, m = 5.0, 3.0, 0.001, 0.013, 2.0
    L = 3000.0
    prof = fl.gvf_profile(q, b, s0, n, 2.5, L, side_slope=m, n=120)
    xs = L - prof["x_m"]                      # distance from the upstream end (control is at the downstream end)
    order = np.argsort(xs)
    xs, y = xs[order], prof["depth_m"][order]
    bed = (L - xs) * s0
    wscale = 3.0                             # exaggerate width so the section is visible
    fig = plt.figure(figsize=(13, 6))
    ax = ax3(fig, (1, 1, 1), f"3D channel model: M1 backwater (trapezoid b = {b:g} m, {m:g}:1, Q = {q:g} m3/s)", 20, -68)
    half = lambda d: (b / 2 + m * d)           # noqa: E731
    for i in range(len(xs) - 1):
        x0, x1 = xs[i], xs[i + 1]
        z0b, z1b = bed[i], bed[i + 1]
        w0, w1 = b / 2 * wscale, b / 2 * wscale
        # channel bed
        _box(ax, [[(x0, -w0, z0b), (x1, -w1, z1b), (x1, w1, z1b), (x0, w0, z0b)]], "#8D6E63")
        # water surface and the two sloping banks under it
        ya, yb = y[i], y[i + 1]
        ws0, ws1 = half(ya) * wscale, half(yb) * wscale
        col = cm.Blues(0.35 + 0.6 * (1 - (ya - y.min()) / (y.max() - y.min() + 1e-9)))
        _box(ax, [[(x0, -ws0, z0b + ya), (x1, -ws1, z1b + yb), (x1, ws1, z1b + yb), (x0, ws0, z0b + ya)]], col, 0.55)
        for sgn in (-1, 1):
            _box(ax, [[(x0, sgn * w0, z0b), (x1, sgn * w1, z1b), (x1, sgn * ws1, z1b + yb), (x0, sgn * ws0, z0b + ya)]], "#A1887F", 0.8)
    ax.plot(xs, np.zeros_like(xs), bed + y, color="navy", lw=2, label="water surface (centreline)")
    ax.plot(xs, np.zeros_like(xs), bed + fl.open_channel(q, b, s0, n, m)["normal_depth_m"], "k--", lw=1.2, label="normal depth")
    wm = half(y.max()) * wscale
    ax.set_xlim(0, L); ax.set_ylim(-wm, wm); ax.set_zlim(0, bed.max() + y.max())
    ax.set_xlabel("distance along channel (m)"); ax.set_ylabel("width (not to scale)"); ax.set_zlabel("elevation (m)")
    ax.set_box_aspect((3.2, 0.9, 0.8)); ax.set_yticks([])
    ax.legend(loc="upper left")
    save(fig, "model_3d_channel_backwater.png")


def network_model():
    """3D model of the example pipe network: node height = hydraulic head, pipe thickness = flow, colour = head loss."""
    e = fl.EXAMPLE_NETWORK
    res = fl.pipe_network(e["pipes"], e["fixed_heads"], e["demands"])
    pos = e["positions"]
    heads = res["heads_m"]
    floor = min(heads.values()) - 0.6
    qmax = max(abs(p["flow_m3_s"]) for p in res["pipes"])
    lmax = max(abs(p["head_loss_m"]) for p in res["pipes"])
    fig = plt.figure(figsize=(12, 7))
    ax = ax3(fig, (1, 1, 1), "3D pipe-network model: node height = head (m), line width = flow, colour = head loss", 24, -62)
    for p in res["pipes"]:
        a, b = p["from"], p["to"]
        xs = [pos[a][0], pos[b][0]]
        ys = [pos[a][1], pos[b][1]]
        zs = [heads[a], heads[b]]
        ax.plot(xs, ys, zs, color=cm.magma(0.15 + 0.7 * abs(p["head_loss_m"]) / lmax), lw=2 + 9 * abs(p["flow_m3_s"]) / qmax)
        ax.text((xs[0] + xs[1]) / 2, (ys[0] + ys[1]) / 2, (zs[0] + zs[1]) / 2 + 0.07, f"{p['name']}  {p['flow_m3_s'] * 1e3:+.1f} L/s", fontsize=8)
    for node, h in heads.items():
        x, y = pos[node]
        ax.plot([x, x], [y, y], [floor, h], color="gray", lw=0.8, ls=":")
        fixed = node in e["fixed_heads"]
        ax.scatter([x], [y], [h], s=260 if fixed else 150, c="#1565C0" if fixed else "#2E7D32", edgecolor="k", depthshade=False)
        ax.text(x, y, h + 0.1, f"{node}\n{h:.2f} m", ha="center", fontsize=9, weight="bold")
    xx, yy = np.meshgrid([-0.3, 3.3], [-0.3, 2.3])
    ax.plot_surface(xx, yy, np.full_like(xx, floor), color="#ECEFF1", alpha=0.5)
    ax.set_xlabel("x (layout)"); ax.set_ylabel("y (layout)"); ax.set_zlabel("head (m)")
    ax.set_zlim(floor, max(heads.values()) + 0.45)
    save(fig, "model_3d_pipe_network.png")


def blasius_model():
    """3D boundary-layer surface u(x, y) on a flat plate (U = 5 m/s air) with the delta99 edge."""
    bl = fl.blasius()
    U, nu = 5.0, 1.5e-5
    x = np.linspace(0.02, 1.0, 90)
    y = np.linspace(0, 0.012, 80)
    X, Y = np.meshgrid(x, y)
    eta = Y * np.sqrt(U / (nu * X))
    Uu = np.interp(eta, bl["eta"], bl["fp"], right=1.0)
    fig = plt.figure(figsize=(12, 6))
    ax = ax3(fig, (1, 1, 1), "3D Blasius boundary layer on a flat plate: u/U over (x, y)", 25, -55)
    ax.plot_surface(X, Y * 1e3, Uu, cmap="viridis", linewidth=0, antialiased=True, alpha=0.95)
    d99 = bl["delta99"] * np.sqrt(nu * x / U) * 1e3
    ax.plot(x, d99, np.ones_like(x), color="red", lw=3, label="delta99 (edge, u = 0.99 U)")
    ax.plot(x, np.zeros_like(x), np.zeros_like(x), color="k", lw=2, label="plate")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (mm)"); ax.set_zlabel("u / U"); ax.legend(loc="upper left")
    ax.set_box_aspect((1.6, 1, 0.7))
    save(fig, "model_3d_boundary_layer.png")


def pump_model():
    """3D pump surface H(Q, speed ratio) with the system curve and the operating-point line."""
    base = dict(shutoff_head=40.0, design_flow=0.01, design_head=30.0, diameter=0.1, length=200.0, static_head=10.0)
    k = 10.0 / 0.01 ** 2
    q = np.linspace(0, 0.035, 70)
    s = np.linspace(0.7, 1.1, 50)
    Q, S = np.meshgrid(q, s)
    H = S ** 2 * 40.0 - k * Q ** 2
    area = math.pi * 0.1 ** 2 / 4
    sysh = np.array([10.0 + (fl.pipe_pressure_drop(0.1, 200.0, qq / area)["head_loss_m"] if qq > 0 else 0.0) for qq in q])
    fig = plt.figure(figsize=(12, 6.5))
    ax = ax3(fig, (1, 1, 1), "3D pump model: head surface vs flow and speed, with system curve and operating points", 24, -125)
    ax.plot_surface(Q * 1e3, S, np.clip(H, 0, None), cmap="plasma", alpha=0.8, linewidth=0)
    ax.plot(q * 1e3, np.full_like(q, 0.7), sysh, color="k", lw=3, label="system curve (10 m lift + friction)")
    ops = [fl.pump_operating_point(**base, speed_ratio=float(si)) for si in s]
    ax.plot([o["flow_m3_s"] * 1e3 for o in ops], s, [o["head_m"] for o in ops], color="lime", lw=4, label="operating points")
    ax.set_ylabel("speed ratio N / N_rated"); ax.set_xlabel("flow (L/s)"); ax.set_zlabel("head (m)"); ax.set_zlim(0, 50)
    ax.legend(loc="upper left")
    save(fig, "model_3d_pump_surface.png")


# ------------------------------------------------------------------ 3D comparison tables
def bars3d(ax, data, xlabels, ylabels, colors, zlabel, fmt="{:.3f}", label_size=7):
    """3D bar chart: data[iy][ix]; value printed on top of every bar."""
    for iy, row in enumerate(data):
        for ix, val in enumerate(row):
            ax.bar3d(ix, iy, 0, 0.6, 0.6, val, color=colors[iy], edgecolor="k", linewidth=0.3, alpha=0.92, shade=True)
            ax.text(ix + 0.3, iy + 0.3, val, fmt.format(val), ha="center", fontsize=label_size)
    ax.set_xticks(np.arange(len(xlabels)) + 0.3); ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_yticks(np.arange(len(ylabels)) + 0.3); ax.set_yticklabels(ylabels, fontsize=7, rotation=-15)
    ax.set_zlabel(zlabel)


def comparison_tables():
    """Four comparison tables as 3D bar charts (and as Markdown in comparison_tables.md): solver vs published reference."""
    md = []
    fig = plt.figure(figsize=(18, 13))

    # 1. natural convection Nusselt number
    ras = (1e3, 1e4, 1e5)
    ns = (25, 41, 61)
    rows = [[fl.DE_VAHL_DAVIS_NU[r] for r in ras]]
    for nn in ns:
        line = []
        for r in ras:
            key = ("conv", r, nn)
            if key not in _CACHE:
                c = fl.NaturalConvectionCavity(nn, r)
                c.run(tol=1e-6, max_time=20)
                _CACHE[key] = c
            line.append(_CACHE[key].nusselt("hot"))
        rows.append(line)
    ylab = ["de Vahl Davis 1983"] + [f"this solver {nn}x{nn}" for nn in ns]
    ax = ax3(fig, (2, 2, 1), "Natural convection: mean Nusselt number", 34, -118)
    bars3d(ax, rows, [f"Ra = {r:.0e}" for r in ras], ylab, ["#B0BEC5", "#90CAF9", "#42A5F5", "#1565C0"], "Nu", "{:.3f}")
    md.append("### Natural convection, Pr = 0.71 (mean Nusselt number, hot wall)\n\n| Source | " + " | ".join(f"Ra = {r:.0e}" for r in ras) + " |\n|---|" + "---|" * len(ras))
    for lab, row in zip(ylab, rows):
        md.append(f"| {lab} | " + " | ".join(f"{v:.3f}" for v in row) + " |")

    # 2. cavity stream-function minimum vs Ghia
    grids = (17, 33, 65)
    data, ylab2 = [], []
    for re in (100.0, 400.0):
        ref = abs(fl.GHIA_VORTEX[int(re)][2])
        data.append([ref] * len(grids)); ylab2.append(f"Ghia 1982, Re {int(re)}")
        data.append([abs(cavity(g, re).vortex_centre()[2]) for g in grids]); ylab2.append(f"this solver, Re {int(re)}")
    ax = ax3(fig, (2, 2, 2), "Lid-driven cavity: |psi_min| of the primary vortex", 34, -118)
    bars3d(ax, data, [f"{g}x{g}" for g in grids], ylab2, ["#C5E1A5", "#689F38", "#FFCC80", "#EF6C00"], "|psi_min|", "{:.4f}")
    md.append("\n### Lid-driven cavity, primary-vortex |psi_min|\n\n| Source | " + " | ".join(f"{g} x {g}" for g in grids) + " |\n|---|" + "---|" * len(grids))
    for lab, row in zip(ylab2, data):
        md.append(f"| {lab} | " + " | ".join(f"{v:.4f}" for v in row) + " |")

    # 3. Blasius
    bl = fl.blasius()
    lit = [0.33206, 4.910, 1.7208, 0.6641]
    mine = [bl["fpp0"], bl["delta99"], bl["delta_star"], bl["theta"]]
    # normalise by literature so bars are comparable; print the true values in the table
    ax = ax3(fig, (2, 2, 3), "Blasius boundary layer: this solver / literature value", 34, -118)
    bars3d(ax, [[1.0] * 4, [m / l for m, l in zip(mine, lit)]], ["f''(0)", "delta99", "delta*", "theta"],
           ["literature", "this solver"], ["#CE93D8", "#7B1FA2"], "ratio", "{:.4f}")
    ax.set_zlim(0, 1.25)
    md.append("\n### Blasius flat-plate boundary layer (similarity units)\n\n| Quantity | Literature | This solver | Ratio |\n|---|---|---|---|")
    for nm, l, m_ in zip(("f''(0)", "delta99 / (x/sqrt(Re_x))", "delta* / (x/sqrt(Re_x))", "theta / (x/sqrt(Re_x))"), lit, mine):
        md.append(f"| {nm} | {l:.4f} | {m_:.4f} | {m_ / l:.4f} |")

    # 4. Haaland vs Colebrook
    res_ = (1e4, 1e5, 1e6, 1e7)
    rough = (0.0, 1e-4, 1e-3, 1e-2)
    diff = [[100 * abs(fl.friction_factor_haaland(r, e) - fl.friction_factor_colebrook(r, e)) / fl.friction_factor_colebrook(r, e)
             for r in res_] for e in rough]
    ax = ax3(fig, (2, 2, 4), "Friction factor: |Haaland - Colebrook| / Colebrook (%)", 34, -118)
    bars3d(ax, diff, [f"Re = {r:.0e}" for r in res_], [f"e/D = {e:g}" for e in rough], ["#FFAB91", "#FF7043", "#E64A19", "#BF360C"], "% difference", "{:.2f}")
    md.append("\n### Haaland vs Colebrook friction factor, |difference| in %\n\n| e/D | " + " | ".join(f"Re = {r:.0e}" for r in res_) + " |\n|---|" + "---|" * len(res_))
    for e, row in zip(rough, diff):
        md.append(f"| {e:g} | " + " | ".join(f"{v:.2f}" for v in row) + " |")

    fig.suptitle("Comparison tables in 3D form: this solver against published references and exact solutions", weight="bold", fontsize=14)
    save(fig, "chart_3d_comparison_tables.png")
    (OUT / "comparison_tables.md").write_text("\n".join(md) + "\n")
    print("wrote comparison_tables.md")


ALL = {"mesh": mesh_views, "fields": field_views, "channel-model": channel_model, "network-model": network_model,
       "blasius": blasius_model, "pump": pump_model, "compare": comparison_tables}

if __name__ == "__main__":
    names = sys.argv[1:] or list(ALL)
    unknown = [n for n in names if n not in ALL]
    if unknown:
        sys.exit(f"unknown figure(s) {unknown}; choose from {sorted(ALL)}")
    for name in names:
        ALL[name]()
