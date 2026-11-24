"""FluidFlow CFD - small, validated 2D incompressible-flow toolkit in pure NumPy/SciPy.

Solvers
    cavity        lid-driven cavity, streamfunction-vorticity, 3rd-order upwind + SSP-RK3, DST Poisson solver
    taylor-green  periodic 2D flow, pseudo-spectral vorticity solver with exact Taylor-Green check
    shear         periodic double shear layer (roll-up into vortices), same spectral solver
    channel       transient plane Poiseuille flow, Crank-Nicolson, compared with the exact series
    pipe          Reynolds number, friction factor (laminar / Haaland / Colebrook), pressure drop, minor losses, pump power
    blasius       laminar flat-plate boundary layer (similarity solution, f''(0) = 0.33206)
    pump          pump curve vs system curve operating point (speed ratio, parallel pumps)
    open-channel  Manning normal and critical depth, Froude number
    backwater     gradually varied flow profile (M1/M2/supercritical) from a control depth
    jump          hydraulic jump: sequent depth, Froude numbers, energy loss
    network       looped pipe network: flows and heads by the global gradient algorithm (Darcy-Weisbach + minor losses)
    convection    differentially heated cavity (natural convection, Boussinesq), Nusselt number vs de Vahl Davis (1983)
    convergence   grid-convergence study of the cavity solver (observed order of accuracy)

Usage
    python main.py cavity --re 100 --n 65
    python main.py taylor-green --n 32 --nu 0.01 --t-end 2
    python main.py shear --n 128 --nu 1e-3 --t-end 6
    python main.py channel --nu 0.01 --g 1 --t-end 1
    python main.py pipe --diameter 0.05 --length 100 --velocity 2 --minor-k 2.5
    python main.py convergence --re 100 --grids 17 33 65
    python main.py convection --ra 1e5 --n 65
    python main.py cavity --re 100 --n 65 --aspect 2
    python main.py cavity --re 100 --n 65 --plot3d
    python main.py blasius --u-inf 5 --nu 1.5e-5 --x 0.5 --length 1
    python main.py network [--spec network.json]
    python main.py pump --shutoff-head 40 --design-flow 0.01 --design-head 30 --diameter 0.1 --length 200 --static-head 10
    python main.py open-channel --flow 5 --width 3 --slope 0.001
    python main.py backwater --flow 5 --width 3 --slope 0.001 --y-control 2 --length 3000
    python main.py jump --flow 5 --width 3 --y1 0.4
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
from scipy import fft as sfft
from scipy.linalg import solve_banded

__version__ = "1.6.0"
log = logging.getLogger("fluidflow")

# Ghia, Ghia & Shin (1982), J. Comput. Phys. 48: u-velocity on the vertical centreline x = 0.5 (y, u) for Re = 100, 400, 1000.
GHIA_U = {
    100: (
        (1.0000, 1.00000), (0.9766, 0.84123), (0.9688, 0.78871), (0.9609, 0.73722), (0.9531, 0.68717),
        (0.8516, 0.23151), (0.7344, 0.00332), (0.6172, -0.13641), (0.5000, -0.20581), (0.4531, -0.21090),
        (0.2813, -0.15662), (0.1719, -0.10150), (0.1016, -0.06434), (0.0703, -0.04775), (0.0625, -0.04192),
        (0.0547, -0.03717), (0.0000, 0.00000),
    ),
    400: (
        (1.0000, 1.00000), (0.9766, 0.75837), (0.9688, 0.68439), (0.9609, 0.61756), (0.9531, 0.55892),
        (0.8516, 0.29093), (0.7344, 0.16256), (0.6172, 0.02135), (0.5000, -0.11477), (0.4531, -0.17119),
        (0.2813, -0.32726), (0.1719, -0.24299), (0.1016, -0.14612), (0.0703, -0.10338), (0.0625, -0.09266),
        (0.0547, -0.08186), (0.0000, 0.00000),
    ),
    1000: (
        (1.0000, 1.00000), (0.9766, 0.65928), (0.9688, 0.57492), (0.9609, 0.51117), (0.9531, 0.46604),
        (0.8516, 0.33304), (0.7344, 0.18719), (0.6172, 0.05702), (0.5000, -0.06080), (0.4531, -0.10648),
        (0.2813, -0.27805), (0.1719, -0.38289), (0.1016, -0.29730), (0.0703, -0.22220), (0.0625, -0.20196),
        (0.0547, -0.18109), (0.0000, 0.00000),
    ),
}
GHIA_RE100_U = GHIA_U[100]  # backwards-compatible alias (v1.0)
# Primary-vortex (x, y, psi_min) from Ghia et al. (1982), used for reference only.
GHIA_VORTEX = {100: (0.6172, 0.7344, -0.1034), 400: (0.5547, 0.6055, -0.1139), 1000: (0.5313, 0.5625, -0.1179)}


def ghia_reference(re: float):
    """Ghia et al. centreline data for ``re`` (100, 400 or 1000), or ``None`` if there is no benchmark for it."""
    for key, data in GHIA_U.items():
        if abs(re - key) < 1e-9:
            return np.array(data)
    return None


# ---------------------------------------------------------------- lid-driven cavity
class LidDrivenCavity:
    """Unit square, lid (y = 1) moving with speed ``lid`` in +x, no-slip elsewhere, Re = lid * L / nu.

    Streamfunction-vorticity form (u = psi_y, v = -psi_x, omega = v_x - u_y = -laplacian(psi)):
        omega_t + u omega_x + v omega_y = (1/Re) laplacian(omega),     laplacian(psi) = -omega
    * Poisson equation solved exactly on the grid with a type-I discrete sine transform (psi = 0 on walls).
    * Wall vorticity from Thom's formula (second-order Taylor expansion of psi at the wall).
    * Advection: third-order upwind-biased differences; time stepping: SSP Runge-Kutta 3.
    """

    def __init__(self, n: int = 65, re: float = 100.0, lid: float = 1.0, aspect: float = 1.0):
        """``n`` nodes along the height (length 1); the width is ``aspect`` (same grid spacing in x and y)."""
        if n < 9:
            raise ValueError("n must be at least 9")
        if re <= 0:
            raise ValueError("re must be positive")
        if aspect <= 0:
            raise ValueError("aspect must be positive")
        self.n, self.re, self.lid, self.aspect = int(n), float(re), float(lid), float(aspect)
        self.h = 1.0 / (self.n - 1)
        self.nx = max(9, int(round(self.aspect * (self.n - 1))) + 1)
        self.nu = self.lid / self.re
        self.omega = np.zeros((self.n, self.nx))
        self.psi = np.zeros((self.n, self.nx))
        my = np.arange(1, self.n - 1)
        mx = np.arange(1, self.nx - 1)
        lam_y = (2.0 - 2.0 * np.cos(np.pi * my / (self.n - 1))) / self.h ** 2
        lam_x = (2.0 - 2.0 * np.cos(np.pi * mx / (self.nx - 1))) / self.h ** 2
        self._denom = lam_y[:, None] + lam_x[None, :]
        self.time = 0.0
        self.history: list[tuple[float, float]] = []

    def set_reynolds(self, re: float) -> None:
        """Change the Reynolds number but keep the current vorticity field (continuation / warm start)."""
        if re <= 0:
            raise ValueError("re must be positive")
        self.re = float(re)
        self.nu = self.lid / self.re
        self.time = 0.0
        self.history = []

    # -- helpers
    def _solve_psi(self, omega: np.ndarray) -> np.ndarray:
        psi = np.zeros_like(omega)
        hat = sfft.dstn(omega[1:-1, 1:-1], type=1, norm="ortho") / self._denom
        psi[1:-1, 1:-1] = sfft.idstn(hat, type=1, norm="ortho")
        return psi

    def _apply_wall_vorticity(self, omega: np.ndarray, psi: np.ndarray) -> None:
        h2 = self.h ** 2
        omega[-1, :] = -2.0 * psi[-2, :] / h2 - 2.0 * self.lid / self.h   # moving lid
        omega[0, :] = -2.0 * psi[1, :] / h2
        omega[:, 0] = -2.0 * psi[:, 1] / h2
        omega[:, -1] = -2.0 * psi[:, -2] / h2

    def velocity(self, psi: np.ndarray | None = None):
        """Velocity components on all nodes (walls: no-slip, lid: u = lid)."""
        psi = self.psi if psi is None else psi
        u = np.zeros_like(psi)
        v = np.zeros_like(psi)
        u[1:-1, 1:-1] = (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2 * self.h)
        v[1:-1, 1:-1] = -(psi[1:-1, 2:] - psi[1:-1, :-2]) / (2 * self.h)
        u[-1, :] = self.lid
        return u, v

    def _rhs(self, omega: np.ndarray) -> np.ndarray:
        h = self.h
        psi = self._solve_psi(omega)
        w = omega.copy()
        self._apply_wall_vorticity(w, psi)
        u, v = self.velocity(psi)
        p = np.pad(w, 2, mode="edge")
        c = p[2:-2, 2:-2]
        xp1, xp2, xm1, xm2 = p[2:-2, 3:-1], p[2:-2, 4:], p[2:-2, 1:-3], p[2:-2, :-4]
        yp1, yp2, ym1, ym2 = p[3:-1, 2:-2], p[4:, 2:-2], p[1:-3, 2:-2], p[:-4, 2:-2]
        cx = (-xp2 + 8 * xp1 - 8 * xm1 + xm2) / (12 * h)
        cy = (-yp2 + 8 * yp1 - 8 * ym1 + ym2) / (12 * h)
        dx4 = (xp2 - 4 * xp1 + 6 * c - 4 * xm1 + xm2) / (12 * h)
        dy4 = (yp2 - 4 * yp1 + 6 * c - 4 * ym1 + ym2) / (12 * h)
        adv = u * cx + np.abs(u) * dx4 + v * cy + np.abs(v) * dy4
        lap = (xp1 + xm1 + yp1 + ym1 - 4 * c) / h ** 2
        out = np.zeros_like(omega)
        out[1:-1, 1:-1] = (-adv + self.nu * lap)[1:-1, 1:-1]
        return out

    def stable_dt(self, cfl: float = 0.4, diff: float = 0.2) -> float:
        return min(cfl * self.h / (1.2 * self.lid), diff * self.h ** 2 / self.nu)

    def step(self, dt: float) -> float:
        w = self.omega
        w1 = w + dt * self._rhs(w)
        w2 = 0.75 * w + 0.25 * (w1 + dt * self._rhs(w1))
        w3 = w / 3.0 + 2.0 / 3.0 * (w2 + dt * self._rhs(w2))
        change = float(np.max(np.abs(w3[1:-1, 1:-1] - w[1:-1, 1:-1]))) / dt
        self.omega = w3
        self.time += dt
        return change

    def run(self, tol: float = 1e-5, max_time: float = 200.0, max_steps: int = 400_000, dt: float | None = None) -> dict:
        """March in pseudo-time until max|d omega / dt| < tol (steady state)."""
        dt = dt or self.stable_dt()
        residual, steps = np.inf, 0
        while steps < max_steps and self.time < max_time:
            residual = self.step(dt)
            steps += 1
            if steps % 50 == 0:
                self.history.append((self.time, residual))
                if steps % 5000 == 0:
                    log.info("cavity Re=%g n=%d: step %d, t=%.1f, residual %.2e", self.re, self.n, steps, self.time, residual)
            if not np.isfinite(residual):
                raise FloatingPointError("cavity solver diverged - lower dt or Reynolds number, or raise n")
            if residual < tol:
                break
        self.psi = self._solve_psi(self.omega)
        self._apply_wall_vorticity(self.omega, self.psi)
        return {"steps": steps, "time": round(self.time, 3), "residual": float(residual), "converged": bool(residual < tol)}

    # -- diagnostics
    @property
    def x(self) -> np.ndarray:
        return np.linspace(0.0, (self.nx - 1) * self.h, self.nx)

    @property
    def y(self) -> np.ndarray:
        return np.linspace(0.0, 1.0, self.n)

    def vortex_centre(self) -> tuple[float, float, float]:
        """Primary-vortex centre (x, y) and psi_min, with 3-point parabolic refinement."""
        j, i = np.unravel_index(np.argmin(self.psi), self.psi.shape)
        j = int(np.clip(j, 1, self.n - 2))
        i = int(np.clip(i, 1, self.nx - 2))

        def refine(a, b, c):
            den = a - 2 * b + c
            return 0.0 if abs(den) < 1e-14 else 0.5 * (a - c) / den

        dy = refine(self.psi[j - 1, i], self.psi[j, i], self.psi[j + 1, i])
        dx = refine(self.psi[j, i - 1], self.psi[j, i], self.psi[j, i + 1])
        return (i + dx) * self.h, (j + dy) * self.h, float(self.psi[j, i])

    def centreline_u(self):
        """u(y) along the vertical centreline x = width / 2 (interpolated between the two middle columns if nx is even)."""
        u, _ = self.velocity()
        mid = self.nx // 2
        col = u[:, mid] if self.nx % 2 == 1 else 0.5 * (u[:, mid - 1] + u[:, mid])
        return self.y, col

    def divergence_max(self) -> float:
        """max |u_x + v_y| on interior nodes (mass conservation check; ~0 because u, v come from psi)."""
        u, v = self.velocity()
        h = self.h
        div = (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * h) + (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * h)
        return float(np.max(np.abs(div[1:-1, 1:-1])))


def ghia_error(cav: LidDrivenCavity) -> dict:
    """RMS / max deviation from the Ghia et al. (1982) centreline u data (Re = 100, 400 or 1000)."""
    ref = ghia_reference(cav.re)
    if ref is None or cav.nx != cav.n:
        raise ValueError("Ghia et al. (1982) benchmark data exist only for the square cavity at Re = 100, 400 and 1000")
    y, u = cav.centreline_u()
    err = np.interp(ref[:, 0], y, u) - ref[:, 1]
    return {"rms": float(np.sqrt(np.mean(err ** 2))), "max": float(np.max(np.abs(err)))}


def cavity_convergence(re: float = 100.0, grids=(17, 33, 65), tol: float = 1e-6) -> dict:
    """Grid-convergence study: psi_min and centreline error on successively refined grids, with observed order.

    The observed order uses the three finest grids (refinement ratio 2) on psi_min: p = log2(|d1| / |d2|).
    """
    rows = []
    for n in grids:
        cav = LidDrivenCavity(n, re)
        info = cav.run(tol=tol)
        _, _, pmin = cav.vortex_centre()
        row = {"n": int(n), "psi_min": pmin, "converged": info["converged"]}
        if ghia_reference(re) is not None:
            row["ghia_rms"] = ghia_error(cav)["rms"]
        rows.append(row)
    order = None
    if len(rows) >= 3:
        d1 = rows[-2]["psi_min"] - rows[-3]["psi_min"]
        d2 = rows[-1]["psi_min"] - rows[-2]["psi_min"]
        if d1 != 0 and d2 != 0 and d1 / d2 > 0:
            order = float(math.log2(abs(d1 / d2)))
    return {"re": re, "rows": rows, "observed_order": order}


# ---------------------------------------------------------------- heat transport: differentially heated cavity
# de Vahl Davis (1983), Int. J. Numer. Methods Fluids 3, Pr = 0.71: average Nusselt number on the hot wall, and
# (u_max on the vertical centreline, v_max on the horizontal centreline), in units of alpha / H.
DE_VAHL_DAVIS_NU = {1e3: 1.118, 1e4: 2.243, 1e5: 4.519, 1e6: 8.800}
DE_VAHL_DAVIS_VMAX = {1e3: (3.649, 3.697), 1e4: (16.178, 19.617), 1e5: (34.73, 68.59)}


_trapezoid = getattr(np, "trapezoid", None) or np.trapz    # np.trapezoid needs NumPy >= 2.0


def _upwind3(f: np.ndarray, u: np.ndarray, v: np.ndarray, h: float) -> np.ndarray:
    """u f_x + v f_y with third-order upwind-biased differences (edge-padded, so only interior values are meaningful)."""
    p = np.pad(f, 2, mode="edge")
    c = p[2:-2, 2:-2]
    xp1, xp2, xm1, xm2 = p[2:-2, 3:-1], p[2:-2, 4:], p[2:-2, 1:-3], p[2:-2, :-4]
    yp1, yp2, ym1, ym2 = p[3:-1, 2:-2], p[4:, 2:-2], p[1:-3, 2:-2], p[:-4, 2:-2]
    cx = (-xp2 + 8 * xp1 - 8 * xm1 + xm2) / (12 * h)
    cy = (-yp2 + 8 * yp1 - 8 * ym1 + ym2) / (12 * h)
    dx4 = (xp2 - 4 * xp1 + 6 * c - 4 * xm1 + xm2) / (12 * h)
    dy4 = (yp2 - 4 * yp1 + 6 * c - 4 * ym1 + ym2) / (12 * h)
    return u * cx + np.abs(u) * dx4 + v * cy + np.abs(v) * dy4


class NaturalConvectionCavity:
    """Differentially heated cavity (Boussinesq natural convection) of height 1 and width ``aspect``.

    Left wall hot (T = 1), right wall cold (T = 0), top and bottom adiabatic, no-slip everywhere. Lengths are scaled by the
    height H and velocity by alpha / H, so the only parameters are the Rayleigh and Prandtl numbers:

        omega_t + u omega_x + v omega_y = Pr laplacian(omega) + Ra Pr T_x        (omega = v_x - u_y = -laplacian(psi))
        T_t     + u T_x     + v T_y     = laplacian(T)

    Same numerics as :class:`LidDrivenCavity` (DST Poisson solver, Thom wall vorticity, 3rd-order upwind, SSP-RK3) with an
    adaptive pseudo-time step. The Nusselt number is Nu = W * mean(-T_x) on a wall of width W = ``aspect`` (conduction: Nu = 1).
    """

    def __init__(self, n: int = 41, ra: float = 1e4, pr: float = 0.71, aspect: float = 1.0):
        if n < 9:
            raise ValueError("n must be at least 9")
        if ra <= 0 or pr <= 0 or aspect <= 0:
            raise ValueError("ra, pr and aspect must be positive")
        self.n, self.ra, self.pr, self.aspect = int(n), float(ra), float(pr), float(aspect)
        self.ny = self.n
        self.nx = max(9, int(round(self.aspect * (self.n - 1))) + 1)
        self.h = 1.0 / (self.n - 1)
        self.width = (self.nx - 1) * self.h
        self.x = np.linspace(0.0, self.width, self.nx)
        self.y = np.linspace(0.0, 1.0, self.ny)
        self.omega = np.zeros((self.ny, self.nx))
        self.psi = np.zeros((self.ny, self.nx))
        self.T = np.tile(1.0 - self.x / self.width, (self.ny, 1))        # start from pure conduction
        lam_y = (2.0 - 2.0 * np.cos(np.pi * np.arange(1, self.ny - 1) / (self.ny - 1))) / self.h ** 2
        lam_x = (2.0 - 2.0 * np.cos(np.pi * np.arange(1, self.nx - 1) / (self.nx - 1))) / self.h ** 2
        self._denom = lam_y[:, None] + lam_x[None, :]
        self.time = 0.0
        self.history: list[tuple[float, float]] = []

    def _solve_psi(self, omega: np.ndarray) -> np.ndarray:
        psi = np.zeros_like(omega)
        hat = sfft.dstn(omega[1:-1, 1:-1], type=1, norm="ortho") / self._denom
        psi[1:-1, 1:-1] = sfft.idstn(hat, type=1, norm="ortho")
        return psi

    def _wall_vorticity(self, omega: np.ndarray, psi: np.ndarray) -> None:
        h2 = self.h ** 2
        omega[-1, :] = -2.0 * psi[-2, :] / h2
        omega[0, :] = -2.0 * psi[1, :] / h2
        omega[:, 0] = -2.0 * psi[:, 1] / h2
        omega[:, -1] = -2.0 * psi[:, -2] / h2

    def _temperature_bc(self, T: np.ndarray) -> None:
        T[0, :] = (4.0 * T[1, :] - T[2, :]) / 3.0          # adiabatic (second-order Neumann)
        T[-1, :] = (4.0 * T[-2, :] - T[-3, :]) / 3.0
        T[:, 0] = 1.0                                      # hot wall
        T[:, -1] = 0.0                                     # cold wall

    def velocity(self, psi: np.ndarray | None = None):
        psi = self.psi if psi is None else psi
        u = np.zeros_like(psi)
        v = np.zeros_like(psi)
        u[1:-1, 1:-1] = (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2 * self.h)
        v[1:-1, 1:-1] = -(psi[1:-1, 2:] - psi[1:-1, :-2]) / (2 * self.h)
        return u, v

    def _rhs(self, omega: np.ndarray, T: np.ndarray):
        h = self.h
        psi = self._solve_psi(omega)
        w = omega.copy()
        self._wall_vorticity(w, psi)
        u, v = self.velocity(psi)
        pw = np.pad(w, 1, mode="edge")
        pt = np.pad(T, 1, mode="edge")
        lap_w = (pw[1:-1, 2:] + pw[1:-1, :-2] + pw[2:, 1:-1] + pw[:-2, 1:-1] - 4 * w) / h ** 2
        lap_t = (pt[1:-1, 2:] + pt[1:-1, :-2] + pt[2:, 1:-1] + pt[:-2, 1:-1] - 4 * T) / h ** 2
        tx = (pt[1:-1, 2:] - pt[1:-1, :-2]) / (2 * h)
        dw = np.zeros_like(omega)
        dt_ = np.zeros_like(T)
        dw[1:-1, 1:-1] = (-_upwind3(w, u, v, h) + self.pr * lap_w + self.ra * self.pr * tx)[1:-1, 1:-1]
        dt_[1:-1, 1:-1] = (-_upwind3(T, u, v, h) + lap_t)[1:-1, 1:-1]
        return dw, dt_

    def stable_dt(self, umax: float, cfl: float = 0.4, diff: float = 0.2) -> float:
        return min(cfl * self.h / max(umax, 1e-9), diff * self.h ** 2 / max(self.pr, 1.0))

    def step(self, dt: float) -> float:
        w, T = self.omega, self.T

        def stage(w0, T0, w1, T1, a, b):
            dw, dT = self._rhs(w1, T1)
            wn = a * w0 + b * (w1 + dt * dw)
            Tn = a * T0 + b * (T1 + dt * dT)
            self._temperature_bc(Tn)
            return wn, Tn

        w1, T1 = stage(w, T, w, T, 0.0, 1.0)
        w2, T2 = stage(w, T, w1, T1, 0.75, 0.25)
        w3, T3 = stage(w, T, w2, T2, 1.0 / 3.0, 2.0 / 3.0)
        rate_w = float(np.max(np.abs(w3 - w))) / dt / max(1.0, float(np.max(np.abs(w3))))
        rate_t = float(np.max(np.abs(T3 - T))) / dt
        self.omega, self.T = w3, T3
        self.time += dt
        return max(rate_w, rate_t)

    def run(self, tol: float = 1e-6, max_time: float = 20.0, max_steps: int = 400_000) -> dict:
        """March in pseudo-time until the relative rate of change of omega and T drops below ``tol``."""
        residual, steps, dt = np.inf, 0, 1e-4
        while steps < max_steps and self.time < max_time:
            if steps % 25 == 0:
                u, v = self.velocity(self._solve_psi(self.omega))
                dt = self.stable_dt(float(max(np.max(np.abs(u)), np.max(np.abs(v)))))
            residual = self.step(dt)
            steps += 1
            if steps % 50 == 0:
                self.history.append((self.time, residual))
                if steps % 5000 == 0:
                    log.info("convection Ra=%g n=%d: step %d, t=%.3f, residual %.2e", self.ra, self.n, steps, self.time, residual)
            if not np.isfinite(residual):
                raise FloatingPointError("convection solver diverged - raise n or lower Ra")
            if residual < tol:
                break
        self.psi = self._solve_psi(self.omega)
        self._wall_vorticity(self.omega, self.psi)
        return {"steps": steps, "time": round(self.time, 4), "residual": float(residual), "converged": bool(residual < tol)}

    # -- diagnostics
    def nusselt_local(self, wall: str = "hot") -> np.ndarray:
        """Local heat flux -dT/dx (hot wall) or the flux entering the cold wall, one value per y node (second order)."""
        T, h = self.T, self.h
        if wall == "hot":
            return (3 * T[:, 0] - 4 * T[:, 1] + T[:, 2]) / (2 * h)
        if wall == "cold":
            return (T[:, -3] - 4 * T[:, -2] + 3 * T[:, -1]) / (2 * h) * -1.0
        raise ValueError("wall must be 'hot' or 'cold'")

    def nusselt(self, wall: str = "hot") -> float:
        """Average Nusselt number Nu = W * integral(-T_x dy) / H (trapezoid rule, H = 1)."""
        return float(self.width * _trapezoid(self.nusselt_local(wall), self.y))

    def max_velocities(self) -> tuple[float, float]:
        """(max u on the vertical centreline, max v on the horizontal centreline)."""
        u, v = self.velocity()
        xm, ym = self.nx // 2, self.ny // 2
        uc = u[:, xm] if self.nx % 2 else 0.5 * (u[:, xm - 1] + u[:, xm])
        vc = v[ym, :] if self.ny % 2 else 0.5 * (v[ym - 1, :] + v[ym, :])
        return float(np.max(uc)), float(np.max(vc))

    def heat_balance_error(self) -> float:
        """|Nu_hot - Nu_cold| / Nu_hot: at steady state the heat leaving the hot wall must enter the cold wall."""
        return abs(self.nusselt("hot") - self.nusselt("cold")) / self.nusselt("hot")


# ---------------------------------------------------------------- periodic pseudo-spectral solver
class Spectral2D:
    """Vorticity equation on the periodic box [0, 2 pi)^2, 2/3 de-aliasing, integrating-factor RK4.

        omega_t + u omega_x + v omega_y = nu laplacian(omega),    u = psi_y, v = -psi_x, laplacian(psi) = -omega
    """

    def __init__(self, n: int = 64, nu: float = 1e-3):
        if n % 2 or n < 8:
            raise ValueError("n must be an even integer >= 8")
        self.n, self.nu = int(n), float(nu)
        x = 2 * np.pi * np.arange(n) / n
        self.x, self.y = np.meshgrid(x, x)           # arrays are indexed [j (y), i (x)]
        k = sfft.fftfreq(n, 1.0 / n)
        self.kx, self.ky = np.meshgrid(k, k)
        self.k2 = self.kx ** 2 + self.ky ** 2
        self.k2_inv = np.where(self.k2 > 0, 1.0 / np.where(self.k2 > 0, self.k2, 1.0), 0.0)
        self.mask = (np.abs(self.kx) < n / 3) & (np.abs(self.ky) < n / 3)
        self.wh = np.zeros((n, n), dtype=complex)
        self.time = 0.0

    def set_vorticity(self, omega: np.ndarray) -> None:
        self.wh = sfft.fft2(omega) * self.mask
        self.time = 0.0

    def set_velocity(self, u: np.ndarray, v: np.ndarray) -> None:
        self.wh = (1j * self.kx * sfft.fft2(v) - 1j * self.ky * sfft.fft2(u)) * self.mask
        self.time = 0.0

    def fields(self, wh: np.ndarray | None = None):
        wh = self.wh if wh is None else wh
        psih = wh * self.k2_inv
        u = sfft.ifft2(1j * self.ky * psih).real
        v = sfft.ifft2(-1j * self.kx * psih).real
        return sfft.ifft2(wh).real, u, v

    def _nonlinear(self, wh: np.ndarray) -> np.ndarray:
        psih = wh * self.k2_inv
        u = sfft.ifft2(1j * self.ky * psih).real
        v = sfft.ifft2(-1j * self.kx * psih).real
        wx = sfft.ifft2(1j * self.kx * wh).real
        wy = sfft.ifft2(1j * self.ky * wh).real
        return -sfft.fft2(u * wx + v * wy) * self.mask

    def step(self, dt: float) -> None:
        e = np.exp(-self.nu * self.k2 * dt / 2)
        e2 = e * e
        w = self.wh
        k1 = self._nonlinear(w)
        k2 = self._nonlinear(e * (w + 0.5 * dt * k1))
        k3 = self._nonlinear(e * w + 0.5 * dt * k2)
        k4 = self._nonlinear(e2 * w + dt * e * k3)
        self.wh = e2 * w + dt / 6.0 * (e2 * k1 + 2 * e * (k2 + k3) + k4)
        self.time += dt

    def energy(self) -> float:
        _, u, v = self.fields()
        return float(0.5 * np.mean(u ** 2 + v ** 2))

    def enstrophy(self) -> float:
        w, _, _ = self.fields()
        return float(0.5 * np.mean(w ** 2))

    def run(self, t_end: float, dt: float, record_every: int = 1, snapshots: tuple = ()) -> dict:
        steps = max(1, int(round(t_end / dt)))
        t, e, z = [self.time], [self.energy()], [self.enstrophy()]
        snaps = {}
        want = {int(round(s / dt)): s for s in snapshots}
        if 0 in want:
            snaps[want[0]] = self.fields()[0]
        for k in range(1, steps + 1):
            self.step(dt)
            if k % record_every == 0 or k == steps:
                t.append(self.time); e.append(self.energy()); z.append(self.enstrophy())
            if k in want:
                snaps[want[k]] = self.fields()[0]
            if not np.isfinite(e[-1]):
                raise FloatingPointError("spectral solver diverged - reduce dt")
        return {"t": np.array(t), "energy": np.array(e), "enstrophy": np.array(z), "snapshots": snaps}


def taylor_green(n: int = 32, nu: float = 0.01, t_end: float = 2.0, dt: float = 0.01) -> dict:
    """Taylor-Green vortex: omega = -2 cos x cos y exp(-2 nu t). Returns the max-norm error against that exact solution."""
    sim = Spectral2D(n, nu)
    sim.set_vorticity(-2 * np.cos(sim.x) * np.cos(sim.y))
    res = sim.run(t_end, dt)
    exact = -2 * np.cos(sim.x) * np.cos(sim.y) * np.exp(-2 * nu * sim.time)
    err = float(np.max(np.abs(sim.fields()[0] - exact)))
    e_exact = 0.25 * np.exp(-4 * nu * res["t"])
    return {"error_max": err, "energy_error_max": float(np.max(np.abs(res["energy"] - e_exact))), "t": res["t"],
            "energy": res["energy"], "energy_exact": e_exact, "n": n, "nu": nu}


def shear_layer(n: int = 128, nu: float = 1e-3, t_end: float = 6.0, dt: float = 0.01, rho: float = 30.0,
                delta: float = 0.05, snapshots=(0.0, 2.0, 4.0, 6.0)) -> dict:
    """Double shear layer (Brown & Minion style): thin layers roll up into vortices after a small perturbation."""
    sim = Spectral2D(n, nu)
    y, x = sim.y, sim.x
    u = np.where(y <= np.pi, np.tanh(rho * (y - np.pi / 2)), np.tanh(rho * (3 * np.pi / 2 - y)))
    v = delta * np.sin(x)
    sim.set_velocity(u, v)
    res = sim.run(t_end, dt, record_every=5, snapshots=tuple(snapshots))
    res["x"], res["y"] = sim.x, sim.y
    res["n"], res["nu"] = n, nu
    return res


# ---------------------------------------------------------------- plane Poiseuille flow (transient)
def poiseuille_exact(y: np.ndarray, t: float, nu: float, g: float, height: float = 1.0, terms: int = 400) -> np.ndarray:
    """Exact start-up solution of u_t = g + nu u_yy, u(0)=u(H)=0, u(y,0)=0 (g = -dp/dx / rho)."""
    u = g / (2 * nu) * y * (height - y)
    for n in range(1, 2 * terms, 2):
        u = u - 4 * g * height ** 2 / (nu * np.pi ** 3 * n ** 3) * np.sin(n * np.pi * y / height) * np.exp(
            -nu * (n * np.pi / height) ** 2 * t)
    return u


def poiseuille_transient(ny: int = 101, nu: float = 0.01, g: float = 1.0, height: float = 1.0,
                         t_end: float = 1.0, dt: float = 1e-3) -> dict:
    """Crank-Nicolson finite differences (second order in space and time) for the start-up of channel flow."""
    y = np.linspace(0.0, height, ny)
    h = y[1] - y[0]
    m = ny - 2
    r = nu * dt / (2 * h ** 2)
    ab = np.zeros((3, m))
    ab[0, 1:] = -r
    ab[1, :] = 1 + 2 * r
    ab[2, :-1] = -r
    u = np.zeros(ny)
    steps = max(1, int(round(t_end / dt)))
    for _ in range(steps):
        inner = u[1:-1]
        rhs = inner + r * (u[2:] - 2 * inner + u[:-2]) + dt * g
        u[1:-1] = solve_banded((1, 1), ab, rhs)
    t = steps * dt
    exact = poiseuille_exact(y, t, nu, g, height)
    return {"y": y, "u": u, "u_exact": exact, "t": t, "error_max": float(np.max(np.abs(u - exact))),
            "u_max_steady": g * height ** 2 / (8 * nu)}


# ---------------------------------------------------------------- pipe flow engineering helpers
def reynolds(rho: float, velocity: float, diameter: float, mu: float) -> float:
    if min(rho, diameter, mu) <= 0:
        raise ValueError("rho, diameter and mu must be positive")
    return rho * abs(velocity) * diameter / mu


def friction_factor_haaland(re: float, rel_rough: float = 0.0) -> float:
    """Darcy friction factor, Haaland explicit approximation (turbulent flow)."""
    return (-1.8 * math.log10((rel_rough / 3.7) ** 1.11 + 6.9 / re)) ** -2


def friction_factor_colebrook(re: float, rel_rough: float = 0.0, tol: float = 1e-12) -> float:
    """Darcy friction factor from the implicit Colebrook-White equation (fixed-point iteration)."""
    x = 1 / math.sqrt(friction_factor_haaland(re, rel_rough))
    for _ in range(100):
        new = -2.0 * math.log10(rel_rough / 3.7 + 2.51 * x / re)
        if abs(new - x) < tol:
            x = new
            break
        x = new
    return 1 / x ** 2


def friction_factor(re: float, rel_rough: float = 0.0) -> float:
    """Laminar: 64/Re. Turbulent: Colebrook. 2300 < Re < 4000 is a transitional (uncertain) range."""
    if re <= 0:
        raise ValueError("Reynolds number must be positive")
    if re < 2300:
        return 64.0 / re
    return friction_factor_colebrook(re, rel_rough)


def pipe_pressure_drop(diameter: float, length: float, velocity: float, rho: float = 998.0, mu: float = 1.0e-3,
                       roughness: float = 4.5e-5, minor_k: float = 0.0, efficiency: float = 1.0) -> dict:
    """Darcy-Weisbach pressure drop of a straight circular pipe, plus optional minor losses.

    ``minor_k`` is the summed loss coefficient of fittings (bends, valves, ...): dp_minor = K * 0.5 * rho * V^2.
    ``efficiency`` (0 < eta <= 1) converts hydraulic power to the shaft power a pump must supply.
    """
    if min(diameter, length) <= 0:
        raise ValueError("diameter and length must be positive")
    if roughness < 0 or minor_k < 0:
        raise ValueError("roughness and minor_k must be non-negative")
    if not 0 < efficiency <= 1:
        raise ValueError("efficiency must be in (0, 1]")
    re = reynolds(rho, velocity, diameter, mu)
    f = friction_factor(re, roughness / diameter)
    dyn = 0.5 * rho * velocity ** 2
    dp_major = f * length / diameter * dyn
    dp_minor = minor_k * dyn
    dp = dp_major + dp_minor
    q = abs(velocity) * math.pi * diameter ** 2 / 4.0
    regime = "laminar" if re < 2300 else ("transitional (uncertain)" if re < 4000 else "turbulent")
    return {"reynolds": re, "regime": regime, "friction_factor": f, "pressure_drop_pa": dp,
            "pressure_drop_major_pa": dp_major, "pressure_drop_minor_pa": dp_minor,
            "head_loss_m": dp / (rho * 9.80665), "flow_rate_m3_s": q, "pump_power_w": dp * q / efficiency}


# ---------------------------------------------------------------- Blasius flat-plate boundary layer
_trapz = getattr(np, "trapezoid", None) or np.trapz  # np.trapezoid needs NumPy >= 2.0


def blasius(eta_max: float = 10.0, n: int = 401) -> dict:
    """Laminar flat-plate boundary layer: f''' + f f''/2 = 0, f(0) = f'(0) = 0, f'(inf) = 1 (shooting on f''(0)).

    Returns the similarity profile plus the classical integral quantities (all in units of x / sqrt(Re_x)):
    wall shear f''(0) (= 0.33206), delta_99 (= 4.91), displacement and momentum thickness (1.7208, 0.6641).
    """
    from scipy.integrate import solve_ivp
    from scipy.optimize import brentq

    if eta_max < 6 or n < 20:
        raise ValueError("eta_max must be >= 6 and n >= 20")

    def rhs(_, s):
        return [s[1], s[2], -0.5 * s[0] * s[2]]

    def shoot(fpp0, dense=None):
        return solve_ivp(rhs, (0.0, eta_max), [0.0, 0.0, fpp0], rtol=1e-11, atol=1e-12, t_eval=dense)

    fpp0 = brentq(lambda g: shoot(g).y[1, -1] - 1.0, 0.1, 1.0, xtol=1e-14)
    eta = np.linspace(0.0, eta_max, n)
    sol = shoot(fpp0, eta)
    f, fp, fpp = sol.y
    delta99 = float(np.interp(0.99, fp, eta))
    return {"eta": eta, "f": f, "fp": fp, "fpp": fpp, "fpp0": float(fpp0), "delta99": delta99,
            "delta_star": float(eta[-1] - f[-1]), "theta": float(_trapz(fp * (1 - fp), eta)),
            "cf_sqrt_re": 2.0 * float(fpp0)}


def flat_plate(u_inf: float, nu: float, x: float, length: float | None = None) -> dict:
    """Dimensional Blasius quantities at distance x from the leading edge (laminar only: Re_x < ~5e5)."""
    if min(u_inf, nu, x) <= 0:
        raise ValueError("u_inf, nu and x must be positive")
    bl = blasius()
    re_x = u_inf * x / nu
    s = x / math.sqrt(re_x)
    out = {"re_x": re_x, "regime": "laminar" if re_x < 5e5 else "beyond laminar range (Blasius not valid)",
           "delta99_m": bl["delta99"] * s, "displacement_thickness_m": bl["delta_star"] * s,
           "momentum_thickness_m": bl["theta"] * s, "cf_local": bl["cf_sqrt_re"] / math.sqrt(re_x),
           "wall_shear_per_rho_m2_s2": 0.5 * u_inf ** 2 * bl["cf_sqrt_re"] / math.sqrt(re_x)}
    if length is not None:
        if length <= 0:
            raise ValueError("length must be positive")
        re_l = u_inf * length / nu
        out["drag_coefficient_plate"] = 2.0 * bl["cf_sqrt_re"] / math.sqrt(re_l)  # C_D = 1.328 / sqrt(Re_L)
        out["re_length"] = re_l
    return out


# ---------------------------------------------------------------- pipe networks (global gradient algorithm)
EXAMPLE_NETWORK = {
    "fixed_heads": {"R": 50.0},
    "demands": {"B": 0.004, "C": 0.003, "D": 0.005},
    "positions": {"R": [0, 1], "A": [1, 1], "B": [2, 2], "C": [2, 0], "D": [3, 1]},  # only used for the plot
    "pipes": [
        {"name": "P1", "from": "R", "to": "A", "diameter": 0.20, "length": 400},
        {"name": "P2", "from": "A", "to": "B", "diameter": 0.12, "length": 300},
        {"name": "P3", "from": "A", "to": "C", "diameter": 0.12, "length": 350},
        {"name": "P4", "from": "B", "to": "D", "diameter": 0.10, "length": 250},
        {"name": "P5", "from": "C", "to": "D", "diameter": 0.10, "length": 300},
        {"name": "P6", "from": "B", "to": "C", "diameter": 0.08, "length": 200, "minor_k": 3.0},
    ],
}


def pipe_network(pipes: list, fixed_heads: dict, demands: dict, rho: float = 998.0, mu: float = 1.0e-3,
                 roughness: float = 4.5e-5, tol: float = 1e-10, max_iter: int = 200) -> dict:
    """Steady flows and heads of a looped pipe network (Darcy-Weisbach, Colebrook friction, minor losses).

    pipes        list of dicts: from, to, diameter, length, optional roughness, minor_k, name (flow positive from -> to)
    fixed_heads  {node: head in m} (reservoirs; at least one)
    demands      {node: outflow in m3/s} (negative = inflow); nodes not listed have zero demand
    Solved with the Todini-Pilati global gradient algorithm (Newton on all flows and heads together), so no loops have
    to be chosen by hand. Continuity at every free node and the energy balance around every loop hold to ``tol``.
    """
    if not pipes or not fixed_heads:
        raise ValueError("a network needs pipes and at least one fixed-head node")
    names = sorted({p[k] for p in pipes for k in ("from", "to")})
    for node in list(fixed_heads) + list(demands):
        if node not in names:
            raise ValueError(f"node '{node}' is not connected to any pipe")
    if set(fixed_heads) & set(demands):
        raise ValueError("a node cannot have both a fixed head and a demand")
    free = [n for n in names if n not in fixed_heads]
    if not free:
        raise ValueError("all nodes have fixed heads - nothing to solve")
    fidx = {n: i for i, n in enumerate(free)}
    npipe, nfree = len(pipes), len(free)
    a12 = np.zeros((npipe, nfree))
    a10 = np.zeros(npipe)
    d = np.empty(npipe)
    length = np.empty(npipe)
    rough = np.empty(npipe)
    kmin = np.empty(npipe)
    for i, p in enumerate(pipes):
        if p["from"] == p["to"]:
            raise ValueError("a pipe cannot start and end at the same node")
        d[i], length[i] = float(p["diameter"]), float(p["length"])
        if d[i] <= 0 or length[i] <= 0:
            raise ValueError("pipe diameter and length must be positive")
        rough[i] = float(p.get("roughness", roughness))
        kmin[i] = float(p.get("minor_k", 0.0))
        if rough[i] < 0 or kmin[i] < 0:
            raise ValueError("roughness and minor_k must be non-negative")
        for node, sign in ((p["from"], -1.0), (p["to"], 1.0)):
            if node in fidx:
                a12[i, fidx[node]] = sign
            else:
                a10[i] += sign * float(fixed_heads[node])
    q_dem = np.array([float(demands.get(n, 0.0)) for n in free])
    g = 9.80665
    area = math.pi * d ** 2 / 4.0

    def resistance(q):
        r = np.empty(npipe)
        for i in range(npipe):
            re = max(rho * abs(q[i]) * d[i] / (mu * area[i]), 1.0)
            r[i] = (friction_factor(re, rough[i] / d[i]) * length[i] / d[i] + kmin[i]) * 8.0 / (g * math.pi ** 2 * d[i] ** 4)
        return r

    q = 0.3 * area  # initial guess: 0.3 m/s in every pipe (the algorithm is insensitive to this)
    heads = np.zeros(nfree)
    converged, it, err = False, 0, math.inf
    for it in range(1, max_iter + 1):
        a11 = resistance(q) * np.maximum(np.abs(q), 1e-9)   # head loss per unit flow, R|Q|
        ninv = 0.5                                           # exponent n = 2
        a21 = a12.T
        inv_a11 = 1.0 / a11
        m = a21 @ (a12 * (ninv * inv_a11)[:, None])
        rhs = a21 @ (ninv * (q + inv_a11 * a10)) + (q_dem - a21 @ q)
        heads = np.linalg.solve(m, -rhs)
        q_new = (1.0 - ninv) * q - ninv * inv_a11 * (a12 @ heads + a10)
        err = float(np.sum(np.abs(q_new - q)) / max(np.sum(np.abs(q_new)), 1e-30))
        q = q_new
        if err < tol:
            converged = True
            break
    head_all = {n: float(heads[fidx[n]]) if n in fidx else float(fixed_heads[n]) for n in names}
    hl = resistance(q) * q * np.abs(q)
    return {
        "converged": converged, "iterations": it, "relative_change": err, "heads_m": head_all,
        "pipes": [{"name": p.get("name", f"{p['from']}-{p['to']}"), "from": p["from"], "to": p["to"],
                   "flow_m3_s": float(q[i]), "velocity_m_s": float(q[i] / area[i]), "head_loss_m": float(hl[i])}
                  for i, p in enumerate(pipes)],
        "mass_residual_m3_s": float(np.max(np.abs(a12.T @ q - q_dem))),
        "energy_residual_m": float(np.max(np.abs(hl + a12 @ heads + a10))),
        "net_outflow_fixed_nodes_m3_s": float(sum(q[i] for i, p in enumerate(pipes) if p["from"] in fixed_heads)
                                              - sum(q[i] for i, p in enumerate(pipes) if p["to"] in fixed_heads)),
    }


# ---------------------------------------------------------------- pump operating point
def pump_operating_point(shutoff_head: float, design_flow: float, design_head: float, diameter: float, length: float,
                         static_head: float = 0.0, minor_k: float = 0.0, roughness: float = 4.5e-5, rho: float = 998.0,
                         mu: float = 1.0e-3, speed_ratio: float = 1.0, n_parallel: int = 1, efficiency: float = 0.7) -> dict:
    """Intersection of a pump curve with a pipe system curve.

    Pump curve (one pump): H = s^2 * H0 - k Q^2, with k fitted so that it passes through (design_flow, design_head) at
    speed_ratio s = 1 (affinity laws scale H0 with s^2 and Q with s). ``n_parallel`` identical pumps share the flow.
    System curve: H = static_head + Darcy-Weisbach friction (Colebrook) + minor losses of one pipe of the given size.
    """
    from scipy.optimize import brentq

    if min(shutoff_head, design_flow, design_head, diameter, length, speed_ratio) <= 0 or n_parallel < 1:
        raise ValueError("pump and pipe data must be positive and n_parallel >= 1")
    if design_head >= shutoff_head:
        raise ValueError("design_head must be below the shutoff head")
    if static_head < 0 or not 0 < efficiency <= 1:
        raise ValueError("static_head must be >= 0 and efficiency in (0, 1]")
    k = (shutoff_head - design_head) / design_flow ** 2
    area = math.pi * diameter ** 2 / 4.0

    def pump_head(q):
        return speed_ratio ** 2 * shutoff_head - k * (q / n_parallel) ** 2

    def system_head(q):
        if q <= 0:
            return static_head
        return static_head + pipe_pressure_drop(diameter, length, q / area, rho, mu, roughness, minor_k)["head_loss_m"]

    q_max = n_parallel * speed_ratio * math.sqrt(shutoff_head / k)
    if pump_head(1e-9) <= static_head:
        raise ValueError("pump cannot overcome the static head at this speed")
    q = brentq(lambda x: pump_head(x) - system_head(x), 1e-9, q_max, xtol=1e-14)
    h = pump_head(q)
    return {"flow_m3_s": q, "head_m": h, "velocity_m_s": q / area, "flow_per_pump_m3_s": q / n_parallel,
            "friction_and_minor_head_m": h - static_head, "hydraulic_power_w": rho * 9.80665 * q * h,
            "shaft_power_w": rho * 9.80665 * q * h / efficiency, "pump_curve_k": k}


# ---------------------------------------------------------------- open-channel flow (Manning)
def open_channel(flow: float, width: float, slope: float, manning_n: float, side_slope: float = 0.0) -> dict:
    """Uniform (normal) and critical depth of a rectangular (side_slope = 0) or trapezoidal channel, SI units.

    Q = (1/n) A R^(2/3) S^(1/2); critical depth where Q^2 T / (g A^3) = 1. ``side_slope`` is horizontal run per unit rise.
    """
    from scipy.optimize import brentq

    if min(flow, width, slope, manning_n) <= 0 or side_slope < 0:
        raise ValueError("flow, width, slope and manning_n must be positive; side_slope >= 0")
    g = 9.80665

    def geom(y):
        a = (width + side_slope * y) * y
        p = width + 2.0 * y * math.sqrt(1.0 + side_slope ** 2)
        return a, p, width + 2.0 * side_slope * y

    def manning_q(y):
        a, p, _ = geom(y)
        return a * (a / p) ** (2.0 / 3.0) * math.sqrt(slope) / manning_n

    def froude2(y):
        a, _, t = geom(y)
        return flow ** 2 * t / (g * a ** 3)

    hi = 1.0
    while manning_q(hi) < flow:
        hi *= 2.0
        if hi > 1e4:
            raise ValueError("no normal depth found (check units)")
    yn = brentq(lambda y: manning_q(y) - flow, 1e-9, hi, xtol=1e-14)
    yc = brentq(lambda y: froude2(y) - 1.0, 1e-9, hi * 10, xtol=1e-14)
    a, p, t = geom(yn)
    fr = math.sqrt(froude2(yn))
    return {"normal_depth_m": yn, "critical_depth_m": yc, "velocity_m_s": flow / a, "froude": fr,
            "regime": "subcritical (mild slope)" if fr < 1 else "supercritical (steep slope)",
            "hydraulic_radius_m": a / p, "specific_energy_m": yn + (flow / a) ** 2 / (2 * 9.80665)}


def gvf_profile(flow: float, width: float, slope: float, manning_n: float, y_control: float, length: float,
                side_slope: float = 0.0, n: int = 200) -> dict:
    """Gradually varied flow (backwater / drawdown curve): dy/dx = (S0 - Sf) / (1 - Fr^2), integrated from a control depth.

    The integration runs upstream from a downstream control when the flow is subcritical and downstream from an upstream
    control when it is supercritical (x is the distance from the control, positive in the integration direction).
    The march stops early if the depth reaches critical depth (a hydraulic jump or drop would form there).
    """
    from scipy.integrate import solve_ivp

    if min(flow, width, slope, manning_n, y_control, length) <= 0 or side_slope < 0 or n < 10:
        raise ValueError("flow, width, slope, manning_n, y_control and length must be positive; n >= 10")
    g = 9.80665
    ref = open_channel(flow, width, slope, manning_n, side_slope)
    sub = y_control > ref["critical_depth_m"]
    sign = -1.0 if sub else 1.0   # integrate upstream (subcritical) or downstream (supercritical)

    def terms(y):
        a = (width + side_slope * y) * y
        p = width + 2.0 * y * math.sqrt(1.0 + side_slope ** 2)
        t = width + 2.0 * side_slope * y
        sf = (flow * manning_n) ** 2 / (a ** 2 * (a / p) ** (4.0 / 3.0))
        return sf, 1.0 - flow ** 2 * t / (g * a ** 3)

    def rhs(x, y):
        sf, den = terms(y[0])
        return [sign * (slope - sf) / den]

    def hit_critical(x, y):
        return terms(y[0])[1] - 0.02
    hit_critical.terminal = True

    xs = np.linspace(0.0, length, n)
    sol = solve_ivp(rhs, (0.0, length), [y_control], t_eval=xs, rtol=1e-9, atol=1e-12, events=hit_critical)
    y = sol.y[0]
    x = sol.t
    mild = ref["normal_depth_m"] > ref["critical_depth_m"]
    yn, yc = ref["normal_depth_m"], ref["critical_depth_m"]
    if mild:
        kind = ("M1 (backwater)" if y_control > yn else "M2 (drawdown)") if sub else "M3 (rises toward a jump)"
    else:
        kind = "S1 (backwater behind a control)" if sub else ("S2 (falls toward normal depth)" if y_control > yn else "S3 (rises toward normal depth)")
    return {"x_m": x, "depth_m": y, "kind": kind, "subcritical": bool(sub), "normal_depth_m": ref["normal_depth_m"],
            "critical_depth_m": ref["critical_depth_m"], "direction": "upstream" if sub else "downstream",
            "stopped_early": bool(sol.t[-1] < length - 1e-9)}


def hydraulic_jump(flow: float, width: float, y1: float) -> dict:
    """Rectangular hydraulic jump from the supercritical depth y1 (Belanger): sequent depth, Froude numbers, energy loss."""
    if min(flow, width, y1) <= 0:
        raise ValueError("flow, width and y1 must be positive")
    g = 9.80665
    q = flow / width
    fr1 = q / (y1 * math.sqrt(g * y1))
    if fr1 <= 1.0:
        raise ValueError(f"upstream flow is not supercritical (Fr = {fr1:.3f}); no hydraulic jump")
    y2 = 0.5 * y1 * (math.sqrt(1.0 + 8.0 * fr1 ** 2) - 1.0)
    fr2 = q / (y2 * math.sqrt(g * y2))
    e1 = y1 + q ** 2 / (2 * g * y1 ** 2)
    e2 = y2 + q ** 2 / (2 * g * y2 ** 2)
    return {"froude_1": fr1, "froude_2": fr2, "sequent_depth_m": y2, "depth_ratio": y2 / y1, "energy_loss_m": e1 - e2,
            "energy_loss_fraction": (e1 - e2) / e1, "approx_jump_length_m": 6.0 * y2,
            "type": "weak" if fr1 < 1.7 else ("oscillating" if fr1 < 2.5 else ("steady" if fr1 < 4.5 else "strong"))}


# ---------------------------------------------------------------- plotting (CLI outputs)
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_cavity(cav: LidDrivenCavity, path: Path) -> None:
    plt = _plt()
    u, v = cav.velocity()
    x, y = cav.x, cav.y
    speed = np.hypot(u, v)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [max(cav.aspect, 0.6), max(cav.aspect, 0.6), 1]})
    im = ax[0].contourf(x, y, speed, 24, cmap="viridis")
    ax[0].streamplot(x, y, u, v, color="white", density=1.4, linewidth=0.7)
    ax[0].set_title(f"Speed + streamlines (Re = {cav.re:g})")
    fig.colorbar(im, ax=ax[0])
    im = ax[1].contourf(x, y, cav.omega, np.linspace(-12, 12, 25), cmap="RdBu_r", extend="both")
    ax[1].set_title("Vorticity")
    fig.colorbar(im, ax=ax[1])
    y, uc = cav.centreline_u()
    ax[2].plot(uc, y, label="this solver")
    ref = ghia_reference(cav.re)
    if ref is not None and cav.nx == cav.n:
        ax[2].plot(ref[:, 1], ref[:, 0], "o", label="Ghia et al. (1982)")
    ax[2].set_xlabel("u at x = 0.5")
    ax[2].set_ylabel("y")
    ax[2].legend()
    ax[2].grid(alpha=0.3)
    for a in ax[:2]:
        a.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_cavity_3d(cav: LidDrivenCavity, path: Path) -> None:
    """3D mesh surface (height = streamfunction, colour = speed) and 3D vorticity surface of a cavity solution."""
    import matplotlib

    plt = _plt()
    x = np.linspace(0.0, cav.aspect, cav.nx)
    y = np.linspace(0.0, 1.0, cav.n)
    X, Y = np.meshgrid(x, y)
    u, v = cav.velocity()
    sp = np.hypot(u, v)
    step = max(1, cav.n // 40)
    fig = plt.figure(figsize=(13, 5.4))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    rgba = matplotlib.colormaps["viridis"]((sp - sp.min()) / (np.ptp(sp) or 1.0))
    ax.plot_surface(X, Y, cav.psi, facecolors=rgba, rstride=step, cstride=step, edgecolor=(0, 0, 0, 0.5), linewidth=0.3, shade=False)
    ax.set_title(f"3D mesh: {cav.n} x {cav.nx} nodes, Re = {cav.re:g}", weight="bold")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("streamfunction")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    w = np.clip(cav.omega, -10, 10)
    ax2.plot_surface(X, Y, w, cmap="RdBu_r", linewidth=0, antialiased=True)
    ax2.contour(X, Y, w, 12, zdir="z", offset=w.min() - 1.0, cmap="RdBu_r", linewidths=0.8)
    ax2.set_title("3D vorticity surface", weight="bold")
    ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("vorticity")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_convection(cav: NaturalConvectionCavity, path: Path) -> None:
    plt = _plt()
    u, v = cav.velocity()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [max(cav.aspect, 0.6), max(cav.aspect, 0.6), 1]})
    im = ax[0].contourf(cav.x, cav.y, cav.T, 24, cmap="inferno")
    ax[0].streamplot(cav.x, cav.y, u, v, color="white", density=1.3, linewidth=0.7)
    ax[0].set_title(f"Temperature + streamlines (Ra = {cav.ra:g}, Pr = {cav.pr:g})")
    fig.colorbar(im, ax=ax[0])
    im = ax[1].contourf(cav.x, cav.y, cav.psi, 24, cmap="viridis")
    ax[1].set_title("Streamfunction")
    fig.colorbar(im, ax=ax[1])
    ax[2].plot(cav.nusselt_local("hot"), cav.y, label="hot wall")
    ax[2].plot(cav.nusselt_local("cold"), cav.y, "--", label="cold wall")
    ax[2].set_xlabel("local heat flux -dT/dx")
    ax[2].set_ylabel("y")
    ax[2].set_title(f"Nu = {cav.nusselt():.3f}")
    ax[2].legend()
    ax[2].grid(alpha=0.3)
    for a in ax[:2]:
        a.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_shear(res: dict, path: Path) -> None:
    plt = _plt()
    snaps = res["snapshots"]
    keys = sorted(snaps)
    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 4))
    axes = np.atleast_1d(axes)
    for a, k in zip(axes, keys):
        im = a.pcolormesh(res["x"], res["y"], snaps[k], cmap="RdBu_r", vmin=-25, vmax=25, shading="auto")
        a.set_title(f"vorticity, t = {k:g}")
        a.set_aspect("equal")
    fig.colorbar(im, ax=list(axes), shrink=0.8)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_taylor_green(res: dict, path: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(res["t"], res["energy"], lw=2, label="numerical")
    ax.plot(res["t"], res["energy_exact"], "k--", label="exact 0.25 exp(-4 nu t)")
    ax.set_xlabel("time")
    ax.set_ylabel("kinetic energy")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_channel(res_list: list, path: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in res_list:
        ax.plot(r["u"], r["y"], lw=2, label=f"CN, t = {r['t']:.2f}")
        ax.plot(r["u_exact"], r["y"], "k:", lw=1)
    ax.set_xlabel("u")
    ax.set_ylabel("y")
    ax.set_title("Start-up of plane Poiseuille flow (dotted = exact series)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_blasius(bl: dict, path: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.4))
    ax[0].plot(bl["fp"], bl["eta"], lw=2, label="u / U = f'")
    ax[0].plot(bl["fpp"], bl["eta"], lw=2, label="f''")
    ax[0].axhline(bl["delta99"], color="gray", ls="--", lw=1)
    ax[0].text(0.02, bl["delta99"] + 0.15, f"delta99 = {bl['delta99']:.3f}", fontsize=9)
    ax[0].set_xlabel("similarity profile"); ax[0].set_ylabel("eta = y sqrt(U / nu x)"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[0].set_title("Blasius profile", weight="bold")
    re = np.logspace(3, 5.7, 60)
    ax[1].loglog(re, bl["cf_sqrt_re"] / np.sqrt(re), lw=2, label=f"Blasius  cf = {bl['cf_sqrt_re']:.4f} / sqrt(Re_x)")
    ax[1].axvline(5e5, color="C3", ls=":", label="typical transition")
    ax[1].set_xlabel("Re_x"); ax[1].set_ylabel("local skin friction cf"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, which="both")
    ax[1].set_title("Skin friction", weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_network(res: dict, positions: dict | None, path: Path) -> None:
    plt = _plt()
    nodes = list(res["heads_m"])
    if not positions or set(nodes) - set(positions):
        ang = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False)
        positions = {n: (math.cos(a), math.sin(a)) for n, a in zip(nodes, ang)}
    heads = np.array([res["heads_m"][n] for n in nodes])
    qmax = max(abs(p["flow_m3_s"]) for p in res["pipes"]) or 1.0
    fig, ax = plt.subplots(figsize=(8, 5))
    for p in res["pipes"]:
        a, b = (p["from"], p["to"]) if p["flow_m3_s"] >= 0 else (p["to"], p["from"])
        (x0, y0), (x1, y1) = positions[a], positions[b]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "-|>", "lw": 1 + 5 * abs(p["flow_m3_s"]) / qmax,
                                                                  "color": "#546E7A", "shrinkA": 14, "shrinkB": 14})
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, f"{p['name']}\n{abs(p['flow_m3_s']) * 1e3:.2f} L/s", ha="center", va="center",
                fontsize=8, bbox={"fc": "white", "ec": "none", "alpha": 0.85})
    pts = np.array([positions[n] for n in nodes])
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=heads, cmap="viridis_r", s=650, zorder=3, edgecolor="k")
    for n, (x, y) in zip(nodes, pts):
        ax.text(x, y, n, ha="center", va="center", color="white", weight="bold", zorder=4)
    fig.colorbar(sc, ax=ax, label="head (m)")
    ax.set_title("Pipe network: flows and heads", weight="bold")
    ax.margins(0.15)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- CLI
def _out(args) -> Path:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _dump(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2))
    print(json.dumps(data, indent=2))


def cmd_cavity(args) -> dict:
    out = _out(args)
    cav = LidDrivenCavity(args.n, args.re, aspect=args.aspect)
    info = cav.run(tol=args.tol, max_time=args.max_time)
    xc, yc, pmin = cav.vortex_centre()
    summary = {**info, "re": args.re, "n": args.n, "aspect": args.aspect, "psi_min": pmin, "vortex_centre": [xc, yc],
               "divergence_max": cav.divergence_max()}
    if ghia_reference(args.re) is not None and args.aspect == 1.0:
        summary["ghia_centreline_error"] = ghia_error(cav)
    np.savez(out / "cavity.npz", psi=cav.psi, omega=cav.omega, re=args.re)
    plot_cavity(cav, out / "cavity.png")
    if args.plot3d:
        plot_cavity_3d(cav, out / "cavity_3d.png")
    _dump(out / "cavity.json", summary)
    return summary


def cmd_taylor_green(args) -> dict:
    out = _out(args)
    res = taylor_green(args.n, args.nu, args.t_end, args.dt)
    plot_taylor_green(res, out / "taylor_green.png")
    summary = {k: res[k] for k in ("n", "nu", "error_max", "energy_error_max")}
    _dump(out / "taylor_green.json", summary)
    return summary


def cmd_shear(args) -> dict:
    out = _out(args)
    res = shear_layer(args.n, args.nu, args.t_end, args.dt)
    plot_shear(res, out / "shear_layer.png")
    summary = {"n": args.n, "nu": args.nu, "energy_start": float(res["energy"][0]), "energy_end": float(res["energy"][-1]),
               "enstrophy_start": float(res["enstrophy"][0]), "enstrophy_end": float(res["enstrophy"][-1])}
    _dump(out / "shear_layer.json", summary)
    return summary


def cmd_channel(args) -> dict:
    out = _out(args)
    runs = [poiseuille_transient(args.ny, args.nu, args.g, 1.0, t, args.dt) for t in (args.t_end * f for f in (0.1, 0.25, 0.5, 1.0))]
    plot_channel(runs, out / "channel.png")
    summary = {"ny": args.ny, "nu": args.nu, "g": args.g, "error_max_at_end": runs[-1]["error_max"],
               "u_max_steady": runs[-1]["u_max_steady"]}
    _dump(out / "channel.json", summary)
    return summary


def cmd_pipe(args) -> dict:
    res = pipe_pressure_drop(args.diameter, args.length, args.velocity, args.rho, args.mu, args.roughness,
                             args.minor_k, args.efficiency)
    print(json.dumps(res, indent=2))
    return res


def cmd_convection(args) -> dict:
    out = _out(args)
    cav = NaturalConvectionCavity(args.n, args.ra, args.pr, args.aspect)
    info = cav.run(tol=args.tol, max_time=args.max_time)
    umax, vmax = cav.max_velocities()
    summary = {**info, "ra": args.ra, "pr": args.pr, "n": args.n, "aspect": args.aspect, "nusselt_hot": cav.nusselt("hot"),
               "nusselt_cold": cav.nusselt("cold"), "u_max_centreline": umax, "v_max_centreline": vmax}
    ref = DE_VAHL_DAVIS_NU.get(args.ra)
    if ref is not None and args.aspect == 1.0 and abs(args.pr - 0.71) < 1e-9:
        summary["de_vahl_davis_nu"] = ref
        summary["nusselt_relative_error"] = abs(summary["nusselt_hot"] - ref) / ref
    np.savez(out / "convection.npz", psi=cav.psi, omega=cav.omega, T=cav.T, ra=args.ra, pr=args.pr)
    plot_convection(cav, out / "convection.png")
    _dump(out / "convection.json", summary)
    return summary


def cmd_convergence(args) -> dict:
    out = _out(args)
    res = cavity_convergence(args.re, tuple(args.grids))
    _dump(out / "convergence.json", res)
    return res


def cmd_blasius(args) -> dict:
    out = _out(args)
    bl = blasius()
    plot_blasius(bl, out / "blasius.png")
    summary = {"fpp0": bl["fpp0"], "delta99_over_xrex": bl["delta99"], "displacement_over_xrex": bl["delta_star"],
               "momentum_over_xrex": bl["theta"], "cf_sqrt_re": bl["cf_sqrt_re"], "reference_fpp0": 0.33206}
    if args.u_inf and args.nu and args.x:
        summary["plate"] = flat_plate(args.u_inf, args.nu, args.x, args.length)
    _dump(out / "blasius.json", summary)
    return summary


def cmd_network(args) -> dict:
    out = _out(args)
    try:
        spec = json.loads(Path(args.spec).read_text()) if args.spec else EXAMPLE_NETWORK
    except OSError as exc:
        raise ValueError(f"cannot read network spec: {exc}") from exc
    try:
        res = pipe_network(spec["pipes"], spec["fixed_heads"], spec.get("demands", {}), args.rho, args.mu, args.roughness)
    except KeyError as exc:
        raise ValueError(f"network spec is missing {exc}") from exc
    plot_network(res, spec.get("positions"), out / "network.png")
    _dump(out / "network.json", res)
    return res


def cmd_pump(args) -> dict:
    res = pump_operating_point(args.shutoff_head, args.design_flow, args.design_head, args.diameter, args.length, args.static_head,
                               args.minor_k, args.roughness, speed_ratio=args.speed_ratio, n_parallel=args.parallel,
                               efficiency=args.efficiency)
    print(json.dumps(res, indent=2))
    return res


def cmd_channel_flow(args) -> dict:
    res = open_channel(args.flow, args.width, args.slope, args.manning_n, args.side_slope)
    print(json.dumps(res, indent=2))
    return res


def cmd_backwater(args) -> dict:
    res = gvf_profile(args.flow, args.width, args.slope, args.manning_n, args.y_control, args.length, args.side_slope)
    summary = {k: res[k] for k in ("kind", "direction", "normal_depth_m", "critical_depth_m", "stopped_early")}
    summary["depth_at_start_m"] = float(res["depth_m"][0])
    summary["depth_at_end_m"] = float(res["depth_m"][-1])
    summary["distance_m"] = float(res["x_m"][-1])
    print(json.dumps(summary, indent=2))
    return summary


def cmd_jump(args) -> dict:
    res = hydraulic_jump(args.flow, args.width, args.y1)
    print(json.dumps(res, indent=2))
    return res


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fluidflow", description=__doc__.split("\n")[0])
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("-v", "--verbose", action="store_true", help="log solver progress")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("cavity", help="lid-driven cavity")
    c.add_argument("--re", type=float, default=100.0)
    c.add_argument("--n", type=int, default=65, help="grid nodes per side")
    c.add_argument("--aspect", type=float, default=1.0, help="width / height of the cavity (lid on the top side)")
    c.add_argument("--tol", type=float, default=1e-5)
    c.add_argument("--max-time", type=float, default=200.0)
    c.add_argument("--plot3d", action="store_true", help="also write cavity_3d.png (3D mesh surface and vorticity surface)")
    c.add_argument("--out", default="outputs")
    c.set_defaults(fn=cmd_cavity)

    t = sub.add_parser("taylor-green", help="Taylor-Green vortex vs exact solution")
    t.add_argument("--n", type=int, default=32)
    t.add_argument("--nu", type=float, default=0.01)
    t.add_argument("--t-end", type=float, default=2.0)
    t.add_argument("--dt", type=float, default=0.01)
    t.add_argument("--out", default="outputs")
    t.set_defaults(fn=cmd_taylor_green)

    s = sub.add_parser("shear", help="double shear layer roll-up")
    s.add_argument("--n", type=int, default=128)
    s.add_argument("--nu", type=float, default=1e-3)
    s.add_argument("--t-end", type=float, default=6.0)
    s.add_argument("--dt", type=float, default=0.01)
    s.add_argument("--out", default="outputs")
    s.set_defaults(fn=cmd_shear)

    ch = sub.add_parser("channel", help="transient Poiseuille flow vs exact series")
    ch.add_argument("--ny", type=int, default=101)
    ch.add_argument("--nu", type=float, default=0.01)
    ch.add_argument("--g", type=float, default=1.0, help="driving acceleration -dp/dx / rho")
    ch.add_argument("--t-end", type=float, default=1.0)
    ch.add_argument("--dt", type=float, default=1e-3)
    ch.add_argument("--out", default="outputs")
    ch.set_defaults(fn=cmd_channel)

    pp = sub.add_parser("pipe", help="pipe pressure drop (Darcy-Weisbach)")
    pp.add_argument("--diameter", type=float, required=True, help="m")
    pp.add_argument("--length", type=float, required=True, help="m")
    pp.add_argument("--velocity", type=float, required=True, help="m/s")
    pp.add_argument("--rho", type=float, default=998.0, help="kg/m3 (water ~ 998)")
    pp.add_argument("--mu", type=float, default=1.0e-3, help="Pa s (water ~ 1e-3)")
    pp.add_argument("--roughness", type=float, default=4.5e-5, help="absolute roughness, m (commercial steel ~ 4.5e-5)")
    pp.add_argument("--minor-k", type=float, default=0.0, help="summed minor-loss coefficient of fittings")
    pp.add_argument("--efficiency", type=float, default=1.0, help="pump efficiency in (0, 1]")
    pp.set_defaults(fn=cmd_pipe)

    cx = sub.add_parser("convection", help="differentially heated cavity (natural convection)")
    cx.add_argument("--ra", type=float, default=1e4, help="Rayleigh number")
    cx.add_argument("--pr", type=float, default=0.71, help="Prandtl number")
    cx.add_argument("--n", type=int, default=41, help="grid nodes along the height")
    cx.add_argument("--aspect", type=float, default=1.0, help="width / height")
    cx.add_argument("--tol", type=float, default=1e-6)
    cx.add_argument("--max-time", type=float, default=20.0)
    cx.add_argument("--out", default="outputs")
    cx.set_defaults(fn=cmd_convection)

    bz = sub.add_parser("blasius", help="laminar flat-plate boundary layer (Blasius similarity solution)")
    bz.add_argument("--u-inf", type=float, default=None, help="free-stream speed, m/s (optional, for dimensional results)")
    bz.add_argument("--nu", type=float, default=None, help="kinematic viscosity, m2/s")
    bz.add_argument("--x", type=float, default=None, help="distance from the leading edge, m")
    bz.add_argument("--length", type=float, default=None, help="plate length, m (adds the plate drag coefficient)")
    bz.add_argument("--out", default="outputs")
    bz.set_defaults(fn=cmd_blasius)

    nw = sub.add_parser("network", help="looped pipe-network flows and heads (global gradient algorithm)")
    nw.add_argument("--spec", default=None, help="JSON file with pipes / fixed_heads / demands (default: built-in example)")
    nw.add_argument("--rho", type=float, default=998.0)
    nw.add_argument("--mu", type=float, default=1.0e-3)
    nw.add_argument("--roughness", type=float, default=4.5e-5, help="default absolute roughness, m")
    nw.add_argument("--out", default="outputs")
    nw.set_defaults(fn=cmd_network)

    pu = sub.add_parser("pump", help="pump curve vs pipe system curve operating point")
    pu.add_argument("--shutoff-head", type=float, required=True, help="pump head at zero flow, m")
    pu.add_argument("--design-flow", type=float, required=True, help="m3/s")
    pu.add_argument("--design-head", type=float, required=True, help="head at the design flow, m")
    pu.add_argument("--diameter", type=float, required=True, help="pipe diameter, m")
    pu.add_argument("--length", type=float, required=True, help="pipe length, m")
    pu.add_argument("--static-head", type=float, default=0.0, help="elevation lift, m")
    pu.add_argument("--minor-k", type=float, default=0.0)
    pu.add_argument("--roughness", type=float, default=4.5e-5)
    pu.add_argument("--speed-ratio", type=float, default=1.0, help="N / N_rated (affinity laws)")
    pu.add_argument("--parallel", type=int, default=1, help="identical pumps in parallel")
    pu.add_argument("--efficiency", type=float, default=0.7)
    pu.set_defaults(fn=cmd_pump)

    oc = sub.add_parser("open-channel", help="Manning normal and critical depth (rectangular / trapezoidal)")
    oc.add_argument("--flow", type=float, required=True, help="m3/s")
    oc.add_argument("--width", type=float, required=True, help="bottom width, m")
    oc.add_argument("--slope", type=float, required=True, help="bed slope, m/m")
    oc.add_argument("--manning-n", type=float, default=0.013, help="Manning roughness (concrete ~ 0.013)")
    oc.add_argument("--side-slope", type=float, default=0.0, help="horizontal : vertical (0 = rectangular)")
    oc.set_defaults(fn=cmd_channel_flow)

    bw = sub.add_parser("backwater", help="gradually varied flow profile from a control depth (Manning)")
    bw.add_argument("--flow", type=float, required=True)
    bw.add_argument("--width", type=float, required=True)
    bw.add_argument("--slope", type=float, required=True)
    bw.add_argument("--manning-n", type=float, default=0.013)
    bw.add_argument("--y-control", type=float, required=True, help="depth at the control section, m")
    bw.add_argument("--length", type=float, required=True, help="reach length to integrate, m")
    bw.add_argument("--side-slope", type=float, default=0.0)
    bw.set_defaults(fn=cmd_backwater)

    hj = sub.add_parser("jump", help="rectangular hydraulic jump: sequent depth and energy loss")
    hj.add_argument("--flow", type=float, required=True)
    hj.add_argument("--width", type=float, required=True)
    hj.add_argument("--y1", type=float, required=True, help="supercritical depth upstream of the jump, m")
    hj.set_defaults(fn=cmd_jump)

    cv = sub.add_parser("convergence", help="cavity grid-convergence study")
    cv.add_argument("--re", type=float, default=100.0)
    cv.add_argument("--grids", type=int, nargs="+", default=[17, 33, 65])
    cv.add_argument("--out", default="outputs")
    cv.set_defaults(fn=cmd_convergence)
    return p


def main(argv=None):
    """Run the CLI and return the result dictionary (handy for tests and notebooks)."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")
    return args.fn(args)


def cli(argv=None) -> int:
    """Console-script entry point: exit status 0 on success, 2 on invalid input or solver failure."""
    try:
        main(argv)
    except (ValueError, FloatingPointError) as exc:
        print(f"fluidflow: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(cli())
