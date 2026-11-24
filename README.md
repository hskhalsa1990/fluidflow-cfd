# FluidFlow CFD — Validated 2D Incompressible Flow Toolkit in Python (v1.6)

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Tests](https://img.shields.io/badge/tests-pytest-brightgreen) ![Deps](https://img.shields.io/badge/deps-NumPy%20%7C%20SciPy%20%7C%20Matplotlib-orange)

A small computational-fluid-dynamics toolkit written in pure NumPy/SciPy. Every solver is checked against a known reference (benchmark data, an exact solution or a physical balance), and every claim below is reproduced by the test suite. No GPU, Docker or external services needed.

## Features
- **Lid-driven cavity** (`cavity`): streamfunction–vorticity formulation, third-order upwind advection, SSP-RK3 time stepping, exact sine-transform Poisson solver, Thom wall-vorticity boundary condition. Benchmarked against Ghia et al. (1982) at **Re = 100, 400 and 1000**, with warm-start continuation (`set_reynolds`).
- **Heat transport** (`convection`): differentially heated cavity (Boussinesq natural convection) with temperature advection–diffusion, adaptive pseudo-time stepping, local/average Nusselt number; checked against de Vahl Davis (1983) at Ra = 10³–10⁵.
- **Non-square geometry**: both cavity solvers take an aspect ratio (`--aspect`, width/height) — rectangular domains with the same exact DST Poisson solver.
- **Grid-convergence study** (`convergence`): refines the cavity grid and reports the observed order of accuracy (≈ 2).
- **Pseudo-spectral periodic flow** (`taylor-green`, `shear`): integrating-factor RK4, 2/3 de-aliasing. Taylor–Green vortex against the exact solution; double shear layer rolls up into vortices.
- **Plane Poiseuille start-up** (`channel`): Crank–Nicolson finite differences against the exact Fourier-series solution.
- **Pipe flow** (`pipe`): Reynolds number, laminar / Haaland / Colebrook–White friction factor, Darcy–Weisbach pressure drop and head loss, plus minor losses from fittings, flow rate and pump shaft power.
- **Boundary layer** (`blasius`): laminar flat-plate similarity solution by shooting; f''(0) = 0.33206, δ₉₉ = 4.91 x/√Re_x, displacement/momentum thickness, local skin friction and plate drag for real air/water conditions.
- **Pipe networks** (`network`): looped networks with reservoirs, nodal demands, minor losses and per-pipe roughness, solved with the global gradient algorithm (no hand-picked loops). Flow reversals are handled automatically.
- **Pump operating point** (`pump`): pump curve versus pipe system curve (static lift, Colebrook friction, fittings), speed change by the affinity laws, identical pumps in parallel, shaft power.
- **Open-channel flow** (`open-channel`): Manning normal depth, critical depth, Froude number and flow regime for rectangular and trapezoidal channels.
- **Gradually varied flow** (`backwater`): M1/M2/S1/S2/S3/M3 water-surface profiles from a control depth, marching upstream (subcritical) or downstream (supercritical) and stopping at critical depth.
- **Hydraulic jump** (`jump`): sequent depth, Froude numbers, energy loss, jump type for a rectangular channel.
- **3D graphs, charts, meshes and models** (`make_3d.py`, `cavity --plot3d`): 3D surface meshes of the cavity solution (17 → 33 → 65 nodes), 3D field surfaces, 3D channel / pipe-network / boundary-layer / pump models, and comparison tables drawn as 3D bar charts.
- CLI, JSON/NPZ/PNG outputs, tests, GitHub Actions CI, Dockerfile, Makefile, Colab notebook.

## Diagrams

### Architecture
![Architecture](diagram_architecture.png)

```mermaid
flowchart LR
    CLI[python main.py] --> C[cavity]
    CLI --> S[taylor-green / shear]
    CLI --> H[channel]
    CLI --> P[pipe]
    CLI --> G[convergence]
    CLI --> N[convection]
    CLI --> B[blasius]
    CLI --> W[network]
    C --> V1[Ghia et al. 1982: Re 100/400/1000]
    G --> V1
    N --> V5[de Vahl Davis 1983: Nu, umax, vmax]
    S --> V2[Exact TG + energy balance]
    H --> V3[Exact series]
    P --> V4[Haaland vs Colebrook]
    B --> V6["Blasius f-double-prime(0) = 0.33206"]
    W --> V7[Mass + loop energy balance]
    V6 --> O
    V7 --> O
    V1 --> O[JSON / NPZ / PNG]
    V2 --> O
    V3 --> O
    V4 --> O
    V5 --> O
```

### Cavity solver time step
```mermaid
flowchart TD
    A[omega interior] --> B[Poisson solve: psi via DST]
    B --> C[Wall vorticity - Thom formula]
    C --> D[u = psi_y, v = -psi_x]
    D --> E[Advection 3rd-order upwind + diffusion]
    E --> F[SSP-RK3 stage]
    F -->|3 stages| A
```

### 3D views
| Cavity streamfunction (Re = 100) | Cavity vorticity |
|---|---|
| ![3D streamfunction](diagram_3d_cavity_streamfunction.png) | ![3D vorticity](diagram_3d_cavity_vorticity.png) |

| Shear-layer vorticity at t = 6 | Channel velocity developing in time |
|---|---|
| ![3D shear](diagram_3d_shear_vorticity.png) | ![3D channel](diagram_3d_channel_startup.png) |

### Graphs and charts
| Cavity streamlines | Validation vs Ghia et al. |
|---|---|
| ![Streamlines](chart_cavity_streamlines.png) | ![Validation](chart_cavity_validation.png) |

![Cavity at Re = 100, 400, 1000](chart_cavity_multi_re.png)

![Natural convection](chart_convection_temperature.png)

| Nusselt number vs Ra | 3D temperature surface (Ra = 1e4) |
|---|---|
| ![Nu](chart_convection_nusselt.png) | ![3D T](diagram_3d_convection_temperature.png) |

![Rectangular cavities](chart_rectangular_cavities.png)

| Cavity grid convergence | Cavity pseudo-time convergence |
|---|---|
| ![Grid convergence](chart_cavity_grid_convergence.png) | ![Convergence](chart_cavity_convergence.png) |

![Taylor-Green energy decay](chart_taylor_green.png)

![Shear layer snapshots](chart_shear_snapshots.png)

| Shear layer energy / enstrophy | Channel start-up profiles |
|---|---|
| ![Energy enstrophy](chart_shear_energy_enstrophy.png) | ![Channel profiles](chart_channel_profiles.png) |

| Moody diagram | Pipe system: pressure drop and pump power |
|---|---|
| ![Moody diagram](chart_moody.png) | ![Pipe system](chart_pipe_system.png) |

| Blasius boundary layer | Looped pipe network |
|---|---|
| ![Blasius](chart_blasius.png) | ![Network](chart_pipe_network.png) |

| Pump operating points | Open-channel depths |
|---|---|
| ![Pump](chart_pump_operating_point.png) | ![Open channel](chart_open_channel.png) |

![Backwater profiles](chart_backwater.png)

Regenerate every figure with `python make_diagrams.py`.

## 3D views

Regenerate with `python make_3d.py` (or one by name: `mesh fields channel-model network-model blasius pump compare`).

### 3D mesh simulation
![3D mesh refinement](mesh_3d_cavity_refinement.png)

The computational mesh is drawn as a 3D surface mesh: height = streamfunction, colour = speed, every grid line visible. `python main.py cavity --n 65 --plot3d` writes the same view (plus a vorticity surface) for any case.

### 3D charts
![3D cavity fields](chart_3d_cavity_fields.png)

![3D convection surfaces](chart_3d_convection_surfaces.png)

### 3D models
| Pipe network (height = head) | Pump surface H(Q, speed) |
|---|---|
| ![3D pipe network](model_3d_pipe_network.png) | ![3D pump](model_3d_pump_surface.png) |

| Channel with M1 backwater curve | Boundary layer u(x, y) |
|---|---|
| ![3D channel](model_3d_channel_backwater.png) | ![3D boundary layer](model_3d_boundary_layer.png) |

### Comparison tables in 3D form
![3D comparison tables](chart_3d_comparison_tables.png)

The same numbers as plain tables (generated into `comparison_tables.md`):

#### Natural convection, Pr = 0.71 (mean Nusselt number, hot wall)

| Source | Ra = 1e+03 | Ra = 1e+04 | Ra = 1e+05 |
|---|---|---|---|
| de Vahl Davis 1983 | 1.118 | 2.243 | 4.519 |
| this solver 25x25 | 1.125 | 2.331 | 5.234 |
| this solver 41x41 | 1.120 | 2.273 | 4.747 |
| this solver 61x61 | 1.119 | 2.257 | 4.610 |

#### Lid-driven cavity, primary-vortex |psi_min|

| Source | 17 x 17 | 33 x 33 | 65 x 65 |
|---|---|---|---|
| Ghia 1982, Re 100 | 0.1034 | 0.1034 | 0.1034 |
| this solver, Re 100 | 0.0943 | 0.1013 | 0.1030 |
| Ghia 1982, Re 400 | 0.1139 | 0.1139 | 0.1139 |
| this solver, Re 400 | 0.0628 | 0.0984 | 0.1106 |

#### Blasius flat-plate boundary layer (similarity units)

| Quantity | Literature | This solver | Ratio |
|---|---|---|---|
| f''(0) | 0.3321 | 0.3321 | 1.0000 |
| delta99 / (x/sqrt(Re_x)) | 4.9100 | 4.9101 | 1.0000 |
| delta* / (x/sqrt(Re_x)) | 1.7208 | 1.7208 | 1.0000 |
| theta / (x/sqrt(Re_x)) | 0.6641 | 0.6641 | 1.0000 |

#### Haaland vs Colebrook friction factor, |difference| in %

| e/D | Re = 1e+04 | Re = 1e+05 | Re = 1e+06 | Re = 1e+07 |
|---|---|---|---|---|
| 0 | 0.01 | 0.92 | 0.50 | 0.35 |
| 0.0001 | 0.15 | 1.34 | 0.86 | 0.00 |
| 0.001 | 0.64 | 0.94 | 0.01 | 0.18 |
| 0.01 | 0.19 | 0.09 | 0.19 | 0.20 |

The 17 × 17 and 25 × 25 rows show how coarse meshes behave; Re = 400 needs finer grids or continuation to reach the Ghia value (see the validation table below).

## Validation (reproduced by `test_main.py`)
| Case | Result |
|---|---|
| Cavity Re = 100, 65 × 65 | primary vortex centre (0.615, 0.737), ψ_min = −0.1030; Ghia et al. (1982): (0.6172, 0.7344), −0.1034 |
| Cavity centreline u(y) vs Ghia et al., Re = 100 | RMS deviation 0.0019, max 0.0046 (65 × 65) |
| Cavity Re = 400, 129 × 129 (continuation) | vortex centre (0.555, 0.607), ψ_min = −0.1132; Ghia: (0.5547, 0.6055), −0.1139; centreline RMS 0.0021 |
| Cavity Re = 1000, 129 × 129 (continuation) | vortex centre (0.531, 0.567), ψ_min = −0.1166; Ghia: (0.5313, 0.5625), −0.1179; centreline RMS 0.0072 |
| Natural convection, Pr = 0.71, square | Nu = 1.120 / 2.257 / 4.568 at Ra = 10³ / 10⁴ / 10⁵ (41 / 61 / 81 nodes); de Vahl Davis: 1.118 / 2.243 / 4.519. Ra = 10⁴ refined: 2.273 (41), 2.257 (61), 2.251 (81) → 2.243 |
| Natural convection velocities | Ra = 10⁴, 41 × 41: u_max 16.14, v_max 19.49 vs 16.18, 19.62; Ra = 10⁵, 65 × 65: 34.70, 68.13 vs 34.73, 68.59 |
| Natural convection physics checks | Nu(hot wall) = Nu(cold wall) to 1e-13; centro-symmetry T(x,y) + T(W−x,H−y) = 1 also for W/H = 2; Ra → 0 gives Nu = 1 (pure conduction) |
| Cavity grid convergence, Re = 100 | ψ_min = −0.0943, −0.1013, −0.1030, −0.1034 on 17, 33, 65, 129 nodes; observed order ≈ 2.1 |
| Cavity mass conservation | max divergence ≈ 4e-15 |
| Taylor–Green vortex | max error against exact solution ≈ 4e-15 |
| Shear layer | dE/dt matches −2νZ (energy–enstrophy balance) |
| Channel start-up, 101 points | max error vs exact series ≈ 2e-4 (u_max = 12.5 at steady state) |
| Blasius boundary layer | f''(0) = 0.332057; δ₉₉ = 4.910, δ* = 1.7208, θ = 0.6641 (units x/√Re_x); plate C_D = 1.328/√Re_L |
| Pipe network (example, 5 nodes, 6 pipes) | continuity residual ≈ 1e-16 m³/s, energy residual < 1e-8 m, loop head losses close to 1e-8 m; single pipe equals Darcy–Weisbach; laminar parallel pipes split as D⁴/L (ratio 16.000) |
| Pump operating point | head and flow satisfy both the pump and the system curve to 1e-8 m; zero-friction limit gives the static lift; two pumps in parallel give less than 2x the flow |
| Open channel (rectangular) | Manning flow at the normal depth reproduces Q to 1e-8; critical depth equals (q²/g)^(1/3); wide-channel limit y = (nq/√S)^(3/5) |
| Gradually varied flow | starting at normal depth the profile stays uniform (< 1e-6 m); M1 decays to normal depth, M2 rises to it, the steep-slope profile reaches normal depth; the local slope equals (S0 − Sf)/(1 − Fr²) |
| Hydraulic jump | momentum is conserved across the jump to 1e-9; energy is lost; downstream flow is subcritical |
| Smooth pipe, Re = 1e5 | Colebrook f = 0.01799; Haaland within 2 % |

## Physics
- Streamfunction–vorticity: `u = ψ_y`, `v = −ψ_x`, `ω = v_x − u_y = −∇²ψ`, `ω_t + u ω_x + v ω_y = ν ∇²ω`.
- Thom wall vorticity: `ω_wall = −2 ψ_(1) / h² − 2 U_wall / h` (moving lid), second-order Taylor expansion of ψ at the wall.
- Periodic flow: same vorticity equation on [0, 2π)², solved in Fourier space.
- Taylor–Green: `ω = −2 cos x cos y e^(−2νt)`.
- Poiseuille start-up: `u_t = G + ν u_yy`, exact series `u = G y(H−y)/(2ν) − Σ 4GH²/(ν π³ n³) sin(nπy/H) e^(−ν n² π² t/H²)`, n odd.
- Natural convection (Boussinesq, lengths in H, velocity in α/H): `ω_t + u ω_x + v ω_y = Pr ∇²ω + Ra Pr T_x`, `T_t + u T_x + v T_y = ∇²T`; Nu = W · mean(−T_x) on the hot wall.
- Blasius: `f''' + ½ f f'' = 0`, η = y √(U/νx), `u/U = f'(η)`, `c_f = 2 f''(0)/√Re_x = 0.6641/√Re_x`.
- Pipe network: head loss `h = (f L/D + K) · 8 Q|Q| / (g π² D⁴)`, continuity at every free node, Newton on all flows and heads (Todini–Pilati).
- Pump: `H = s² H₀ − k (Q/m)²` against `H = H_static + Δh_friction(Q)`; Manning `Q = A R^(2/3) √S / n`, critical depth from `Q² T / (g A³) = 1`.
- Gradually varied flow: `dy/dx = (S₀ − S_f)/(1 − Fr²)`, `S_f` from Manning; hydraulic jump: `y₂ = ½ y₁ (√(1 + 8 Fr₁²) − 1)` (momentum balance).
- Darcy–Weisbach: `Δp = f (L/D) ½ ρ V²`, laminar `f = 64/Re`, turbulent Colebrook–White (Haaland as the explicit estimate).

## Run locally
```bash
python -m pip install -r requirements.txt
python main.py cavity --re 100 --n 65          # ~6 s, writes outputs/cavity.{json,npz,png}
python main.py taylor-green --n 32 --nu 0.01 --t-end 2
python main.py shear --n 128 --nu 1e-3 --t-end 6
python main.py channel --nu 0.01 --g 1 --t-end 1
python main.py pipe --diameter 0.05 --length 100 --velocity 2 --minor-k 2.5 --efficiency 0.7
python main.py blasius --u-inf 5 --nu 1.5e-5 --x 0.5 --length 1   # dimensional boundary layer
python main.py pump --shutoff-head 40 --design-flow 0.01 --design-head 30 --diameter 0.1 --length 200 --static-head 10
python main.py open-channel --flow 5 --width 3 --slope 0.001 --manning-n 0.013
python main.py backwater --flow 5 --width 3 --slope 0.001 --y-control 2 --length 3000
python main.py jump --flow 5 --width 3 --y1 0.4
python main.py network                                  # built-in example; or --spec network_example.json
python main.py convergence --re 100 --grids 17 33 65
python main.py convection --ra 1e5 --n 65      # ~15 s, writes outputs/convection.{json,npz,png}
python main.py convection --ra 1e4 --aspect 2  # wide heated cavity
python main.py cavity --re 100 --n 65 --aspect 0.5
python main.py cavity --re 400 --n 97 -v      # ~40 s, progress logged
```
The CLI returns exit status 0 on success and 2 (with a one-line message) on invalid input. Re = 1000 on 129 × 129 takes a few minutes; for Re = 400/1000 use continuation from Python:
```python
import main as fl
cav = fl.LidDrivenCavity(129, 100)
for re in (100, 400, 1000):
    cav.set_reynolds(re); cav.run(tol=1e-5, max_time=400)
    print(re, cav.vortex_centre(), fl.ghia_error(cav))
```
Outputs go to `outputs/` (or `--out`).

## Tests
```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q          # 65 tests, ~40 s; or: python -m unittest test_main
```

## Repository layout (flat)
Every file sits in the repository root — there are no sub-folders, so the whole project can be uploaded from a phone browser.

| File(s) | Purpose |
|---|---|
| `main.py` | all solvers + command-line interface |
| `test_main.py` | 65 tests (pytest / unittest) |
| `make_diagrams.py`, `make_3d.py` | regenerate every figure |
| `network_example.json` | sample pipe-network spec |
| `comparison_tables.md` | generated comparison tables |
| `fluidflow_colab.ipynb` | Google Colab notebook |
| `diagram_*.png`, `chart_*.png`, `model_3d_*.png`, `mesh_3d_*.png` | figures used in this README |
| `pyproject.toml`, `requirements*.txt`, `Makefile`, `Dockerfile` | packaging and tooling |
| `ci.yml` | GitHub Actions workflow — to activate CI, move it to `.github/workflows/ci.yml` (GitHub web: *Add file → Create new file*, type that path, paste the content) |

## Limitations
- Open-channel routines assume prismatic channels with constant slope and roughness; the backwater march stops at critical depth and does not locate the jump position (use `jump` for the sequent depth).
- Blasius is the zero-pressure-gradient laminar solution only (valid up to Re_x ≈ 5·10⁵); no transition or turbulent boundary layer. The pipe-network solver is steady and single-phase: no pumps, valves with curves or tanks yet, and the friction factor in the transitional range 2300 < Re < 4000 is uncertain.
- Two-dimensional, incompressible, Newtonian flow only; no turbulence modelling. Geometry is limited to rectangles (aspect ratio), not L-shapes, steps or curved boundaries — those need a masked/immersed-boundary or unstructured solver.
- The cavity solver is benchmarked at Re = 100, 400 and 1000 only (u along the vertical centreline plus the primary-vortex position). Beyond Re ≈ 1000 the steady pseudo-time march needs much finer grids and was not tested. The centreline RMS against Ghia et al. levels off near 2·10⁻³ on fine grids because the reference data itself is a 129 × 129 solution; use ψ_min for convergence studies.
- Natural convection is benchmarked for the square cavity at Pr = 0.71, Ra = 10³–10⁵. Ra = 10⁶ on 97 × 97 gives Nu = 9.06 vs 8.80 (3 % high, boundary layers under-resolved); use finer grids. Rectangular cases (any aspect ratio) are checked only for mass conservation, symmetry and heat balance, not against published data.
- Thom's wall-vorticity condition is only first-order accurate overall near the corners.
- Educational / research-prototype quality, not certified engineering software. Validate against your own data before real design use.

## License
MIT — see `LICENSE`. © 2026 Harpreet Singh.
