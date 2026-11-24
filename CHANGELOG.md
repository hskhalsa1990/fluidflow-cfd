# Changelog

## 1.6.0
- **3D visualisation:** new `make_3d.py` — 3D surface meshes of the cavity solution at three refinements, 3D field surfaces with projected contours (cavity, natural convection), 3D models (trapezoidal channel with M1 backwater curve, pipe network with head as height, Blasius boundary layer, pump head surface) and comparison tables drawn as 3D bar charts; the same tables are written to `comparison_tables.md`.
- `python main.py cavity --plot3d` writes `cavity_3d.png` (3D mesh surface + vorticity surface).
- 10 new 3D figures; tests 62 -> 65; CI lint covers `make_3d.py`.

## 1.5.0
- **Gradually varied flow:** `gvf_profile()` and `python main.py backwater` — M1/M2/M3/S1/S2/S3 profiles by integrating the gradually-varied-flow equation (rectangular or trapezoidal), direction chosen from the control depth.
- **Hydraulic jump:** `hydraulic_jump()` and `python main.py jump`.
- New figure `chart_backwater.png`; tests 55 -> 62; CI and Makefile run the new commands.

## 1.4.0
- **Pumps:** `pump_operating_point()` and `python main.py pump` — pump curve vs pipe system curve, affinity-law speed change, parallel pumps, shaft power.
- **Open channels:** `open_channel()` and `python main.py open-channel` — Manning normal depth, critical depth, Froude number, regime; rectangular and trapezoidal sections.
- New figures `chart_pump_operating_point.png`, `chart_open_channel.png`; tests 45 -> 55; CI and Makefile run the new commands.

## 1.3.0
- **Boundary layer:** `blasius()`, `flat_plate()` and `python main.py blasius` — laminar flat-plate similarity solution (shooting, SciPy), f''(0) = 0.332057, δ₉₉ / δ* / θ, local skin friction and plate drag for given U, ν, x, L.
- **Pipe networks:** `pipe_network()` and `python main.py network [--spec file.json]` — looped networks with reservoirs, demands, per-pipe roughness and minor losses, global gradient algorithm, reports heads, flows, velocities, head losses and mass/energy residuals. `network_example.json` shows the spec format.
- CLI: a missing or unreadable `--spec` file exits with status 2 and a one-line message.
- New figures: `chart_blasius.png`, `chart_pipe_network.png`; architecture diagram redrawn (`make_diagrams.py blasius network architecture`).
- Tests: 32 -> 45. CI and Makefile run the new commands.

## 1.2.0
- **Heat transport:** `NaturalConvectionCavity` and `python main.py convection` — differentially heated cavity (Boussinesq), temperature advection–diffusion, adaptive time step, local/average Nusselt number, centreline velocity maxima. Matches de Vahl Davis (1983) for Nu, u_max and v_max at Ra = 1e3-1e5 (Pr = 0.71).
- **Non-square geometry:** `LidDrivenCavity(aspect=...)` and `NaturalConvectionCavity(aspect=...)` for rectangular cavities (`--aspect`); the DST Poisson solver now handles different x/y sizes. Existing square behaviour is unchanged (all v1.1 tests still pass).
- `ghia_error` raises a clear error for non-square cavities (no benchmark data).
- New figures: temperature fields, Nu vs Ra, 3D temperature surface, rectangular cavities (`make_diagrams.py convection rectangular`).
- Tests: 22 -> 32.

## 1.1.0
- Cavity benchmark data for Re = 400 and Re = 1000 (Ghia et al. 1982); `ghia_error` now works for all three, and the solver is checked against them (Re = 400 in the test suite; Re = 1000 in `make_diagrams.py multi-re`).
- `LidDrivenCavity.set_reynolds` for continuation / warm starts (Re 100 -> 400 -> 1000 on one grid).
- `cavity_convergence` + `python main.py convergence`: grid-convergence study with the observed order of accuracy (about 2).
- Pipe: minor losses (`--minor-k`), volumetric flow rate, pump shaft power (`--efficiency`), input validation.
- **Fix:** the `fluidflow` console script returned a dict to `sys.exit`, giving exit status 1 on success. New `cli()` entry point returns 0 on success and 2 (with a clean message) on invalid input or solver failure.
- `-v/--verbose` flag for solver progress logging.
- New charts: multi-Re cavity validation, grid convergence, pipe system curves; `make_diagrams.py` can build any single figure by name.
- Tests: 16 -> 22. CI: pip cache, coverage, import-order lint, Docker build job. Dockerfile runs as a non-root user.

## 1.0.0
- Lid-driven cavity (streamfunction-vorticity, 3rd-order upwind, SSP-RK3, DST Poisson solver), validated against Ghia et al. (1982) at Re = 100.
- Pseudo-spectral periodic solver: Taylor-Green vortex (exact check) and double shear layer.
- Transient plane Poiseuille flow (Crank-Nicolson) vs exact series.
- Pipe pressure drop: laminar, Haaland, Colebrook, Darcy-Weisbach.
- CLI, tests, CI, Docker, diagrams (`make_diagrams.py`).
