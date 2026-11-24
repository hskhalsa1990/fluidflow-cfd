### Natural convection, Pr = 0.71 (mean Nusselt number, hot wall)

| Source | Ra = 1e+03 | Ra = 1e+04 | Ra = 1e+05 |
|---|---|---|---|
| de Vahl Davis 1983 | 1.118 | 2.243 | 4.519 |
| this solver 25x25 | 1.125 | 2.331 | 5.234 |
| this solver 41x41 | 1.120 | 2.273 | 4.747 |
| this solver 61x61 | 1.119 | 2.257 | 4.610 |

### Lid-driven cavity, primary-vortex |psi_min|

| Source | 17 x 17 | 33 x 33 | 65 x 65 |
|---|---|---|---|
| Ghia 1982, Re 100 | 0.1034 | 0.1034 | 0.1034 |
| this solver, Re 100 | 0.0943 | 0.1013 | 0.1030 |
| Ghia 1982, Re 400 | 0.1139 | 0.1139 | 0.1139 |
| this solver, Re 400 | 0.0628 | 0.0984 | 0.1106 |

### Blasius flat-plate boundary layer (similarity units)

| Quantity | Literature | This solver | Ratio |
|---|---|---|---|
| f''(0) | 0.3321 | 0.3321 | 1.0000 |
| delta99 / (x/sqrt(Re_x)) | 4.9100 | 4.9101 | 1.0000 |
| delta* / (x/sqrt(Re_x)) | 1.7208 | 1.7208 | 1.0000 |
| theta / (x/sqrt(Re_x)) | 0.6641 | 0.6641 | 1.0000 |

### Haaland vs Colebrook friction factor, |difference| in %

| e/D | Re = 1e+04 | Re = 1e+05 | Re = 1e+06 | Re = 1e+07 |
|---|---|---|---|---|
| 0 | 0.01 | 0.92 | 0.50 | 0.35 |
| 0.0001 | 0.15 | 1.34 | 0.86 | 0.00 |
| 0.001 | 0.64 | 0.94 | 0.01 | 0.18 |
| 0.01 | 0.19 | 0.09 | 0.19 | 0.20 |
