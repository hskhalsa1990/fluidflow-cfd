import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import main as fl  # noqa: E402


class TestCavity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cav = fl.LidDrivenCavity(49, 100)
        cls.info = cls.cav.run(tol=1e-5)

    def test_converges(self):
        self.assertTrue(self.info["converged"])

    def test_primary_vortex_matches_ghia(self):
        x, y, pmin = self.cav.vortex_centre()
        self.assertAlmostEqual(pmin, -0.1034, delta=0.004)
        self.assertAlmostEqual(x, 0.6172, delta=0.03)
        self.assertAlmostEqual(y, 0.7344, delta=0.03)

    def test_centreline_profile_close_to_ghia(self):
        self.assertLess(fl.ghia_error(self.cav)["rms"], 0.01)

    def test_mass_conserved(self):
        self.assertLess(self.cav.divergence_max(), 1e-10)

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            fl.LidDrivenCavity(4, 100)
        with self.assertRaises(ValueError):
            fl.LidDrivenCavity(33, -1)

    def test_benchmark_lookup(self):
        self.assertIsNone(fl.ghia_reference(50))
        for re in (100, 400, 1000):
            self.assertEqual(fl.ghia_reference(re).shape, (17, 2))
        with self.assertRaises(ValueError):
            fl.ghia_error(fl.LidDrivenCavity(17, 50))

    def test_set_reynolds_keeps_field(self):
        cav = fl.LidDrivenCavity(17, 100)
        cav.run(tol=1e-4)
        before = cav.omega.copy()
        cav.set_reynolds(200)
        self.assertEqual(cav.re, 200.0)
        self.assertAlmostEqual(cav.nu, 1.0 / 200)
        np.testing.assert_array_equal(cav.omega, before)
        with self.assertRaises(ValueError):
            cav.set_reynolds(0)


class TestRectangularCavity(unittest.TestCase):
    def test_shapes_and_mass(self):
        for aspect in (2.0, 0.5):
            cav = fl.LidDrivenCavity(17, 100, aspect=aspect)
            self.assertEqual(cav.psi.shape, (17, cav.nx))
            self.assertAlmostEqual(cav.x[-1], aspect)
            info = cav.run(tol=1e-4)
            self.assertTrue(info["converged"])
            self.assertLess(cav.divergence_max(), 1e-10)

    def test_square_default_unchanged(self):
        self.assertEqual(fl.LidDrivenCavity(33, 100).nx, 33)
        self.assertEqual(fl.LidDrivenCavity(33, 100, aspect=1.0).psi.shape, (33, 33))

    def test_wide_cavity_vortex_sits_downstream(self):
        cav = fl.LidDrivenCavity(33, 100, aspect=2.0)
        cav.run(tol=1e-5)
        x, y, pmin = cav.vortex_centre()
        self.assertGreater(x, 1.0)           # lid moves in +x, the primary vortex is displaced to the right half
        self.assertLess(pmin, 0.0)

    def test_bad_aspect_and_benchmark(self):
        with self.assertRaises(ValueError):
            fl.LidDrivenCavity(33, 100, aspect=0)
        with self.assertRaises(ValueError):
            fl.ghia_error(fl.LidDrivenCavity(17, 100, aspect=2.0))


class TestNaturalConvection(unittest.TestCase):
    def test_conduction_limit(self):
        cav = fl.NaturalConvectionCavity(25, 1.0)
        self.assertTrue(cav.run(tol=1e-6)["converged"])
        self.assertAlmostEqual(cav.nusselt(), 1.0, delta=1e-3)

    def test_de_vahl_davis_ra1e3(self):
        cav = fl.NaturalConvectionCavity(33, 1e3)
        self.assertTrue(cav.run()["converged"])
        self.assertAlmostEqual(cav.nusselt(), fl.DE_VAHL_DAVIS_NU[1e3], delta=0.03 * fl.DE_VAHL_DAVIS_NU[1e3])

    def test_de_vahl_davis_ra1e4(self):
        cav = fl.NaturalConvectionCavity(41, 1e4)
        self.assertTrue(cav.run()["converged"])
        self.assertAlmostEqual(cav.nusselt(), fl.DE_VAHL_DAVIS_NU[1e4], delta=0.03 * fl.DE_VAHL_DAVIS_NU[1e4])
        umax, vmax = cav.max_velocities()
        self.assertAlmostEqual(umax, fl.DE_VAHL_DAVIS_VMAX[1e4][0], delta=0.03 * fl.DE_VAHL_DAVIS_VMAX[1e4][0])
        self.assertAlmostEqual(vmax, fl.DE_VAHL_DAVIS_VMAX[1e4][1], delta=0.03 * fl.DE_VAHL_DAVIS_VMAX[1e4][1])
        self.assertLess(cav.heat_balance_error(), 1e-6)

    def test_point_symmetry_in_rectangle(self):
        """Boussinesq + symmetric BCs: T(x, y) + T(W - x, H - y) = 1 (centro-symmetry), also for a wide cavity."""
        cav = fl.NaturalConvectionCavity(25, 1e3, aspect=2.0)
        self.assertTrue(cav.run()["converged"])
        np.testing.assert_allclose(cav.T + cav.T[::-1, ::-1], 1.0, atol=2e-3)
        self.assertEqual(cav.psi.shape, (25, cav.nx))

    def test_bad_inputs(self):
        for bad in ((4, 1e4, 0.71), (33, -1, 0.71), (33, 1e4, 0.0)):
            with self.assertRaises(ValueError):
                fl.NaturalConvectionCavity(*bad)
        with self.assertRaises(ValueError):
            fl.NaturalConvectionCavity(17, 1e3).nusselt("left")


class TestCavityHigherRe(unittest.TestCase):
    def test_re400_matches_ghia(self):
        cav = fl.LidDrivenCavity(65, 400)
        info = cav.run(tol=1e-5)
        self.assertTrue(info["converged"])
        x, y, pmin = cav.vortex_centre()
        gx, gy, gp = fl.GHIA_VORTEX[400]
        self.assertAlmostEqual(pmin, gp, delta=0.006)
        self.assertAlmostEqual(x, gx, delta=0.04)
        self.assertAlmostEqual(y, gy, delta=0.04)
        self.assertLess(fl.ghia_error(cav)["rms"], 0.015)


class TestConvergence(unittest.TestCase):
    def test_second_order(self):
        res = fl.cavity_convergence(100.0, (17, 33, 65))
        self.assertTrue(all(r["converged"] for r in res["rows"]))
        self.assertGreater(res["observed_order"], 1.6)
        self.assertLess(res["observed_order"], 2.6)
        errs = [r["ghia_rms"] for r in res["rows"]]
        self.assertTrue(errs[0] > errs[1] > errs[2])


class TestSpectral(unittest.TestCase):
    def test_taylor_green_exact(self):
        res = fl.taylor_green(32, 0.01, 1.0, 0.01)
        self.assertLess(res["error_max"], 1e-10)
        self.assertLess(res["energy_error_max"], 1e-10)

    def test_energy_balance_shear_layer(self):
        res = fl.shear_layer(n=64, nu=1e-3, t_end=1.0, dt=0.01, snapshots=())
        dedt = np.gradient(res["energy"], res["t"])
        self.assertLess(np.max(np.abs(dedt[2:-2] + 2 * 1e-3 * res["enstrophy"][2:-2])), 2e-3)
        self.assertLessEqual(res["energy"][-1], res["energy"][0])

    def test_bad_grid(self):
        with self.assertRaises(ValueError):
            fl.Spectral2D(7, 0.01)


class TestChannel(unittest.TestCase):
    def test_matches_exact_series(self):
        res = fl.poiseuille_transient(81, 0.01, 1.0, 1.0, 0.5, 1e-3)
        self.assertLess(res["error_max"], 1e-3)

    def test_steady_peak(self):
        res = fl.poiseuille_transient(81, 0.01, 1.0, 1.0, 100.0, 0.05)
        self.assertAlmostEqual(res["u"].max(), res["u_max_steady"], delta=0.01 * res["u_max_steady"])


class TestPipe(unittest.TestCase):
    def test_laminar_friction(self):
        self.assertAlmostEqual(fl.friction_factor(1000), 0.064)

    def test_haaland_close_to_colebrook(self):
        for re, rr in ((1e5, 0.0), (1e5, 1e-3), (1e6, 1e-4)):
            self.assertAlmostEqual(fl.friction_factor_haaland(re, rr) / fl.friction_factor_colebrook(re, rr), 1.0, delta=0.02)

    def test_smooth_pipe_reference(self):
        self.assertAlmostEqual(fl.friction_factor_colebrook(1e5, 0.0), 0.0180, delta=0.0003)

    def test_pressure_drop_scales_with_length(self):
        a = fl.pipe_pressure_drop(0.05, 100, 2.0)
        b = fl.pipe_pressure_drop(0.05, 200, 2.0)
        self.assertAlmostEqual(b["pressure_drop_pa"] / a["pressure_drop_pa"], 2.0)
        self.assertEqual(a["regime"], "turbulent")

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            fl.reynolds(998, 1, 0, 1e-3)
        with self.assertRaises(ValueError):
            fl.friction_factor(-5)
        with self.assertRaises(ValueError):
            fl.pipe_pressure_drop(-0.05, 10, 1)
        with self.assertRaises(ValueError):
            fl.pipe_pressure_drop(0.05, 10, 1, efficiency=1.5)

    def test_minor_losses_and_power(self):
        base = fl.pipe_pressure_drop(0.05, 100, 2.0)
        fit = fl.pipe_pressure_drop(0.05, 100, 2.0, minor_k=2.5, efficiency=0.7)
        self.assertAlmostEqual(fit["pressure_drop_minor_pa"], 2.5 * 0.5 * 998.0 * 4.0)
        self.assertAlmostEqual(fit["pressure_drop_pa"], base["pressure_drop_pa"] + fit["pressure_drop_minor_pa"])
        self.assertAlmostEqual(fit["flow_rate_m3_s"], 2.0 * math.pi * 0.05 ** 2 / 4)
        self.assertAlmostEqual(fit["pump_power_w"], fit["pressure_drop_pa"] * fit["flow_rate_m3_s"] / 0.7)


class TestCLI(unittest.TestCase):
    def test_pipe_cli(self):
        res = fl.main(["pipe", "--diameter", "0.05", "--length", "10", "--velocity", "1"])
        self.assertGreater(res["pressure_drop_pa"], 0)
        self.assertTrue(math.isfinite(res["friction_factor"]))

    def test_convection_cli(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            res = fl.main(["convection", "--ra", "1e3", "--n", "25", "--out", d])
            self.assertTrue(res["converged"])
            self.assertAlmostEqual(res["nusselt_hot"], res["nusselt_cold"], places=6)
            self.assertTrue((Path(d) / "convection.png").exists())

    def test_exit_codes(self):
        self.assertEqual(fl.cli(["pipe", "--diameter", "0.05", "--length", "10", "--velocity", "1"]), 0)
        self.assertEqual(fl.cli(["pipe", "--diameter", "-1", "--length", "10", "--velocity", "1"]), 2)


class TestBlasius(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bl = fl.blasius()

    def test_wall_shear_matches_reference(self):
        self.assertAlmostEqual(self.bl["fpp0"], 0.332057, places=5)

    def test_integral_thicknesses(self):
        self.assertAlmostEqual(self.bl["delta99"], 4.91, delta=0.01)
        self.assertAlmostEqual(self.bl["delta_star"], 1.7208, delta=0.001)
        self.assertAlmostEqual(self.bl["theta"], 0.6641, delta=0.001)

    def test_profile_reaches_free_stream(self):
        self.assertAlmostEqual(self.bl["fp"][-1], 1.0, places=6)
        self.assertTrue(np.all(np.diff(self.bl["fp"]) >= -1e-12))

    def test_plate_drag_is_twice_local_cf_at_end(self):
        res = fl.flat_plate(5.0, 1.5e-5, 1.0, length=1.0)
        self.assertAlmostEqual(res["drag_coefficient_plate"], 2 * res["cf_local"], places=10)
        self.assertAlmostEqual(res["drag_coefficient_plate"], 1.328 / math.sqrt(res["re_length"]), delta=1e-4)

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            fl.flat_plate(-1, 1e-5, 1)
        with self.assertRaises(ValueError):
            fl.blasius(eta_max=3)


class TestPipeNetwork(unittest.TestCase):
    def setUp(self):
        e = fl.EXAMPLE_NETWORK
        self.res = fl.pipe_network(e["pipes"], e["fixed_heads"], e["demands"])

    def test_converges_and_conserves_mass(self):
        self.assertTrue(self.res["converged"])
        self.assertLess(self.res["mass_residual_m3_s"], 1e-12)
        self.assertAlmostEqual(self.res["net_outflow_fixed_nodes_m3_s"], sum(fl.EXAMPLE_NETWORK["demands"].values()), places=10)

    def test_energy_balance_every_pipe(self):
        self.assertLess(self.res["energy_residual_m"], 1e-8)

    def test_loop_head_losses_close(self):
        loss = {p["name"]: p["head_loss_m"] for p in self.res["pipes"]}
        # loop A-B-C: P2 (A->B) + P6 (B->C) - P3 (A->C) must be zero
        self.assertAlmostEqual(loss["P2"] + loss["P6"] - loss["P3"], 0.0, places=8)

    def test_single_pipe_matches_darcy_weisbach(self):
        d, length, v = 0.05, 100.0, 2.0
        q = v * math.pi * d ** 2 / 4
        res = fl.pipe_network([{"from": "R", "to": "X", "diameter": d, "length": length}], {"R": 30.0}, {"X": q})
        ref = fl.pipe_pressure_drop(d, length, v)
        self.assertAlmostEqual(res["pipes"][0]["head_loss_m"], ref["head_loss_m"], places=9)
        self.assertAlmostEqual(res["heads_m"]["X"], 30.0 - ref["head_loss_m"], places=9)

    def test_laminar_parallel_split_follows_d4_over_l(self):
        pipes = [{"from": "R", "to": "X", "diameter": 0.01, "length": 10}, {"from": "R", "to": "X", "diameter": 0.02, "length": 10}]
        res = fl.pipe_network(pipes, {"R": 1.0}, {"X": 2e-6})
        q1, q2 = (p["flow_m3_s"] for p in res["pipes"])
        self.assertAlmostEqual(q2 / q1, 16.0, places=6)

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            fl.pipe_network([], {"R": 1.0}, {})
        with self.assertRaises(ValueError):
            fl.pipe_network([{"from": "R", "to": "X", "diameter": 0.1, "length": 10}], {"Z": 1.0}, {})
        with self.assertRaises(ValueError):
            fl.pipe_network([{"from": "R", "to": "X", "diameter": -0.1, "length": 10}], {"R": 1.0}, {"X": 0.001})


class TestNewCLI(unittest.TestCase):
    def test_blasius_cli(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            res = fl.main(["blasius", "--u-inf", "5", "--nu", "1.5e-5", "--x", "0.5", "--out", d])
            self.assertAlmostEqual(res["fpp0"], 0.33206, places=4)
            self.assertIn("plate", res)
            self.assertTrue((Path(d) / "blasius.png").exists())

    def test_network_cli_and_spec_errors(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            res = fl.main(["network", "--out", d])
            self.assertTrue(res["converged"])
            self.assertTrue((Path(d) / "network.png").exists())
            spec = Path(d) / "spec.json"
            spec.write_text(json.dumps({"pipes": [{"from": "R", "to": "X", "diameter": 0.1, "length": 50}], "fixed_heads": {"R": 20},
                                        "demands": {"X": 0.005}}))
            self.assertTrue(fl.main(["network", "--spec", str(spec), "--out", d])["converged"])
            self.assertEqual(fl.cli(["network", "--spec", str(Path(d) / "missing.json"), "--out", d]), 2)


class TestPump(unittest.TestCase):
    base = dict(shutoff_head=40.0, design_flow=0.01, design_head=30.0, diameter=0.1, length=200.0, static_head=10.0)

    def test_intersection_satisfies_both_curves(self):
        r = fl.pump_operating_point(**self.base)
        area = math.pi * 0.1 ** 2 / 4
        sys_h = 10.0 + fl.pipe_pressure_drop(0.1, 200.0, r["flow_m3_s"] / area)["head_loss_m"]
        self.assertAlmostEqual(r["head_m"], sys_h, places=8)
        self.assertAlmostEqual(r["head_m"], 40.0 - r["pump_curve_k"] * r["flow_m3_s"] ** 2, places=8)

    def test_zero_friction_limit_is_static_lift(self):
        r = fl.pump_operating_point(**{**self.base, "length": 1e-3, "diameter": 1.0})
        self.assertAlmostEqual(r["head_m"], 10.0, delta=0.01)

    def test_speed_and_parallel_trends(self):
        slow = fl.pump_operating_point(**self.base, speed_ratio=0.9)
        base = fl.pump_operating_point(**self.base)
        par = fl.pump_operating_point(**self.base, n_parallel=2)
        self.assertLess(slow["flow_m3_s"], base["flow_m3_s"])
        self.assertGreater(par["flow_m3_s"], base["flow_m3_s"])
        self.assertLess(par["flow_m3_s"], 2 * base["flow_m3_s"])

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            fl.pump_operating_point(**{**self.base, "design_head": 45.0})
        with self.assertRaises(ValueError):
            fl.pump_operating_point(**{**self.base, "static_head": 50.0})


class TestOpenChannel(unittest.TestCase):
    def test_normal_depth_reproduces_flow(self):
        r = fl.open_channel(5.0, 3.0, 0.001, 0.013)
        y = r["normal_depth_m"]
        a, p = 3.0 * y, 3.0 + 2 * y
        self.assertAlmostEqual(a * (a / p) ** (2 / 3) * math.sqrt(0.001) / 0.013, 5.0, places=8)

    def test_critical_depth_rectangular_formula(self):
        r = fl.open_channel(5.0, 3.0, 0.001, 0.013)
        self.assertAlmostEqual(r["critical_depth_m"], ((5.0 / 3.0) ** 2 / 9.80665) ** (1 / 3), places=9)

    def test_froude_one_at_critical_depth(self):
        r = fl.open_channel(5.0, 3.0, 0.001, 0.013)
        self.assertLess(r["froude"], 1.0)
        steep = fl.open_channel(5.0, 3.0, 0.05, 0.013)
        self.assertGreater(steep["froude"], 1.0)
        self.assertLess(steep["normal_depth_m"], steep["critical_depth_m"])

    def test_wide_channel_limit(self):
        r = fl.open_channel(1000.0, 1000.0, 0.0005, 0.03)
        self.assertAlmostEqual(r["normal_depth_m"], (0.03 * 1.0 / math.sqrt(0.0005)) ** 0.6, delta=0.02)

    def test_trapezoid_deeper_flow_needs_less_depth(self):
        self.assertLess(fl.open_channel(5.0, 3.0, 0.001, 0.013, 2.0)["normal_depth_m"], fl.open_channel(5.0, 3.0, 0.001, 0.013)["normal_depth_m"])

    def test_cli_and_errors(self):
        self.assertAlmostEqual(fl.main(["open-channel", "--flow", "5", "--width", "3", "--slope", "0.001"])["critical_depth_m"], 0.6567, places=3)
        self.assertEqual(fl.cli(["open-channel", "--flow", "-5", "--width", "3", "--slope", "0.001"]), 2)
        self.assertTrue(fl.main(["pump", "--shutoff-head", "40", "--design-flow", "0.01", "--design-head", "30", "--diameter", "0.1",
                                 "--length", "200"])["flow_m3_s"] > 0)


class TestGradedFlowAndJump(unittest.TestCase):
    args = (5.0, 3.0, 0.001, 0.013)

    def test_normal_depth_start_is_uniform(self):
        yn = fl.open_channel(*self.args)["normal_depth_m"]
        res = fl.gvf_profile(*self.args, y_control=yn, length=500.0)
        self.assertLess(float(np.max(np.abs(res["depth_m"] - yn))), 1e-6)

    def test_m1_backwater_decays_to_normal_depth(self):
        res = fl.gvf_profile(*self.args, y_control=2.0, length=6000.0)
        self.assertTrue(res["kind"].startswith("M1"))
        self.assertTrue(np.all(np.diff(res["depth_m"]) <= 1e-9))
        self.assertAlmostEqual(res["depth_m"][-1], res["normal_depth_m"], delta=2e-3)

    def test_m2_drawdown_rises_to_normal_depth(self):
        res = fl.gvf_profile(*self.args, y_control=0.8, length=4000.0)
        self.assertTrue(res["kind"].startswith("M2"))
        self.assertTrue(np.all(np.diff(res["depth_m"]) >= -1e-9))
        self.assertLess(res["depth_m"][-1], res["normal_depth_m"])

    def test_energy_slope_matches_profile_slope(self):
        res = fl.gvf_profile(*self.args, y_control=2.0, length=50.0, n=400)
        y = res["depth_m"][10]
        dydx = (res["depth_m"][11] - res["depth_m"][9]) / (res["x_m"][11] - res["x_m"][9])
        a, p = 3.0 * y, 3.0 + 2 * y
        sf = (5.0 * 0.013) ** 2 / (a ** 2 * (a / p) ** (4 / 3))
        fr2 = 25.0 * 3.0 / (9.80665 * a ** 3)
        self.assertAlmostEqual(-dydx, (0.001 - sf) / (1 - fr2), places=5)  # integrating upstream flips the sign

    def test_steep_slope_supercritical(self):
        res = fl.gvf_profile(5.0, 3.0, 0.05, 0.013, y_control=0.3, length=200.0)
        self.assertFalse(res["subcritical"])
        self.assertAlmostEqual(res["depth_m"][-1], res["normal_depth_m"], delta=1e-3)

    def test_jump_conserves_momentum_and_loses_energy(self):
        r = fl.hydraulic_jump(5.0, 3.0, 0.4)
        g, q, y1, y2 = 9.80665, 5.0 / 3.0, 0.4, r["sequent_depth_m"]
        self.assertAlmostEqual(q ** 2 / (g * y1) + 0.5 * y1 ** 2, q ** 2 / (g * y2) + 0.5 * y2 ** 2, places=9)
        self.assertGreater(r["energy_loss_m"], 0)
        self.assertLess(r["froude_2"], 1.0)

    def test_errors_and_cli(self):
        with self.assertRaises(ValueError):
            fl.hydraulic_jump(5.0, 3.0, 1.5)
        with self.assertRaises(ValueError):
            fl.gvf_profile(5.0, 3.0, -0.001, 0.013, 1.0, 100.0)
        self.assertTrue(fl.main(["jump", "--flow", "5", "--width", "3", "--y1", "0.4"])["sequent_depth_m"] > 0.4)
        self.assertEqual(fl.main(["backwater", "--flow", "5", "--width", "3", "--slope", "0.001", "--y-control", "2",
                                  "--length", "500"])["direction"], "upstream")
        self.assertEqual(fl.cli(["jump", "--flow", "5", "--width", "3", "--y1", "2"]), 2)


class TestThreeD(unittest.TestCase):
    def test_cavity_plot3d_flag(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            fl.main(["cavity", "--re", "100", "--n", "17", "--max-time", "30", "--plot3d", "--out", d])
            self.assertGreater((Path(d) / "cavity_3d.png").stat().st_size, 5000)

    def test_make_3d_registry_and_model(self):
        import make_3d

        self.assertEqual(set(make_3d.ALL), {"mesh", "fields", "channel-model", "network-model", "blasius", "pump", "compare"})
        for name in ("blasius", "pump", "network-model"):
            self.assertTrue(callable(make_3d.ALL[name]))

    def test_bars3d_helper_draws_every_bar(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        import make_3d

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        make_3d.bars3d(ax, [[1.0, 2.0], [3.0, 4.0]], ["a", "b"], ["r1", "r2"], ["C0", "C1"], "v")
        self.assertEqual(len(ax.texts), 4)
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
