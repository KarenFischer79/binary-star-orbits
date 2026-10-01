import unittest
from pathlib import Path
import sys
import csv
import tempfile
from unittest.mock import patch
from dataclasses import replace
from orbits import read_weights_csv, observation_weights
from orbits import optimize_orbit, OrbitElements, Observation, WDSData, solve_kepler
from orbits import filter_by_residual
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from orbits import apparent_position, read_wds_file, write_residuals_csv, create_weights_csv, main


PROJECT_FILE = Path(
    "/Volumes/Astro02/BoyceAstro/USNO/wds16289+1825_edit.txt"
)


class OrbitsPartOneTests(unittest.TestCase):
    def test_grade5_filter_reproduces_seven_paper_exclusions(self):
        data = read_wds_file(Path(__file__).parent / "InputOrbit" / "wds18540+3723_edit.txt")
        retained, extra, audit = filter_by_residual(data, (), 300)
        excluded = [row for row in audit if row["status"] == "excluded"]
        self.assertEqual([row["epoch"] for row in excluded], [1873.44, 1896.77, 1925.57, 1950.47, 1959.6, 2020.609, 2020.609])
        self.assertEqual(len(retained.observations), 133)
        self.assertEqual(len(data.observations), 140)
        self.assertEqual(len({row["observation_index"] for row in audit}), 140)
        # Equal-to-cutoff is retained; additional records use the same rule.
        threshold = audit[0]["published_residual_mas"]
        _, _, boundary = filter_by_residual(data, (data.observations[0],), threshold)
        self.assertEqual(boundary[0]["status"], "retained")
        self.assertEqual(boundary[-1]["status"], "retained")
        with self.assertRaises(ValueError):
            filter_by_residual(data, (), float("nan"))
        with self.assertRaises(ValueError):
            filter_by_residual(data, (), 0)

    def test_optimization_recovers_synthetic_orbit_and_ignores_zero_weight(self):
        truth = OrbitElements("Synthetic", 40, 1.5, 60, 110, 2000, 0.4, 70)
        epochs = np.linspace(1980, 2060, 50)
        _, _, theta, rho = apparent_position(epochs, truth)
        observations = tuple(Observation(t, th, r, technique="S") for t, th, r in zip(epochs, theta, rho))
        observations += (Observation(2020, 0, 1000, technique="bad"),)
        initial = replace(truth, period_years=43, semimajor_arcsec=1.4, eccentricity=0.35, periastron_epoch=2001)
        data = WDSData("test", observations, initial)
        fitted, report = optimize_orbit(data, {"S": 2, "bad": 0})
        self.assertLess(report["weighted_positional_rms_arcsec_after"], 1e-7)
        self.assertAlmostEqual(fitted.orbit.period_years, 40, places=4)
        self.assertEqual(report["zero_weight_pairs"], 1)
        scaled, _ = optimize_orbit(data, {"S": 200, "bad": 0})
        self.assertAlmostEqual(scaled.orbit.period_years, fitted.orbit.period_years, places=6)
        self.assertTrue(all(value is None for value in fitted.orbit.uncertainties))
        with self.assertRaisesRegex(ValueError, "positive-weight"):
            optimize_orbit(data, {"S": 0, "bad": 0})

    def test_kepler_high_eccentricity(self):
        mean = np.linspace(0, 2 * np.pi, 1000, endpoint=False)
        eccentric = solve_kepler(mean, 0.999999)
        np.testing.assert_allclose(eccentric - 0.999999 * np.sin(eccentric), mean, atol=1e-12)

    def test_kepler_roundoff_stall_from_grade5_fit(self):
        from scipy.optimize import brentq
        mean = 6.282814217584966
        e = 0.9999175810865232
        expected = brentq(lambda value: value - e * np.sin(value) - mean,
                          0, 2 * np.pi, xtol=1e-14)
        actual = solve_kepler(np.array([mean, 0, 1e-10]), e)
        self.assertLess(abs(actual[0] - expected), 2e-13)
        np.testing.assert_allclose(actual - e * np.sin(actual), [mean, 0, 1e-10], atol=1e-14)

    def test_weight_validation_and_matching(self):
        data = read_wds_file(Path(__file__).parent / "InputOrbit" / "wds16289+1825_edit.txt")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.csv"
            for body in ["Ma,-1", "Ma,nan", "Ma,inf", "Ma,", "Ma,abc", "Ma,1\nMa,2"]:
                with self.subTest(body=body):
                    path.write_text("technique_code,weight\n" + body, encoding="utf-8-sig")
                    with self.assertRaises(ValueError):
                        read_weights_csv(path)
            path.write_text("technique_code,weight\nMa,2.5\nS,0\n", encoding="utf-8-sig")
            weights = read_weights_csv(path)
            observations = [replace(data.observations[0], technique=code) for code in ["S", "Ma", "Ma"]]
            self.assertEqual(observation_weights(observations, weights).tolist(), [0, 2.5, 2.5])
            with self.assertRaisesRegex(ValueError, "missing"):
                observation_weights(data.observations, weights)
            output = Path(directory) / "residuals.csv"
            write_residuals_csv(replace(data, observations=tuple(observations)), output, weights=weights)
            with output.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([float(row["technique_weight"]) for row in rows], [0, 2.5, 2.5])

    def test_weights_cli_counts_complete_pairs_and_preserves_edits(self):
        source = Path(__file__).parent / "InputOrbit" / "wds16289+1825_edit.txt"
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / source.name
            input_path.write_bytes(source.read_bytes())
            with patch("orbits.make_plot") as plot:
                self.assertEqual(main([str(input_path), "--create-weights"]), 0)
                plot.assert_not_called()
            output = input_path.with_name(input_path.stem + "_weights.csv")
            with output.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 21)
            self.assertEqual(sum(int(row["measurement_count"]) for row in rows), 576)
            self.assertTrue(all(row["weight"] == "0" for row in rows))
            self.assertEqual(next(row["measurement_count"] for row in rows if row["technique_code"] == "Ma"), "388")
            edited = output.read_bytes().replace(b",0\r\n", b",2.5\r\n", 1)
            output.write_bytes(edited)
            self.assertEqual(main([str(input_path), "--create-weights"]), 2)
            self.assertEqual(output.read_bytes(), edited)

    def test_weights_retains_unknown_code_and_rejects_empty_data(self):
        source = Path(__file__).parent / "InputOrbit" / "wds16289+1825_edit.txt"
        data = read_wds_file(source)
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "sample.txt"
            with self.assertRaisesRegex(ValueError, "No complete"):
                create_weights_csv(replace(data, observations=()), input_path)
            unknown = replace(data.observations[0], technique="")
            output = create_weights_csv(replace(data, observations=(unknown,)), input_path)
            with output.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows, [{"technique_code": "", "measurement_count": "1", "weight": "0"}])

    @unittest.skipUnless(PROJECT_FILE.exists(), "Project source file is not mounted")
    def test_parser_reads_stf2052(self):
        data = read_wds_file(PROJECT_FILE)
        self.assertEqual(data.wds_id, "16289+1825")
        self.assertEqual(data.orbit.designation, "STF2052AB")
        self.assertEqual(len(data.observations), 576)
        self.assertAlmostEqual(data.orbit.period_years, 229.4999)
        self.assertAlmostEqual(data.orbit.semimajor_arcsec, 2.2305)

    @unittest.skipUnless(PROJECT_FILE.exists(), "Project source file is not mounted")
    def test_2025_ephemeris_matches_orb6_rounding(self):
        orbit = read_wds_file(PROJECT_FILE).orbit
        _, _, theta, rho = apparent_position([2025.0], orbit)
        # The source file prints 116.5 degrees and 2.563 arcsec for 2025.0.
        self.assertLess(abs(float(theta[0]) - 116.5), 0.15)
        self.assertLess(abs(float(rho[0]) - 2.563), 0.001)

    @unittest.skipUnless(PROJECT_FILE.exists(), "Project source file is not mounted")
    def test_residual_csv_has_every_complete_pair_and_blanks_space_aperture(self):
        data = read_wds_file(PROJECT_FILE)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "residuals.csv"
            write_residuals_csv(data, output)
            with output.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 576)
        self.assertIn("total_residual_mas", rows[0])
        space_rows = [row for row in rows if row["technique_code"].startswith("H")]
        self.assertTrue(space_rows)
        self.assertTrue(all(row["aperture_m"] == "" for row in space_rows))


if __name__ == "__main__":
    unittest.main()
