from __future__ import annotations

import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from diffapp import (
    SweepConfig,
    SweepSpecification,
    default_p_degrees,
    extend_sweep,
    generate_sweep_specifications,
    read_plain_coefficients,
    run_sweep,
)
from diffapp.cli import main


class SweepTests(unittest.TestCase):
    def test_default_p_degree_range_scales_with_series_length(self) -> None:
        self.assertEqual(default_p_degrees(12), (0, 1, 2, 3, 4))
        self.assertEqual(default_p_degrees(100), tuple(range(9)))

    def test_generator_includes_balanced_full_length_specification(self) -> None:
        specifications = generate_sweep_specifications(
            23,
            SweepConfig(
                orders=(2,),
                p_degrees=(5,),
                degree_spread=0,
                max_terms_omitted=0,
            ),
        )
        self.assertEqual(specifications, (SweepSpecification((5, 5, 5), 5),))

    def test_generator_uses_only_requested_nearby_sizes(self) -> None:
        specifications = generate_sweep_specifications(
            12,
            SweepConfig(
                orders=(1,),
                p_degrees=(0,),
                degree_spread=1,
                max_terms_omitted=2,
            ),
        )
        self.assertTrue(specifications)
        for specification in specifications:
            self.assertGreaterEqual(specification.coefficients_used, 10)
            self.assertLessEqual(specification.coefficients_used, 12)
            self.assertLessEqual(
                max(specification.q_degrees) - min(specification.q_degrees), 1
            )

    def test_sweep_clusters_recurring_exact_singularity(self) -> None:
        coefficients = read_plain_coefficients("coefficients.txt")
        result = run_sweep(
            coefficients,
            SweepConfig(root_interval=(0.2, 0.3)),
        )
        self.assertGreater(len(result.accepted_specifications), 1)
        self.assertTrue(any(item.reason == "rank-deficient" for item in result.rejections))
        self.assertEqual(len(result.recurring_clusters), 1)
        cluster = result.recurring_clusters[0]
        self.assertAlmostEqual(cluster.root.real, 0.25, places=12)
        self.assertAlmostEqual(cluster.exponent.real, -3.0, places=10)
        self.assertEqual(cluster.support_fraction, 1.0)

    def test_modern_cli_json_is_machine_readable(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            status = main(
                [
                    "sweep",
                    "coefficients.txt",
                    "--orders",
                    "1",
                    "--p-degrees",
                    "-1:1",
                    "--root-min",
                    "0.2",
                    "--root-max",
                    "0.3",
                    "--output",
                    "json",
                ]
            )
        self.assertEqual(status, 0)
        payload = json.loads(output.getvalue())
        self.assertGreater(payload["summary"]["accepted"], 0)
        self.assertEqual(payload["summary"]["recurring_clusters"], 1)
        self.assertAlmostEqual(payload["clusters"][0]["root"]["real"], 0.25)

    def test_legacy_cli_routes_encoded_specifications_through_sweep(self) -> None:
        coefficients = [math.comb(n + 2, 2) * 4**n for n in range(8)]
        contents = "\n".join(
            [
                "0 0 0",
                "7",
                *(str(value) for value in coefficients),
                "1 1 1 2 0 0 0 0 -1 0",
                "9",
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.dat"
            path.write_text(contents)
            output = io.StringIO()
            with redirect_stdout(output):
                status = main(["legacy-sweep", str(path), "--output", "json"])
        self.assertEqual(status, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["summary"]["specifications"], 12)
        self.assertGreater(payload["summary"]["accepted"], 0)
        self.assertEqual(payload["summary"]["recurring_clusters"], 1)

    def test_extension_sweep_uses_holdouts_and_recovers_coefficients(self) -> None:
        coefficients = [math.comb(n + 2, 2) * 4**n for n in range(8)]
        result = extend_sweep(
            coefficients,
            11,
            SweepConfig(root_interval=None),
        )
        self.assertGreater(len(result.estimates), 1)
        self.assertGreater(result.validated_estimate_count, 0)
        expected = [math.comb(n + 2, 2) * 4**n for n in range(8, 11)]
        self.assertEqual([item.index for item in result.forecasts], [8, 9, 10])
        for forecast, value in zip(result.forecasts, expected, strict=True):
            self.assertAlmostEqual(float(forecast.median), value, places=5)
            self.assertLess(forecast.relative_spread, 1.0e-10)

    def test_single_extension_cli_can_emit_only_predictions(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            status = main(
                [
                    "extend",
                    "coefficients.txt",
                    "--terms",
                    "14",
                    "--predicted-only",
                    "--output",
                    "plain",
                ]
            )
        self.assertEqual(status, 0)
        self.assertEqual(
            [float(line) for line in output.getvalue().splitlines()],
            [1526726656.0, 7046430720.0],
        )

    def test_extension_cli_defaults_to_ten_additional_terms(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            status = main(
                [
                    "extend",
                    "coefficients.txt",
                    "--predicted-only",
                    "--output",
                    "plain",
                ]
            )
        self.assertEqual(status, 0)
        self.assertEqual(len(output.getvalue().splitlines()), 10)

        output = io.StringIO()
        with redirect_stdout(output):
            status = main(
                ["extend-sweep", "coefficients.txt", "--output", "json"]
            )
        self.assertEqual(status, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["summary"]["input_coefficients"], 12)
        self.assertEqual(payload["summary"]["total_coefficients"], 22)
        self.assertEqual(len(payload["forecasts"]), 10)

    def test_extension_sweep_cli_json_is_machine_readable(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            status = main(
                [
                    "extend-sweep",
                    "coefficients.txt",
                    "--terms",
                    "14",
                    "--output",
                    "json",
                ]
            )
        self.assertEqual(status, 0)
        payload = json.loads(output.getvalue())
        self.assertGreater(payload["summary"]["extended"], 1)
        self.assertGreater(payload["summary"]["holdout_validated"], 0)
        self.assertEqual(
            [forecast["index"] for forecast in payload["forecasts"]],
            [12, 13],
        )


if __name__ == "__main__":
    unittest.main()
