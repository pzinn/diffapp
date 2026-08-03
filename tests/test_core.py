from __future__ import annotations

import math
import unittest

from diffapp import (
    DifferentialApproximant,
    FitDiagnostics,
    default_specification,
    fit_default_differential_approximant,
    fit_differential_approximant,
    read_legacy_dataset,
    read_plain_coefficients,
)


class DifferentialApproximantTests(unittest.TestCase):
    def test_common_factor_diagnostics(self) -> None:
        # The geometric-series equation has been multiplied by (1 - 3x):
        #   (-2x)(1-3x) F + (1-2x)(1-3x) D F = 0.
        approximant = DifferentialApproximant(
            q=((0.0, -2.0, 6.0), (1.0, -5.0, 6.0)),
            p=(),
            input_coefficients=(1.0, 2.0, 4.0),
            diagnostics=FitDiagnostics(
                backend="float64",
                precision_digits=15,
                equations=3,
                scaled_condition_number=1.0,
                relative_residual=0.0,
                estimated_stable_digits=15.0,
            ),
        )
        singularities = approximant.singularities()
        removable = singularities[0]
        physical = singularities[1]
        self.assertAlmostEqual(complex(removable.root).real, 1 / 3, places=13)
        self.assertLess(removable.common_factor_residual, 1.0e-14)
        self.assertLess(removable.common_root_distance, 1.0e-14)
        self.assertLess(removable.sylvester_gcd_score, 1.0e-14)
        self.assertEqual(removable.cancellation_details[0].polynomial, "Q0")
        self.assertAlmostEqual(complex(physical.root).real, 0.5, places=13)
        self.assertGreater(physical.common_factor_residual, 1.0e-2)
        self.assertGreater(physical.common_root_distance, 1.0e-2)

    def test_singularities_are_ordered_by_distance_from_origin(self) -> None:
        approximant = DifferentialApproximant(
            q=((1.0,), (-6.0, 1.0, 1.0)),
            p=(),
            input_coefficients=(1.0,),
            diagnostics=FitDiagnostics(
                backend="float64",
                precision_digits=15,
                equations=1,
                scaled_condition_number=1.0,
                relative_residual=0.0,
                estimated_stable_digits=15.0,
            ),
        )
        roots = [complex(item.root).real for item in approximant.singularities()]
        self.assertEqual(roots, [2.0, -3.0])

    def test_default_specifications(self) -> None:
        self.assertEqual(default_specification(23), ((7, 6, 6), 1))
        self.assertEqual(default_specification(12), ((3, 3, 2), 1))
        self.assertEqual(default_specification(3), ((1, 1), -1))
        self.assertEqual(
            default_specification(12, order=1, p_degree=2), ((4, 4), 2)
        )

    def test_geometric_series(self) -> None:
        approximant = fit_differential_approximant(
            [1, 2, 4], q_degrees=(1, 1), p_degree=-1
        )
        singularity = approximant.physical_singularity()
        self.assertIsNotNone(singularity)
        assert singularity is not None
        self.assertAlmostEqual(complex(singularity.root).real, 0.5, places=13)
        self.assertAlmostEqual(complex(singularity.exponent).real, -1.0, places=13)
        self.assertEqual(approximant.extend_series(8), tuple(2.0**n for n in range(8)))

    def test_default_fit_falls_back_for_exact_lower_order_series(self) -> None:
        approximant = fit_default_differential_approximant([1, 2, 4, 8, 16, 32])
        singularity = approximant.physical_singularity()
        self.assertIsNotNone(singularity)
        assert singularity is not None
        self.assertEqual(approximant.coefficients_used, 3)
        self.assertEqual(approximant.diagnostics.holdout_terms, 3)
        self.assertAlmostEqual(complex(singularity.root).real, 0.5, places=13)
        self.assertAlmostEqual(complex(singularity.exponent).real, -1.0, places=13)

    def test_algebraic_singularity(self) -> None:
        # F(x) = (1 - 4x)^-3 has theta=-3 and coefficients C(n+2,2) 4^n.
        coefficients = [math.comb(n + 2, 2) * 4**n for n in range(8)]
        approximant = fit_differential_approximant(
            coefficients, q_degrees=(1, 1), p_degree=-1
        )
        singularity = approximant.physical_singularity()
        self.assertIsNotNone(singularity)
        assert singularity is not None
        self.assertAlmostEqual(complex(singularity.root).real, 0.25, places=13)
        self.assertAlmostEqual(complex(singularity.exponent).real, -3.0, places=13)

    def test_mpmath_backend_preserves_exact_integer_input(self) -> None:
        coefficients = [math.comb(n + 2, 2) * 4**n for n in range(8)]
        approximant = fit_differential_approximant(
            coefficients,
            q_degrees=(1, 1),
            p_degree=-1,
            backend="mpmath",
            precision_digits=60,
        )
        singularity = approximant.physical_singularity()
        self.assertIsNotNone(singularity)
        assert singularity is not None
        self.assertAlmostEqual(float(singularity.root), 0.25, places=14)
        self.assertAlmostEqual(float(singularity.exponent), -3.0, places=14)

    def test_rank_deficiency_and_effective_degree_diagnostics(self) -> None:
        coefficients = read_plain_coefficients("coefficients.txt")
        approximant = fit_differential_approximant(
            coefficients,
            q_degrees=(3, 3, 2),
            p_degree=1,
            backend="mpmath",
            precision_digits=80,
        )
        self.assertLess(
            approximant.diagnostics.numerical_rank,
            approximant.diagnostics.equations,
        )
        self.assertEqual(approximant.effective_p_degree, 0)
        self.assertTrue(
            all(
                math.isinf(singularity.common_root_distance)
                for singularity in approximant.singularities()
            )
        )

        automatic = fit_default_differential_approximant(
            coefficients, backend="mpmath", precision_digits=80
        )
        self.assertEqual(automatic.diagnostics.numerical_rank, 3)
        self.assertEqual(automatic.diagnostics.holdout_terms, 9)
        self.assertEqual(len(automatic.singularities()), 1)
        self.assertAlmostEqual(
            float(automatic.singularities()[0].root), 0.25, places=14
        )

    def test_default_fit_selects_minimal_validated_first_order_model(self) -> None:
        # F(x) = ((1 - 4x)(1 - 8x))^-1/2 satisfies a first-order
        # homogeneous equation with Q degrees (2, 2).
        coefficients = [
            sum(
                math.comb(2 * k, k)
                * math.comb(2 * (n - k), n - k)
                * 2 ** (n - k)
                for k in range(n + 1)
            )
            for n in range(20)
        ]
        for backend in ("float64", "mpmath"):
            approximant = fit_default_differential_approximant(
                coefficients,
                backend=backend,
                precision_digits=80,
            )
            self.assertEqual(approximant.order, 1)
            self.assertEqual(approximant.q_degrees, (2, 2))
            self.assertEqual(approximant.p_degree, -1)
            self.assertEqual(approximant.coefficients_used, 5)
            self.assertEqual(approximant.diagnostics.holdout_terms, 15)
            singularities = approximant.singularities()
            self.assertEqual(len(singularities), 2)
            self.assertAlmostEqual(float(singularities[0].root), 0.125, places=13)
            self.assertAlmostEqual(float(singularities[1].root), 0.25, places=13)
            self.assertAlmostEqual(
                float(singularities[0].exponent), -0.5, places=12
            )
            self.assertAlmostEqual(
                float(singularities[1].exponent), -0.5, places=12
            )

    def test_zinn_legacy_input(self) -> None:
        dataset = read_legacy_dataset("zinn.dat")
        self.assertEqual(dataset.highest_series_order, 22)
        self.assertEqual(len(dataset.coefficients), 23)
        self.assertEqual(dataset.coefficients[0], 1)
        self.assertEqual(dataset.coefficients[-1], 40558226664529024000)
        self.assertEqual(dataset.sweep.minimum_order, 1)
        self.assertEqual(dataset.sweep.maximum_order, 2)

    def test_zinn_first_order_regression(self) -> None:
        dataset = read_legacy_dataset("zinn.dat")
        approximant = fit_differential_approximant(
            dataset.coefficients, q_degrees=(5, 5), p_degree=1
        )
        singularity = approximant.physical_singularity(
            real_interval=(0.0875169, 0.0877169)
        )
        self.assertIsNotNone(singularity)
        assert singularity is not None
        self.assertAlmostEqual(complex(singularity.root).real, 0.0876152, places=7)
        self.assertAlmostEqual(complex(singularity.exponent).real, 1.825461, places=6)

    def test_plain_coefficient_input(self) -> None:
        self.assertEqual(
            read_plain_coefficients("tests/data/simple_series.txt"),
            (1, 2, 4, 8, 16, 32),
        )


if __name__ == "__main__":
    unittest.main()
