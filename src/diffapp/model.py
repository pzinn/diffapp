"""Public data models for differential approximants."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import mpmath as mp
import numpy as np


Backend = Literal["float64", "mpmath"]
Scalar = Any


class ApproximantError(ValueError):
    """Base exception for an invalid or failed differential approximant."""


class SingularSystemError(ApproximantError):
    """Raised when the defining linear system is singular."""


@dataclass(frozen=True)
class FitDiagnostics:
    """Numerical diagnostics for the linear system defining an approximant."""

    backend: Backend
    precision_digits: int
    equations: int
    scaled_condition_number: float | None
    relative_residual: float
    estimated_stable_digits: float | None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class PolynomialCancellation:
    """Cancellation evidence contributed by one polynomial."""

    polynomial: str
    normalized_residual: float
    nearest_root_distance: float | None
    sylvester_gcd_score: float | None


@dataclass(frozen=True)
class Singularity:
    """A zero of the highest-order polynomial and its indicial exponent.

    ``exponent`` uses the conventional differential-approximant definition
    ``F(x) ~ (1 - x / root)**exponent``.  A divergent generating function
    therefore normally has a negative exponent.

    ``common_factor_residual`` is the maximum normalized evaluation of every
    other active polynomial at this root. ``common_root_distance`` is the
    maximum normalized distance to their nearest roots. ``sylvester_gcd_score``
    is a global, pairwise approximate-GCD diagnostic and is therefore the same
    for every root of a given approximant.
    """

    root: Scalar
    exponent: Scalar | None
    normalized_root_residual: float
    derivative_magnitude: float
    common_factor_residual: float | None = None
    common_root_distance: float | None = None
    sylvester_gcd_score: float | None = None
    cancellation_details: tuple[PolynomialCancellation, ...] = ()


def _polyval(coefficients: Sequence[Scalar], value: Scalar) -> Scalar:
    result = coefficients[-1] * 0
    for coefficient in reversed(coefficients):
        result = result * value + coefficient
    return result


def _polyder(coefficients: Sequence[Scalar], value: Scalar) -> Scalar:
    if len(coefficients) <= 1:
        return coefficients[0] * 0
    result = coefficients[-1] * (len(coefficients) - 1)
    for power in range(len(coefficients) - 2, 0, -1):
        result = result * value + coefficients[power] * power
    return result


def _normalized_polynomial_residual(
    coefficients: Sequence[Scalar], root: Scalar
) -> float:
    numerator = abs(_polyval(coefficients, root))
    denominator = sum(
        abs(coefficient) * abs(root) ** power
        for power, coefficient in enumerate(coefficients)
    )
    if denominator == 0:
        return float(numerator)
    return float(numerator / denominator)


def _trim_polynomial(coefficients: Sequence[Scalar]) -> tuple[Scalar, ...]:
    result = list(coefficients)
    while result and result[-1] == 0:
        result.pop()
    return tuple(result)


def _sylvester_gcd_score(
    first: Sequence[Scalar], second: Sequence[Scalar]
) -> float | None:
    """Return a scale-free pairwise approximate-GCD score.

    Zero means that the double-precision Sylvester matrix is singular. Values
    near one provide no evidence for a common polynomial factor.
    """

    first = _trim_polynomial(first)
    second = _trim_polynomial(second)
    if not first or not second:
        return None
    first_degree = len(first) - 1
    second_degree = len(second) - 1
    if first_degree == 0 or second_degree == 0:
        return 1.0

    def normalized_descending(values: Sequence[Scalar]) -> np.ndarray:
        scale = max(abs(value) for value in values)
        return np.asarray(
            [complex(value / scale) for value in reversed(values)],
            dtype=np.complex128,
        )

    first_values = normalized_descending(first)
    second_values = normalized_descending(second)
    size = first_degree + second_degree
    matrix = np.zeros((size, size), dtype=np.complex128)
    for row in range(second_degree):
        matrix[row, row : row + first_degree + 1] = first_values
    for offset in range(first_degree):
        row = second_degree + offset
        matrix[row, offset : offset + second_degree + 1] = second_values
    try:
        singular_values = np.linalg.svd(matrix, compute_uv=False)
    except np.linalg.LinAlgError:
        return None
    if singular_values[0] == 0:
        return 0.0
    return float(singular_values[-1] / singular_values[0])


@dataclass(frozen=True)
class DifferentialApproximant:
    """A fitted inhomogeneous differential approximant.

    Polynomial coefficients are stored in ascending order.  ``q[k]`` is the
    polynomial multiplying ``(x d/dx)**k``.
    """

    q: tuple[tuple[Scalar, ...], ...]
    p: tuple[Scalar, ...]
    input_coefficients: tuple[Scalar, ...]
    diagnostics: FitDiagnostics

    @property
    def order(self) -> int:
        return len(self.q) - 1

    @property
    def q_degrees(self) -> tuple[int, ...]:
        return tuple(len(polynomial) - 1 for polynomial in self.q)

    @property
    def p_degree(self) -> int:
        return len(self.p) - 1

    @property
    def coefficients_used(self) -> int:
        return self.diagnostics.equations

    def singularities(self) -> tuple[Singularity, ...]:
        """Return roots of the head polynomial and simple-root exponents."""

        head = self.q[-1]
        if len(head) <= 1:
            return ()

        if self.diagnostics.backend == "mpmath":
            with mp.workdps(self.diagnostics.precision_digits):
                try:
                    roots = list(
                        mp.polyroots(
                            list(head),
                            asc=True,
                            maxsteps=1000,
                            extraprec=max(20, self.diagnostics.precision_digits // 2),
                        )
                    )
                except (mp.libmp.libhyper.NoConvergence, ZeroDivisionError) as error:
                    raise ApproximantError(
                        "arbitrary-precision polynomial root finding did not converge"
                    ) from error
                tolerance = mp.power(
                    10, -max(12, self.diagnostics.precision_digits // 2)
                )
                return self._analyse_roots(roots, tolerance)

        roots = list(np.polynomial.polynomial.polyroots(np.asarray(head)))
        return self._analyse_roots(roots, 1.0e-10)

    def _analyse_roots(
        self, roots: list[Scalar], tolerance: Scalar
    ) -> tuple[Singularity, ...]:
        head = self.q[-1]
        labelled_polynomials = [
            (f"Q{index}", polynomial)
            for index, polynomial in enumerate(self.q[:-1])
        ]
        labelled_polynomials.append(("P", self.p))
        labelled_polynomials = [
            (label, polynomial)
            for label, polynomial in labelled_polynomials
            if _trim_polynomial(polynomial)
        ]
        other_polynomials = [polynomial for _, polynomial in labelled_polynomials]
        other_roots = [
            self._diagnostic_roots(polynomial) for polynomial in other_polynomials
        ]
        pairwise_gcd_scores = [
            _sylvester_gcd_score(head, polynomial)
            for polynomial in other_polynomials
        ]
        available_gcd_scores = [
            score for score in pairwise_gcd_scores if score is not None
        ]
        combined_gcd_score = (
            max(available_gcd_scores) if available_gcd_scores else None
        )
        roots.sort(key=lambda root: (float(mp.re(root)), float(mp.im(root))))
        lower = self.q[-2] if self.order >= 1 else None
        singularities: list[Singularity] = []
        for root in roots:
            derivative = _polyder(head, root)
            derivative_scale = sum(
                power * abs(coefficient) * abs(root) ** max(0, power - 1)
                for power, coefficient in enumerate(head)
            )
            is_simple = derivative_scale != 0 and abs(derivative) > tolerance * derivative_scale
            exponent = None
            if lower is not None and is_simple and root != 0:
                exponent = self.order - 1 - _polyval(lower, root) / (root * derivative)
            polynomial_residuals = [
                _normalized_polynomial_residual(polynomial, root)
                for polynomial in other_polynomials
            ]
            common_factor_residual = (
                max(polynomial_residuals) if polynomial_residuals else None
            )
            root_distances: list[float] = []
            root_distances_available = True
            for polynomial_roots in other_roots:
                if polynomial_roots is None:
                    root_distances_available = False
                    break
                if not polynomial_roots:
                    root_distances.append(math.inf)
                else:
                    root_distances.append(
                        float(
                            min(abs(root - other) for other in polynomial_roots)
                            / (1 + abs(root))
                        )
                    )
            common_root_distance = (
                max(root_distances)
                if root_distances_available and root_distances
                else None
            )
            cancellation_details = tuple(
                PolynomialCancellation(
                    polynomial=label,
                    normalized_residual=polynomial_residual,
                    nearest_root_distance=(
                        root_distances[index]
                        if root_distances_available
                        else None
                    ),
                    sylvester_gcd_score=pairwise_gcd_scores[index],
                )
                for index, ((label, _), polynomial_residual) in enumerate(
                    zip(labelled_polynomials, polynomial_residuals, strict=True)
                )
            )
            singularities.append(
                Singularity(
                    root=root,
                    exponent=exponent,
                    normalized_root_residual=_normalized_polynomial_residual(head, root),
                    derivative_magnitude=float(abs(derivative)),
                    common_factor_residual=common_factor_residual,
                    common_root_distance=common_root_distance,
                    sylvester_gcd_score=combined_gcd_score,
                    cancellation_details=cancellation_details,
                )
            )
        return tuple(singularities)

    def _diagnostic_roots(
        self, coefficients: Sequence[Scalar]
    ) -> list[Scalar] | None:
        coefficients = _trim_polynomial(coefficients)
        if len(coefficients) <= 1:
            return []
        try:
            if self.diagnostics.backend == "mpmath":
                return list(
                    mp.polyroots(
                        list(coefficients),
                        asc=True,
                        maxsteps=1000,
                        extraprec=max(
                            20, self.diagnostics.precision_digits // 2
                        ),
                    )
                )
            return list(
                np.polynomial.polynomial.polyroots(np.asarray(coefficients))
            )
        except (
            mp.libmp.libhyper.NoConvergence,
            np.linalg.LinAlgError,
            ZeroDivisionError,
        ):
            return None

    def physical_singularity(
        self,
        imaginary_tolerance: float = 1.0e-8,
        real_interval: tuple[float, float] = (0.0, math.inf),
    ) -> Singularity | None:
        """Return the closest positive-real candidate in ``real_interval``.

        The mathematics alone cannot identify which positive root is the
        physical one.  A known interval is therefore preferable when one is
        available.
        """

        lower, upper = real_interval
        if lower < 0 or upper < lower:
            raise ApproximantError("invalid physical-singularity interval")
        candidates = [
            singularity
            for singularity in self.singularities()
            if lower <= float(mp.re(singularity.root)) <= upper
            and float(mp.re(singularity.root)) > 0
            and abs(float(mp.im(singularity.root)))
            <= imaginary_tolerance * max(1.0, abs(float(mp.re(singularity.root))))
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda item: float(mp.re(item.root)))

    def extend_series(self, total_coefficients: int) -> tuple[Scalar, ...]:
        """Generate further coefficients using the fitted recurrence."""

        if self.diagnostics.backend == "mpmath":
            with mp.workdps(self.diagnostics.precision_digits):
                return self._extend_series(total_coefficients)
        return self._extend_series(total_coefficients)

    def _extend_series(self, total_coefficients: int) -> tuple[Scalar, ...]:
        if total_coefficients <= len(self.input_coefficients):
            return self.input_coefficients[:total_coefficients]

        coefficients = list(self.input_coefficients)
        zero = self.q[0][0] * 0
        relative_tolerance = (
            100 * mp.eps
            if self.diagnostics.backend == "mpmath"
            else 100 * np.finfo(float).eps
        )
        for n in range(len(coefficients), total_coefficients):
            numerator = self.p[n] if n < len(self.p) else zero
            denominator = zero
            for derivative_order, polynomial in enumerate(self.q):
                denominator += polynomial[0] * n**derivative_order
                for power in range(1, min(n, len(polynomial) - 1) + 1):
                    numerator -= (
                        polynomial[power]
                        * (n - power) ** derivative_order
                        * coefficients[n - power]
                    )
            scale = sum(abs(polynomial[0]) * n**order for order, polynomial in enumerate(self.q))
            if scale == 0 or abs(denominator) <= scale * relative_tolerance:
                raise ApproximantError(
                    f"recurrence denominator is zero or numerically singular at coefficient {n}"
                )
            coefficients.append(numerator / denominator)
        return tuple(coefficients)
