"""Construction of differential approximants."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from decimal import Decimal
from typing import Any

import mpmath as mp
import numpy as np
import scipy.linalg

from .model import (
    ApproximantError,
    Backend,
    DifferentialApproximant,
    FitDiagnostics,
    SingularSystemError,
)


Descriptor = tuple[str, int, int]


def _validate(
    coefficients: Sequence[Any], q_degrees: Sequence[int], p_degree: int, backend: Backend
) -> tuple[int, ...]:
    degrees = tuple(int(degree) for degree in q_degrees)
    if not degrees:
        raise ApproximantError("at least one Q polynomial is required")
    if any(degree < 0 for degree in degrees):
        raise ApproximantError("Q polynomial degrees must be non-negative")
    if p_degree < -1:
        raise ApproximantError("P degree must be -1 (homogeneous) or non-negative")
    if backend not in ("float64", "mpmath"):
        raise ApproximantError(f"unknown numerical backend: {backend}")
    if not coefficients:
        raise ApproximantError("the input series is empty")
    return degrees


def _unknowns(q_degrees: tuple[int, ...], p_degree: int) -> list[Descriptor]:
    order = len(q_degrees) - 1
    unknowns = [
        ("q", derivative_order, power)
        for derivative_order, degree in enumerate(q_degrees)
        for power in range(degree + 1)
        if not (derivative_order == order and power == 0)
    ]
    if p_degree >= 0:
        unknowns.extend(("p", 0, power) for power in range(p_degree + 1))
    return unknowns


def required_coefficients(q_degrees: Sequence[int], p_degree: int) -> int:
    """Return the number of consecutive series coefficients needed for a fit."""

    degrees = tuple(int(degree) for degree in q_degrees)
    return len(_unknowns(degrees, p_degree))


def default_specification(
    coefficient_count: int,
    *,
    order: int | None = None,
    p_degree: int | None = None,
) -> tuple[tuple[int, ...], int]:
    """Choose a balanced approximant specification for a series length.

    The default prefers a second-order approximant and a linear
    inhomogeneous polynomial.  If the series is too short, it falls back to a
    homogeneous approximant and then to first order.  Polynomial degrees are
    kept within one of each other, with any remainder assigned from ``Q0``
    upward.
    """

    if coefficient_count < 3:
        raise ApproximantError(
            "at least three coefficients are needed to choose a useful default"
        )
    if order is not None and order < 1:
        raise ApproximantError("automatic differential-equation order must be positive")
    if p_degree is not None and p_degree < -1:
        raise ApproximantError("P degree must be -1 (homogeneous) or non-negative")

    orders = (order,) if order is not None else (2, 1)
    p_candidates = (p_degree,) if p_degree is not None else (1, -1)
    for candidate_order in orders:
        polynomial_count = candidate_order + 1
        for candidate_p_degree in p_candidates:
            assert candidate_p_degree is not None
            q_coefficient_count = (
                coefficient_count - candidate_p_degree
                if candidate_p_degree >= 0
                else coefficient_count + 1
            )
            # Requiring degree >= 1 for every Q gives the head polynomial at
            # least one finite root and avoids especially weak defaults.
            if q_coefficient_count < 2 * polynomial_count:
                continue
            base, remainder = divmod(q_coefficient_count, polynomial_count)
            q_degrees = tuple(
                base + (1 if index < remainder else 0) - 1
                for index in range(polynomial_count)
            )
            if required_coefficients(q_degrees, candidate_p_degree) == coefficient_count:
                return q_degrees, candidate_p_degree

    requested = "the requested order" if order is not None else "first order"
    raise ApproximantError(
        f"{coefficient_count} coefficients are insufficient for {requested} "
        "with non-constant balanced Q polynomials"
    )


def _coefficient_value(
    descriptor: Descriptor, n: int, coefficients: Sequence[Any]
) -> Any:
    kind, derivative_order, power = descriptor
    if kind == "p":
        return -1 if n == power else 0
    if n < power:
        return 0
    return coefficients[n - power] * (n - power) ** derivative_order


def _float64_fit(
    coefficients: Sequence[Any], q_degrees: tuple[int, ...], p_degree: int
) -> tuple[list[float], FitDiagnostics]:
    unknowns = _unknowns(q_degrees, p_degree)
    size = len(unknowns)
    values = np.asarray([float(value) for value in coefficients[:size]], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ApproximantError("coefficients are outside the finite float64 range")

    matrix = np.empty((size, size), dtype=np.float64)
    rhs = np.empty(size, dtype=np.float64)
    order = len(q_degrees) - 1
    for n in range(size):
        for column, descriptor in enumerate(unknowns):
            matrix[n, column] = _coefficient_value(descriptor, n, values)
        rhs[n] = -(n**order) * values[n]

    row_scales = np.maximum(np.max(np.abs(matrix), axis=1), np.abs(rhs))
    if np.any(row_scales == 0):
        raise SingularSystemError("the defining system contains an all-zero equation")
    row_matrix = matrix / row_scales[:, None]
    row_rhs = rhs / row_scales
    column_scales = np.max(np.abs(row_matrix), axis=0)
    if np.any(column_scales == 0):
        raise SingularSystemError("the defining system contains an unconstrained coefficient")
    scaled_matrix = row_matrix / column_scales[None, :]

    captured: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", scipy.linalg.LinAlgWarning)
            scaled_solution = scipy.linalg.solve(
                scaled_matrix, row_rhs, assume_a="gen", check_finite=True
            )
        captured.extend(str(item.message) for item in caught)
    except np.linalg.LinAlgError as error:
        raise SingularSystemError("the defining linear system is singular") from error
    solution = scaled_solution / column_scales

    singular_values = np.linalg.svd(scaled_matrix, compute_uv=False)
    condition = float(singular_values[0] / singular_values[-1])
    rank_tolerance = (
        singular_values[0] * max(scaled_matrix.shape) * np.finfo(float).eps
    )
    numerical_rank = int(np.count_nonzero(singular_values > rank_tolerance))
    residual = matrix @ solution - rhs
    denominator = (
        np.linalg.norm(matrix, ord=np.inf) * np.linalg.norm(solution, ord=np.inf)
        + np.linalg.norm(rhs, ord=np.inf)
    )
    relative_residual = float(np.linalg.norm(residual, ord=np.inf) / denominator)
    stable_digits = (
        max(0.0, -math.log10(condition * np.finfo(float).eps))
        if condition > 0
        else None
    )
    if condition > 1.0e12:
        captured.append(
            f"scaled system is ill-conditioned (condition number {condition:.3e})"
        )
    if numerical_rank < size:
        captured.append(
            f"scaled system is numerically rank deficient "
            f"(rank {numerical_rank} of {size}); fitted roots may depend on precision"
        )
    coefficient_zero_tolerance = min(
        0.1,
        max(
            100 * np.finfo(float).eps,
            10 * condition * np.finfo(float).eps,
        ),
    )
    diagnostics = FitDiagnostics(
        backend="float64",
        precision_digits=15,
        equations=size,
        scaled_condition_number=condition,
        relative_residual=relative_residual,
        estimated_stable_digits=stable_digits,
        numerical_rank=numerical_rank,
        coefficient_zero_tolerance=coefficient_zero_tolerance,
        warnings=tuple(captured),
    )
    return solution.tolist(), diagnostics


def _to_mpf(value: Any) -> mp.mpf:
    if isinstance(value, Decimal):
        return mp.mpf(str(value))
    if isinstance(value, (int, np.integer)):
        return mp.mpf(int(value))
    return mp.mpf(str(value))


def _mpmath_fit(
    coefficients: Sequence[Any],
    q_degrees: tuple[int, ...],
    p_degree: int,
    precision_digits: int,
) -> tuple[list[mp.mpf], FitDiagnostics]:
    unknowns = _unknowns(q_degrees, p_degree)
    size = len(unknowns)
    with mp.workdps(precision_digits):
        values = [_to_mpf(value) for value in coefficients[:size]]
        matrix = mp.matrix(size, size)
        rhs = mp.matrix(size, 1)
        order = len(q_degrees) - 1
        for n in range(size):
            for column, descriptor in enumerate(unknowns):
                matrix[n, column] = _coefficient_value(descriptor, n, values)
            rhs[n] = -(mp.mpf(n) ** order) * values[n]

        row_scales: list[mp.mpf] = []
        for row in range(size):
            scale = max([abs(rhs[row])] + [abs(matrix[row, column]) for column in range(size)])
            if scale == 0:
                raise SingularSystemError("the defining system contains an all-zero equation")
            row_scales.append(scale)
        row_matrix = mp.matrix(size, size)
        row_rhs = mp.matrix(size, 1)
        for row in range(size):
            row_rhs[row] = rhs[row] / row_scales[row]
            for column in range(size):
                row_matrix[row, column] = matrix[row, column] / row_scales[row]
        column_scales = [
            max(abs(row_matrix[row, column]) for row in range(size))
            for column in range(size)
        ]
        if any(scale == 0 for scale in column_scales):
            raise SingularSystemError("the defining system contains an unconstrained coefficient")
        scaled_matrix = mp.matrix(size, size)
        for row in range(size):
            for column in range(size):
                scaled_matrix[row, column] = row_matrix[row, column] / column_scales[column]
        try:
            scaled_solution = mp.lu_solve(scaled_matrix, row_rhs)
        except (ZeroDivisionError, ValueError) as error:
            raise SingularSystemError("the defining linear system is singular") from error
        solution = [scaled_solution[column] / column_scales[column] for column in range(size)]

        residual_norm = max(
            abs(sum(matrix[row, column] * solution[column] for column in range(size)) - rhs[row])
            for row in range(size)
        )
        matrix_norm = max(
            sum(abs(matrix[row, column]) for column in range(size)) for row in range(size)
        )
        solution_norm = max(abs(value) for value in solution)
        rhs_norm = max(abs(rhs[row]) for row in range(size))
        denominator = matrix_norm * solution_norm + rhs_norm
        relative_residual = float(residual_norm / denominator) if denominator else 0.0
        captured: list[str] = []
        try:
            singular_values = mp.svd(scaled_matrix, compute_uv=False)
            largest = singular_values[0]
            smallest = singular_values[size - 1]
            condition_mp = largest / smallest if smallest else mp.inf
            rank_tolerance = largest * size * mp.eps
            numerical_rank = sum(
                singular_values[index] > rank_tolerance for index in range(size)
            )
            try:
                condition = float(condition_mp)
            except (OverflowError, ValueError):
                condition = math.inf
            stable_digits = max(
                0.0,
                precision_digits - float(mp.log10(condition_mp)),
            ) if condition_mp > 0 else None
            coefficient_zero_tolerance = float(
                min(
                    mp.mpf("0.1"),
                    max(100 * mp.eps, 10 * condition_mp * mp.eps),
                )
            )
        except (ValueError, ZeroDivisionError):
            condition = None
            numerical_rank = None
            stable_digits = None
            coefficient_zero_tolerance = float(mp.power(10, 4 - precision_digits))
            captured.append("unable to estimate arbitrary-precision matrix rank")
        if condition is not None and condition > 10 ** (precision_digits / 2):
            captured.append(
                f"scaled system is ill-conditioned (condition number {condition:.3e})"
            )
        if numerical_rank is not None and numerical_rank < size:
            captured.append(
                f"scaled system is numerically rank deficient "
                f"(rank {numerical_rank} of {size}); fitted roots may depend on precision"
            )
        diagnostics = FitDiagnostics(
            backend="mpmath",
            precision_digits=precision_digits,
            equations=size,
            scaled_condition_number=condition,
            relative_residual=relative_residual,
            estimated_stable_digits=stable_digits,
            numerical_rank=numerical_rank,
            coefficient_zero_tolerance=coefficient_zero_tolerance,
            warnings=tuple(captured),
        )
        return solution, diagnostics


def _assemble_approximant(
    solution: Sequence[Any],
    coefficients: Sequence[Any],
    q_degrees: tuple[int, ...],
    p_degree: int,
    diagnostics: FitDiagnostics,
) -> DifferentialApproximant:
    precision = diagnostics.precision_digits if diagnostics.backend == "mpmath" else 15
    with mp.workdps(precision):
        zero = solution[0] * 0
        one = zero + 1
        q = [[zero for _ in range(degree + 1)] for degree in q_degrees]
        q[-1][0] = one
        p = [zero for _ in range(p_degree + 1)] if p_degree >= 0 else []
        for descriptor, value in zip(_unknowns(q_degrees, p_degree), solution, strict=True):
            kind, derivative_order, power = descriptor
            if kind == "q":
                q[derivative_order][power] = value
            else:
                p[power] = value
        used = diagnostics.equations
        stored_coefficients = tuple(
            _to_mpf(value) if diagnostics.backend == "mpmath" else float(value)
            for value in coefficients[:used]
        )
        return DifferentialApproximant(
            q=tuple(tuple(polynomial) for polynomial in q),
            p=tuple(p),
            input_coefficients=stored_coefficients,
            diagnostics=diagnostics,
        )


def fit_differential_approximant(
    coefficients: Sequence[Any],
    q_degrees: Sequence[int],
    p_degree: int = -1,
    *,
    backend: Backend = "float64",
    precision_digits: int = 80,
) -> DifferentialApproximant:
    """Fit a differential approximant to consecutive coefficients from x**0.

    The fitted equation is

    ``sum(Q[k](x) * (x d/dx)**k F(x), k=0..M) = P(x)``.

    ``Q[M](0)`` is normalized to one.  Set ``p_degree=-1`` for a
    homogeneous approximant.  Inputs such as integers and ``Decimal`` values
    are converted only after the backend is selected.
    """

    degrees = _validate(coefficients, q_degrees, p_degree, backend)
    needed = required_coefficients(degrees, p_degree)
    if len(coefficients) < needed:
        raise ApproximantError(
            f"approximant needs {needed} coefficients but only {len(coefficients)} were supplied"
        )
    if backend == "float64":
        solution, diagnostics = _float64_fit(coefficients, degrees, p_degree)
    else:
        if precision_digits < 20:
            raise ApproximantError("arbitrary precision must be at least 20 decimal digits")
        solution, diagnostics = _mpmath_fit(
            coefficients, degrees, p_degree, precision_digits
        )
    return _assemble_approximant(
        solution, coefficients, degrees, p_degree, diagnostics
    )


def fit_default_differential_approximant(
    coefficients: Sequence[Any],
    *,
    order: int | None = None,
    p_degree: int | None = None,
    backend: Backend = "float64",
    precision_digits: int = 80,
) -> DifferentialApproximant:
    """Fit the default balanced approximant, with singular-system fallback.

    The longest possible fit is tried first.  Exact lower-order series can
    make an unnecessarily high-order system singular; in that case the
    coefficient count is reduced until a nonsingular balanced fit is found.
    """

    last_error: SingularSystemError | None = None
    tried: set[tuple[tuple[int, ...], int]] = set()
    for available in range(len(coefficients), 2, -1):
        try:
            q_degrees, selected_p_degree = default_specification(
                available, order=order, p_degree=p_degree
            )
        except ApproximantError:
            continue
        specification = (q_degrees, selected_p_degree)
        if specification in tried:
            continue
        tried.add(specification)
        try:
            approximant = fit_differential_approximant(
                coefficients,
                q_degrees,
                selected_p_degree,
                backend=backend,
                precision_digits=precision_digits,
            )
            rank = approximant.diagnostics.numerical_rank
            if rank is not None and rank < approximant.diagnostics.equations:
                last_error = SingularSystemError(
                    "automatic approximant is numerically rank deficient"
                )
                continue
            return approximant
        except SingularSystemError as error:
            last_error = error
    if last_error is not None:
        raise SingularSystemError(
            "no nonsingular automatic approximant could be constructed"
        ) from last_error
    raise ApproximantError("the series is too short for an automatic approximant")
