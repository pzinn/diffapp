"""Single-family extrapolation using differential-approximant recurrences."""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import mpmath as mp

from .model import ApproximantError, Backend, DifferentialApproximant
from .sweep import (
    SweepConfig,
    SweepRejection,
    SweepSpecification,
    fit_sweep,
)


@dataclass(frozen=True)
class ExtensionEstimate:
    """A coefficient sequence extrapolated by one accepted approximant."""

    specification: SweepSpecification
    approximant: DifferentialApproximant
    coefficients: tuple[Any, ...]
    holdout_terms: int
    holdout_median_error: float | None
    holdout_max_error: float | None


@dataclass(frozen=True)
class CoefficientForecast:
    """Robust ensemble summary for one extrapolated coefficient."""

    index: int
    median: Any
    median_absolute_deviation: Any
    relative_spread: float
    minimum: Any
    maximum: Any
    support: int


@dataclass(frozen=True)
class ExtensionSweepResult:
    config: SweepConfig
    input_coefficient_count: int
    total_coefficient_count: int
    specifications: tuple[SweepSpecification, ...]
    estimates: tuple[ExtensionEstimate, ...]
    rejections: tuple[SweepRejection, ...]
    forecasts: tuple[CoefficientForecast, ...]

    @property
    def validated_estimate_count(self) -> int:
        return sum(estimate.holdout_terms > 0 for estimate in self.estimates)


def _numeric_input(value: Any, backend: Backend) -> Any:
    if backend == "float64":
        return float(value)
    if isinstance(value, Decimal):
        return mp.mpf(str(value))
    if isinstance(value, int):
        return mp.mpf(value)
    return mp.mpf(str(value))


def _holdout_errors(
    predicted: Sequence[Any],
    actual: Sequence[Any],
    first_holdout: int,
    backend: Backend,
) -> tuple[float | None, float | None, int]:
    errors = [
        float(
            abs(predicted[index] - _numeric_input(actual[index], backend))
            / (1 + abs(_numeric_input(actual[index], backend)))
        )
        for index in range(first_holdout, len(actual))
    ]
    if not errors:
        return None, None, 0
    return float(statistics.median(errors)), max(errors), len(errors)


def _finite(value: Any) -> bool:
    try:
        return bool(mp.isfinite(value))
    except (TypeError, ValueError):
        return False


def _summarize_forecasts(
    estimates: Sequence[ExtensionEstimate], first_index: int, total: int
) -> tuple[CoefficientForecast, ...]:
    forecasts: list[CoefficientForecast] = []
    for index in range(first_index, total):
        values = [
            estimate.coefficients[index]
            for estimate in estimates
            if _finite(estimate.coefficients[index])
        ]
        if not values:
            continue
        median = statistics.median(values)
        deviation = statistics.median(abs(value - median) for value in values)
        forecasts.append(
            CoefficientForecast(
                index=index,
                median=median,
                median_absolute_deviation=deviation,
                relative_spread=float(deviation / (1 + abs(median))),
                minimum=min(values),
                maximum=max(values),
                support=len(values),
            )
        )
    return tuple(forecasts)


def extend_sweep(
    coefficients: Sequence[Any],
    total_coefficients: int,
    config: SweepConfig = SweepConfig(),
    *,
    specifications: Sequence[SweepSpecification] | None = None,
    backend: Backend = "float64",
    precision_digits: int = 80,
    maximum_holdout_error: float | None = None,
) -> ExtensionSweepResult:
    """Extrapolate with a family and summarize each new coefficient robustly.

    Fits that use fewer than all supplied coefficients are scored against the
    known trailing coefficients before they contribute to the ensemble.
    ``maximum_holdout_error`` optionally rejects fits by their largest
    normalized holdout error.
    """

    if total_coefficients <= len(coefficients):
        raise ApproximantError(
            "extension length must exceed the number of input coefficients"
        )
    if maximum_holdout_error is not None and maximum_holdout_error < 0:
        raise ApproximantError("maximum holdout error must be non-negative")

    fitted = fit_sweep(
        coefficients,
        config,
        specifications=specifications,
        backend=backend,
        precision_digits=precision_digits,
    )
    context = (
        mp.workdps(precision_digits) if backend == "mpmath" else nullcontext()
    )
    with context:
        estimates: list[ExtensionEstimate] = []
        rejections = list(fitted.rejections)
        for fit in fitted.fits:
            try:
                extended = fit.approximant.extend_series(total_coefficients)
            except ApproximantError as error:
                rejections.append(
                    SweepRejection(
                        fit.specification, "extension-failed", str(error)
                    )
                )
                continue
            median_error, max_error, holdout_terms = _holdout_errors(
                extended,
                coefficients,
                fit.specification.coefficients_used,
                backend,
            )
            if (
                maximum_holdout_error is not None
                and max_error is not None
                and max_error > maximum_holdout_error
            ):
                rejections.append(
                    SweepRejection(
                        fit.specification,
                        "holdout-error",
                        f"{max_error:.3e} exceeds {maximum_holdout_error:.3e}",
                    )
                )
                continue
            estimates.append(
                ExtensionEstimate(
                    specification=fit.specification,
                    approximant=fit.approximant,
                    coefficients=extended,
                    holdout_terms=holdout_terms,
                    holdout_median_error=median_error,
                    holdout_max_error=max_error,
                )
            )

        forecasts = _summarize_forecasts(
            estimates, len(coefficients), total_coefficients
        )
    return ExtensionSweepResult(
        config=config,
        input_coefficient_count=len(coefficients),
        total_coefficient_count=total_coefficients,
        specifications=fitted.specifications,
        estimates=tuple(estimates),
        rejections=tuple(rejections),
        forecasts=forecasts,
    )
