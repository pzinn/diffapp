"""Format-independent construction and analysis of approximant families."""

from __future__ import annotations

import itertools
import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import mpmath as mp

from .core import fit_differential_approximant, required_coefficients
from .model import ApproximantError, Backend, DifferentialApproximant, Singularity


@dataclass(frozen=True, order=True)
class SweepSpecification:
    """Polynomial degrees defining one differential approximant."""

    q_degrees: tuple[int, ...]
    p_degree: int

    @property
    def order(self) -> int:
        return len(self.q_degrees) - 1

    @property
    def coefficients_used(self) -> int:
        return required_coefficients(self.q_degrees, self.p_degree)


@dataclass(frozen=True)
class SweepConfig:
    """Controls specification generation, root selection, and clustering."""

    orders: tuple[int, ...] = (1, 2)
    p_degrees: tuple[int, ...] | None = None
    degree_spread: int = 2
    max_terms_omitted: int | None = None
    root_interval: tuple[float, float] | None = None
    imaginary_tolerance: float = 1.0e-8
    cluster_tolerance: float = 1.0e-4
    minimum_cluster_fraction: float = 0.5
    reject_rank_deficient: bool = True

    def validated(self, coefficient_count: int) -> "SweepConfig":
        if coefficient_count < 3:
            raise ApproximantError("at least three coefficients are needed for a sweep")
        if not self.orders or any(order < 1 for order in self.orders):
            raise ApproximantError("sweep orders must be positive")
        if self.p_degrees is not None and any(degree < -1 for degree in self.p_degrees):
            raise ApproximantError("sweep P degrees must be -1 or non-negative")
        if self.degree_spread < 0:
            raise ApproximantError("degree spread must be non-negative")
        if self.max_terms_omitted is not None and self.max_terms_omitted < 0:
            raise ApproximantError("maximum omitted terms must be non-negative")
        if self.root_interval is not None:
            lower, upper = self.root_interval
            if upper < lower:
                raise ApproximantError("invalid sweep root interval")
        if self.imaginary_tolerance < 0:
            raise ApproximantError("imaginary tolerance must be non-negative")
        if self.cluster_tolerance <= 0:
            raise ApproximantError("cluster tolerance must be positive")
        if not 0 < self.minimum_cluster_fraction <= 1:
            raise ApproximantError("minimum cluster fraction must lie in (0, 1]")
        return self


@dataclass(frozen=True)
class SweepEstimate:
    """One candidate singularity contributed by one accepted approximant."""

    specification: SweepSpecification
    approximant: DifferentialApproximant
    singularity: Singularity

    @property
    def root(self) -> Any:
        return self.singularity.root

    @property
    def exponent(self) -> Any | None:
        return self.singularity.exponent


@dataclass(frozen=True)
class SweepRejection:
    specification: SweepSpecification
    reason: str
    detail: str = ""


@dataclass(frozen=True)
class SweepFit:
    """One successfully fitted member of a sweep family."""

    specification: SweepSpecification
    approximant: DifferentialApproximant


@dataclass(frozen=True)
class FitSweepResult:
    """Approximants accepted before any root-specific analysis."""

    config: SweepConfig
    specifications: tuple[SweepSpecification, ...]
    fits: tuple[SweepFit, ...]
    rejections: tuple[SweepRejection, ...]


@dataclass(frozen=True)
class RootCluster:
    """A recurring root and descriptive spread across approximants."""

    root: Any
    root_spread: float
    exponent: Any | None
    exponent_spread: float | None
    estimates: tuple[SweepEstimate, ...]
    approximant_support: int
    support_fraction: float
    recurring: bool

    @property
    def orders(self) -> tuple[int, ...]:
        return tuple(sorted({estimate.specification.order for estimate in self.estimates}))


@dataclass(frozen=True)
class SweepResult:
    config: SweepConfig
    specifications: tuple[SweepSpecification, ...]
    estimates: tuple[SweepEstimate, ...]
    rejections: tuple[SweepRejection, ...]
    clusters: tuple[RootCluster, ...]

    @property
    def accepted_specifications(self) -> tuple[SweepSpecification, ...]:
        return tuple(
            sorted({estimate.specification for estimate in self.estimates})
        )

    @property
    def recurring_clusters(self) -> tuple[RootCluster, ...]:
        return tuple(cluster for cluster in self.clusters if cluster.recurring)


def default_p_degrees(coefficient_count: int) -> tuple[int, ...]:
    """Return a modest range of inhomogeneous degrees for an automatic sweep."""

    return tuple(range(0, min(8, max(1, coefficient_count // 3)) + 1))


def _balanced_q_degrees(
    q_coefficient_count: int, polynomial_count: int, degree_spread: int
) -> tuple[tuple[int, ...], ...]:
    if q_coefficient_count < 2 * polynomial_count:
        return ()
    average = q_coefficient_count / polynomial_count
    lower = max(2, math.floor(average) - degree_spread)
    upper = math.ceil(average) + degree_spread
    degrees: set[tuple[int, ...]] = set()
    for sizes in itertools.product(range(lower, upper + 1), repeat=polynomial_count):
        if sum(sizes) != q_coefficient_count:
            continue
        if max(sizes) - min(sizes) > degree_spread:
            continue
        degrees.add(tuple(size - 1 for size in sizes))
    return tuple(sorted(degrees))


def generate_sweep_specifications(
    coefficient_count: int, config: SweepConfig = SweepConfig()
) -> tuple[SweepSpecification, ...]:
    """Generate balanced specifications using all or nearly all coefficients."""

    config.validated(coefficient_count)
    p_degrees = config.p_degrees or default_p_degrees(coefficient_count)
    omitted = (
        min(10, coefficient_count - 3)
        if config.max_terms_omitted is None
        else min(config.max_terms_omitted, coefficient_count - 3)
    )
    minimum_used = coefficient_count - omitted
    specifications: set[SweepSpecification] = set()
    for order in sorted(set(config.orders)):
        polynomial_count = order + 1
        for p_degree in sorted(set(p_degrees)):
            for used in range(minimum_used, coefficient_count + 1):
                q_coefficient_count = (
                    used - p_degree if p_degree >= 0 else used + 1
                )
                for q_degrees in _balanced_q_degrees(
                    q_coefficient_count, polynomial_count, config.degree_spread
                ):
                    specification = SweepSpecification(q_degrees, p_degree)
                    if specification.coefficients_used == used:
                        specifications.add(specification)
    return tuple(
        sorted(
            specifications,
            key=lambda item: (
                item.order,
                item.p_degree,
                -item.coefficients_used,
                item.q_degrees,
            ),
        )
    )


def _selected_singularities(
    approximant: DifferentialApproximant, config: SweepConfig
) -> tuple[Singularity, ...]:
    singularities = approximant.singularities()
    if config.root_interval is None:
        return singularities
    lower, upper = config.root_interval
    candidates = [
        singularity
        for singularity in singularities
        if lower <= float(mp.re(singularity.root)) <= upper
        and abs(float(mp.im(singularity.root)))
        <= config.imaginary_tolerance
        * max(1.0, abs(float(mp.re(singularity.root))))
    ]
    if not candidates:
        return ()
    return (min(candidates, key=lambda item: abs(complex(item.root))),)


def _median_complex(values: Sequence[Any]) -> Any:
    real = statistics.median(mp.re(value) for value in values)
    imaginary = statistics.median(mp.im(value) for value in values)
    if all(isinstance(value, mp.mpf) for value in values):
        return mp.mpf(real)
    if any(isinstance(value, (mp.mpf, mp.mpc)) for value in values):
        return mp.mpc(real, imaginary)
    return complex(real, imaginary)


def _cluster_estimates(
    estimates: Sequence[SweepEstimate],
    tolerance: float,
    minimum_fraction: float,
    accepted_count: int,
) -> tuple[RootCluster, ...]:
    groups: list[list[SweepEstimate]] = []
    centers: list[Any] = []
    for estimate in sorted(estimates, key=lambda item: abs(item.root)):
        distances = [
            abs(estimate.root - center) / (1 + abs(center)) for center in centers
        ]
        eligible = [index for index, distance in enumerate(distances) if distance <= tolerance]
        if not eligible:
            groups.append([estimate])
            centers.append(estimate.root)
            continue
        index = min(eligible, key=lambda candidate: distances[candidate])
        groups[index].append(estimate)
        centers[index] = sum(item.root for item in groups[index]) / len(groups[index])

    clusters: list[RootCluster] = []
    for group in groups:
        roots = [estimate.root for estimate in group]
        center = _median_complex(roots)
        root_spread = statistics.median(abs(root - center) for root in roots)
        exponent_values = [
            estimate.exponent for estimate in group if estimate.exponent is not None
        ]
        exponent = _median_complex(exponent_values) if exponent_values else None
        exponent_spread = (
            statistics.median(abs(value - exponent) for value in exponent_values)
            if exponent is not None
            else None
        )
        support = len({estimate.specification for estimate in group})
        fraction = support / accepted_count if accepted_count else 0.0
        clusters.append(
            RootCluster(
                root=center,
                root_spread=float(root_spread),
                exponent=exponent,
                exponent_spread=(
                    float(exponent_spread) if exponent_spread is not None else None
                ),
                estimates=tuple(group),
                approximant_support=support,
                support_fraction=fraction,
                recurring=fraction >= minimum_fraction,
            )
        )
    return tuple(
        sorted(
            clusters,
            key=lambda cluster: (
                not cluster.recurring,
                abs(cluster.root),
                -cluster.support_fraction,
            ),
        )
    )


def fit_sweep(
    coefficients: Sequence[Any],
    config: SweepConfig = SweepConfig(),
    *,
    specifications: Sequence[SweepSpecification] | None = None,
    backend: Backend = "float64",
    precision_digits: int = 80,
) -> FitSweepResult:
    """Fit a family of approximants with common rank and size checks."""

    config.validated(len(coefficients))
    selected_specifications = tuple(
        dict.fromkeys(
            specifications
            if specifications is not None
            else generate_sweep_specifications(len(coefficients), config)
        )
    )
    fits: list[SweepFit] = []
    rejections: list[SweepRejection] = []
    for specification in selected_specifications:
        if specification.coefficients_used > len(coefficients):
            rejections.append(
                SweepRejection(specification, "insufficient-coefficients")
            )
            continue
        try:
            approximant = fit_differential_approximant(
                coefficients,
                specification.q_degrees,
                specification.p_degree,
                backend=backend,
                precision_digits=precision_digits,
            )
            rank = approximant.diagnostics.numerical_rank
            if (
                config.reject_rank_deficient
                and rank is not None
                and rank < approximant.diagnostics.equations
            ):
                rejections.append(
                    SweepRejection(
                        specification,
                        "rank-deficient",
                        f"rank {rank} of {approximant.diagnostics.equations}",
                    )
                )
                continue
        except ApproximantError as error:
            rejections.append(
                SweepRejection(specification, "fit-failed", str(error))
            )
            continue
        fits.append(SweepFit(specification, approximant))

    return FitSweepResult(
        config=config,
        specifications=selected_specifications,
        fits=tuple(fits),
        rejections=tuple(rejections),
    )


def run_sweep(
    coefficients: Sequence[Any],
    config: SweepConfig = SweepConfig(),
    *,
    specifications: Sequence[SweepSpecification] | None = None,
    backend: Backend = "float64",
    precision_digits: int = 80,
) -> SweepResult:
    """Fit a family of approximants and cluster their candidate roots."""

    fitted = fit_sweep(
        coefficients,
        config,
        specifications=specifications,
        backend=backend,
        precision_digits=precision_digits,
    )
    estimates: list[SweepEstimate] = []
    rejections = list(fitted.rejections)
    for fit in fitted.fits:
        try:
            singularities = _selected_singularities(fit.approximant, config)
        except ApproximantError as error:
            rejections.append(
                SweepRejection(
                    fit.specification, "root-analysis-failed", str(error)
                )
            )
            continue
        if not singularities:
            rejections.append(
                SweepRejection(fit.specification, "no-selected-root")
            )
            continue
        estimates.extend(
            SweepEstimate(fit.specification, fit.approximant, singularity)
            for singularity in singularities
        )

    accepted_count = len({estimate.specification for estimate in estimates})
    clusters = _cluster_estimates(
        estimates,
        config.cluster_tolerance,
        config.minimum_cluster_fraction,
        accepted_count,
    )
    return SweepResult(
        config=config,
        specifications=fitted.specifications,
        estimates=tuple(estimates),
        rejections=tuple(rejections),
        clusters=clusters,
    )
