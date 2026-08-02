"""Command-line interface for differential-approximant analysis."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import mpmath as mp

from .core import (
    fit_default_differential_approximant,
    fit_differential_approximant,
    required_coefficients,
)
from .io import read_legacy_dataset, read_plain_coefficients
from .model import ApproximantError, Singularity
from .sweep import (
    RootCluster,
    SweepConfig,
    SweepEstimate,
    SweepResult,
    SweepSpecification,
    run_sweep,
)


def _format_number(value: Any, digits: int = 12) -> str:
    if isinstance(value, (mp.mpf, mp.mpc)):
        return mp.nstr(value, digits)
    value = complex(value)
    if abs(value.imag) <= 1.0e-13 * max(1.0, abs(value.real)):
        return f"{value.real:.{digits}g}"
    return f"{value.real:.{digits}g}{value.imag:+.{digits}g}j"


def _describe_singularity(singularity: Singularity, digits: int) -> str:
    exponent = (
        "indeterminate"
        if singularity.exponent is None
        else _format_number(singularity.exponent, digits)
    )
    description = (
        f"x = {_format_number(singularity.root, digits)}, "
        f"theta = {exponent}, root residual = {singularity.normalized_root_residual:.2e}"
    )
    cancellation = []
    if singularity.common_factor_residual is not None:
        cancellation.append(f"max={singularity.common_factor_residual:.2e}")
    if singularity.common_root_distance is not None:
        cancellation.append(f"root-distance={singularity.common_root_distance:.2e}")
    if singularity.sylvester_gcd_score is not None:
        cancellation.append(f"gcd-svd={singularity.sylvester_gcd_score:.2e}")
    if cancellation:
        description += "\n  cancellation: " + ", ".join(cancellation)
    return description


def _backend_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend", choices=("float64", "mpmath"), default="float64"
    )
    parser.add_argument(
        "--precision", type=int, default=80, help="decimal digits for the mpmath backend"
    )
    parser.add_argument("--digits", type=int, default=12, help="displayed significant digits")


def _fit_command(arguments: argparse.Namespace) -> int:
    coefficients = (
        read_legacy_dataset(arguments.path).coefficients
        if arguments.format == "legacy"
        else read_plain_coefficients(arguments.path)
    )
    automatic = arguments.q_degrees is None
    if automatic:
        approximant = fit_default_differential_approximant(
            coefficients,
            order=arguments.order,
            p_degree=arguments.p_degree,
            backend=arguments.backend,
            precision_digits=arguments.precision,
        )
        q_degrees = approximant.q_degrees
        p_degree = approximant.p_degree
    else:
        if arguments.order is not None:
            raise ApproximantError("--order cannot be combined with --q-degrees")
        q_degrees = tuple(int(value) for value in arguments.q_degrees.split(","))
        if arguments.p_degree is not None:
            p_degree = arguments.p_degree
        elif required_coefficients(q_degrees, 1) <= len(coefficients):
            p_degree = 1
        else:
            p_degree = -1
        approximant = fit_differential_approximant(
            coefficients,
            q_degrees,
            p_degree,
            backend=arguments.backend,
            precision_digits=arguments.precision,
        )
    if automatic:
        print(
            f"automatic specification: Q degrees {q_degrees}, P degree {p_degree}"
        )
    print(
        f"order {approximant.order}, Q degrees {approximant.q_degrees}, "
        f"P degree {approximant.p_degree}, coefficients used {approximant.coefficients_used}"
    )
    diagnostics = approximant.diagnostics
    effective_degrees = (
        approximant.effective_q_degrees,
        approximant.effective_p_degree,
    )
    nominal_degrees = (approximant.q_degrees, approximant.p_degree)
    if effective_degrees != nominal_degrees:
        print(
            f"effective degrees: Q {approximant.effective_q_degrees}, "
            f"P {approximant.effective_p_degree} "
            f"(relative cutoff {diagnostics.coefficient_zero_tolerance:.2e})"
        )
    if diagnostics.scaled_condition_number is not None:
        print(f"scaled condition number: {diagnostics.scaled_condition_number:.3e}")
    if diagnostics.numerical_rank is not None:
        print(
            f"numerical rank: {diagnostics.numerical_rank}/{diagnostics.equations}"
        )
    print(f"relative residual: {diagnostics.relative_residual:.3e}")
    for warning in diagnostics.warnings:
        print(f"warning: {warning}")
    for singularity in approximant.singularities():
        print(_describe_singularity(singularity, arguments.digits))
    return 0


def _integer_selection(value: str) -> tuple[int, ...]:
    """Parse comma-separated integers and inclusive ``start:stop`` ranges."""

    selected: list[int] = []
    try:
        for part in value.split(","):
            token = part.strip()
            if not token:
                raise ValueError
            if ":" not in token:
                selected.append(int(token))
                continue
            endpoints = token.split(":")
            if len(endpoints) != 2:
                raise ValueError
            start, stop = (int(endpoint) for endpoint in endpoints)
            step = 1 if stop >= start else -1
            selected.extend(range(start, stop + step, step))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected comma-separated integers or inclusive ranges such as 0:8"
        ) from error
    return tuple(dict.fromkeys(selected))


def _root_interval(arguments: argparse.Namespace) -> tuple[float, float] | None:
    if arguments.root_min is None and arguments.root_max is None:
        return None
    return (
        -math.inf if arguments.root_min is None else arguments.root_min,
        math.inf if arguments.root_max is None else arguments.root_max,
    )


def _sweep_config(arguments: argparse.Namespace) -> SweepConfig:
    return SweepConfig(
        orders=arguments.orders,
        p_degrees=arguments.p_degrees,
        degree_spread=arguments.degree_spread,
        max_terms_omitted=arguments.max_terms_omitted,
        root_interval=_root_interval(arguments),
        imaginary_tolerance=arguments.imaginary_tolerance,
        cluster_tolerance=arguments.cluster_tolerance,
        minimum_cluster_fraction=arguments.minimum_cluster_fraction,
        reject_rank_deficient=not arguments.include_rank_deficient,
    )


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _complex_record(value: Any | None) -> dict[str, float | None] | None:
    if value is None:
        return None
    converted = complex(value)
    return {
        "real": _finite_float(converted.real),
        "imag": _finite_float(converted.imag),
    }


def _specification_record(specification: SweepSpecification) -> dict[str, Any]:
    return {
        "order": specification.order,
        "q_degrees": list(specification.q_degrees),
        "p_degree": specification.p_degree,
        "coefficients_used": specification.coefficients_used,
    }


def _estimate_record(estimate: SweepEstimate) -> dict[str, Any]:
    diagnostics = estimate.approximant.diagnostics
    singularity = estimate.singularity
    return {
        **_specification_record(estimate.specification),
        "root": _complex_record(singularity.root),
        "exponent": _complex_record(singularity.exponent),
        "condition_number": _finite_float(diagnostics.scaled_condition_number),
        "numerical_rank": diagnostics.numerical_rank,
        "relative_residual": _finite_float(diagnostics.relative_residual),
        "cancellation_max": _finite_float(singularity.common_factor_residual),
        "common_root_distance": _finite_float(singularity.common_root_distance),
        "gcd_svd": _finite_float(singularity.sylvester_gcd_score),
    }


def _cluster_record(cluster: RootCluster) -> dict[str, Any]:
    return {
        "root": _complex_record(cluster.root),
        "root_spread": _finite_float(cluster.root_spread),
        "exponent": _complex_record(cluster.exponent),
        "exponent_spread": _finite_float(cluster.exponent_spread),
        "approximant_support": cluster.approximant_support,
        "support_fraction": cluster.support_fraction,
        "orders": list(cluster.orders),
        "recurring": cluster.recurring,
    }


def _print_estimate(estimate: SweepEstimate, digits: int) -> None:
    specification = estimate.specification
    singularity = estimate.singularity
    exponent = (
        "?"
        if singularity.exponent is None
        else _format_number(singularity.exponent, digits)
    )
    condition = estimate.approximant.diagnostics.scaled_condition_number
    condition_text = (
        "n/a"
        if condition is None or not math.isfinite(condition)
        else f"{condition:.2e}"
    )
    cancellation_text = (
        f"{singularity.common_factor_residual:.1e}"
        if singularity.common_factor_residual is not None
        else "n/a"
    )
    print(
        f"Q={specification.q_degrees} P={specification.p_degree:2d} "
        f"used={specification.coefficients_used:2d} "
        f"x={_format_number(singularity.root, digits):>16} "
        f"theta={exponent:>16} cond={condition_text} max={cancellation_text}"
    )


def _reported_clusters(
    result: SweepResult, show_all_clusters: bool
) -> tuple[RootCluster, ...]:
    return result.clusters if show_all_clusters else result.recurring_clusters


def _write_sweep_table(result: SweepResult, arguments: argparse.Namespace) -> None:
    accepted = len(result.accepted_specifications)
    print(
        f"{len(result.specifications)} specifications: {accepted} accepted, "
        f"{len(result.rejections)} rejected; {len(result.estimates)} root estimates"
    )
    if arguments.show_approximants:
        print("\nApproximant estimates:")
        for estimate in result.estimates:
            _print_estimate(estimate, arguments.digits)

    clusters = _reported_clusters(result, arguments.all_clusters)
    heading = (
        "All root clusters"
        if arguments.all_clusters
        else "Recurring root clusters"
    )
    print(f"\n{heading}:")
    if not clusters:
        print("none")
    else:
        print(
            f"{'root':>18} {'theta':>18} {'root spread':>12} "
            f"{'theta spread':>12} {'support':>9} {'fraction':>9} {'orders':>8}"
        )
        for cluster in clusters:
            exponent = (
                "?"
                if cluster.exponent is None
                else _format_number(cluster.exponent, arguments.digits)
            )
            exponent_spread = (
                "n/a"
                if cluster.exponent_spread is None
                else f"{cluster.exponent_spread:.2e}"
            )
            orders = ",".join(str(order) for order in cluster.orders)
            print(
                f"{_format_number(cluster.root, arguments.digits):>18} "
                f"{exponent:>18} {cluster.root_spread:12.2e} "
                f"{exponent_spread:>12} {cluster.approximant_support:9d} "
                f"{cluster.support_fraction:9.1%} {orders:>8}"
            )

    if arguments.verbose and result.rejections:
        print("\nRejected specifications:", file=sys.stderr)
        for rejection in result.rejections:
            detail = f": {rejection.detail}" if rejection.detail else ""
            print(
                f"Q={rejection.specification.q_degrees} "
                f"P={rejection.specification.p_degree}: {rejection.reason}{detail}",
                file=sys.stderr,
            )


def _write_sweep_json(result: SweepResult, arguments: argparse.Namespace) -> None:
    payload = {
        "summary": {
            "specifications": len(result.specifications),
            "accepted": len(result.accepted_specifications),
            "rejected": len(result.rejections),
            "root_estimates": len(result.estimates),
            "clusters": len(result.clusters),
            "recurring_clusters": len(result.recurring_clusters),
        },
        "clusters": [
            _cluster_record(cluster)
            for cluster in _reported_clusters(result, arguments.all_clusters)
        ],
        "estimates": [
            _estimate_record(estimate) for estimate in result.estimates
        ],
        "rejections": [
            {
                **_specification_record(rejection.specification),
                "reason": rejection.reason,
                "detail": rejection.detail,
            }
            for rejection in result.rejections
        ],
    }
    json.dump(payload, sys.stdout, indent=2, allow_nan=False)
    print()


def _write_sweep_csv(result: SweepResult, arguments: argparse.Namespace) -> None:
    writer = csv.writer(sys.stdout)
    writer.writerow(
        (
            "root_real",
            "root_imag",
            "root_spread",
            "exponent_real",
            "exponent_imag",
            "exponent_spread",
            "approximant_support",
            "support_fraction",
            "orders",
            "recurring",
        )
    )
    for cluster in _reported_clusters(result, arguments.all_clusters):
        exponent = cluster.exponent
        writer.writerow(
            (
                cluster.root.real,
                cluster.root.imag,
                cluster.root_spread,
                "" if exponent is None else exponent.real,
                "" if exponent is None else exponent.imag,
                "" if cluster.exponent_spread is None else cluster.exponent_spread,
                cluster.approximant_support,
                cluster.support_fraction,
                ";".join(str(order) for order in cluster.orders),
                str(cluster.recurring).lower(),
            )
        )


def _report_sweep(result: SweepResult, arguments: argparse.Namespace) -> None:
    if arguments.output == "json":
        _write_sweep_json(result, arguments)
    elif arguments.output == "csv":
        _write_sweep_csv(result, arguments)
    else:
        _write_sweep_table(result, arguments)


def _modern_sweep_command(arguments: argparse.Namespace) -> int:
    coefficients = (
        read_legacy_dataset(arguments.path).coefficients
        if arguments.format == "legacy"
        else read_plain_coefficients(arguments.path)
    )
    result = run_sweep(
        coefficients,
        _sweep_config(arguments),
        backend=arguments.backend,
        precision_digits=arguments.precision,
    )
    _report_sweep(result, arguments)
    return 0


def _legacy_sweep_command(arguments: argparse.Namespace) -> int:
    dataset = read_legacy_dataset(arguments.path)
    specifications = tuple(
        SweepSpecification(q_degrees, p_degree)
        for q_degrees, p_degree in dataset.sweep.specifications()
    )
    result = run_sweep(
        dataset.coefficients,
        _sweep_config(arguments),
        specifications=specifications,
        backend=arguments.backend,
        precision_digits=arguments.precision,
    )
    _report_sweep(result, arguments)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="diffapp", description="Analyse power series with differential approximants"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit_parser = subparsers.add_parser("fit", help="fit one approximant")
    fit_parser.add_argument("path", type=Path, help="coefficient file")
    fit_parser.add_argument(
        "--format",
        choices=("plain", "legacy"),
        default="plain",
        help="input format (plain decimal coefficients by default)",
    )
    fit_parser.add_argument(
        "--q-degrees",
        help="comma-separated degrees Q0,...,QM; default is a balanced automatic choice",
    )
    fit_parser.add_argument(
        "--p-degree",
        type=int,
        help="degree of P; -1 is homogeneous, default is 1 when practical",
    )
    fit_parser.add_argument(
        "--order",
        type=int,
        help="differential-equation order for automatic Q degrees (default: 2)",
    )
    _backend_arguments(fit_parser)
    fit_parser.set_defaults(handler=_fit_command)

    def add_sweep_arguments(
        sweep_parser: argparse.ArgumentParser, *, generated: bool
    ) -> None:
        if generated:
            sweep_parser.add_argument(
                "--orders",
                type=_integer_selection,
                default=(1, 2),
                help=(
                    "equation orders, as integers or inclusive ranges "
                    "(default: 1,2)"
                ),
            )
            sweep_parser.add_argument(
                "--p-degrees",
                type=_integer_selection,
                help="P degrees, e.g. 0:8 or -1,0,1 (default: automatic range)",
            )
            sweep_parser.add_argument(
                "--degree-spread",
                type=int,
                default=2,
                help="maximum degree difference among Q polynomials (default: 2)",
            )
            sweep_parser.add_argument(
                "--max-terms-omitted",
                type=int,
                help="generate fits omitting at most this many trailing coefficients",
            )
        sweep_parser.add_argument("--imaginary-tolerance", type=float, default=1.0e-8)
        sweep_parser.add_argument(
            "--root-min", type=float, help="lower end of the selected real-root interval"
        )
        sweep_parser.add_argument(
            "--root-max", type=float, help="upper end of the selected real-root interval"
        )
        sweep_parser.add_argument(
            "--cluster-tolerance",
            type=float,
            default=1.0e-4,
            help="relative distance used to group roots (default: 1e-4)",
        )
        sweep_parser.add_argument(
            "--minimum-cluster-fraction",
            type=float,
            default=0.5,
            help="accepted-fit fraction required for a recurring cluster (default: 0.5)",
        )
        sweep_parser.add_argument(
            "--include-rank-deficient",
            action="store_true",
            help="retain fits whose scaled linear systems are rank deficient",
        )
        sweep_parser.add_argument(
            "--all-clusters",
            action="store_true",
            help="report singleton and weakly supported clusters too",
        )
        sweep_parser.add_argument(
            "--show-approximants",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="show every selected root in the table output",
        )
        sweep_parser.add_argument(
            "--output", choices=("table", "json", "csv"), default="table"
        )
        sweep_parser.add_argument("--verbose", action="store_true")
        _backend_arguments(sweep_parser)

    sweep_parser = subparsers.add_parser(
        "sweep", help="generate and analyse a balanced family of approximants"
    )
    sweep_parser.add_argument("path", type=Path, help="coefficient file")
    sweep_parser.add_argument(
        "--format", choices=("plain", "legacy"), default="plain"
    )
    add_sweep_arguments(sweep_parser, generated=True)
    sweep_parser.set_defaults(handler=_modern_sweep_command)

    legacy_sweep_parser = subparsers.add_parser(
        "legacy-sweep", help="run the approximant family encoded in a legacy input"
    )
    legacy_sweep_parser.add_argument("path", type=Path)
    add_sweep_arguments(legacy_sweep_parser, generated=False)
    legacy_sweep_parser.set_defaults(
        handler=_legacy_sweep_command,
        orders=(1, 2),
        p_degrees=None,
        degree_spread=2,
        max_terms_omitted=None,
        root_min=0.0,
        root_max=math.inf,
        show_approximants=True,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        return int(arguments.handler(arguments))
    except ApproximantError as error:
        parser.error(str(error))
    return 2
