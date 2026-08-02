"""Command-line interface for differential-approximant analysis."""

from __future__ import annotations

import argparse
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


def _sweep_command(arguments: argparse.Namespace) -> int:
    dataset = read_legacy_dataset(arguments.path)
    produced = 0
    rejected = 0
    for q_degrees, p_degree in dataset.sweep.specifications():
        needed = required_coefficients(q_degrees, p_degree)
        if needed > len(dataset.coefficients):
            continue
        try:
            approximant = fit_differential_approximant(
                dataset.coefficients,
                q_degrees,
                p_degree,
                backend=arguments.backend,
                precision_digits=arguments.precision,
            )
            physical = approximant.physical_singularity(
                arguments.imaginary_tolerance,
                (arguments.root_min, arguments.root_max),
            )
        except ApproximantError as error:
            rejected += 1
            if arguments.verbose:
                print(f"rejected Q={q_degrees}, P={p_degree}: {error}", file=sys.stderr)
            continue
        if physical is None:
            rejected += 1
            continue
        exponent = (
            "?"
            if physical.exponent is None
            else _format_number(physical.exponent, arguments.digits)
        )
        condition = approximant.diagnostics.scaled_condition_number
        condition_text = (
            "n/a"
            if condition is None or not math.isfinite(condition)
            else f"{condition:.2e}"
        )
        cancellation_text = (
            f"{physical.common_factor_residual:.1e}"
            if physical.common_factor_residual is not None
            else "n/a"
        )
        print(
            f"Q={q_degrees} P={p_degree:2d} used={needed:2d} "
            f"x={_format_number(physical.root, arguments.digits):>16} "
            f"theta={exponent:>16} cond={condition_text} "
            f"cancel={cancellation_text}"
        )
        produced += 1
    print(f"{produced} approximants reported; {rejected} rejected", file=sys.stderr)
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

    sweep_parser = subparsers.add_parser(
        "legacy-sweep", help="run the approximant family encoded in a legacy input"
    )
    sweep_parser.add_argument("path", type=Path)
    sweep_parser.add_argument("--imaginary-tolerance", type=float, default=1.0e-8)
    sweep_parser.add_argument(
        "--root-min",
        type=float,
        default=0.0,
        help="lower end of the physical-root interval",
    )
    sweep_parser.add_argument(
        "--root-max",
        type=float,
        default=math.inf,
        help="upper end of the physical-root interval",
    )
    sweep_parser.add_argument("--verbose", action="store_true")
    _backend_arguments(sweep_parser)
    sweep_parser.set_defaults(handler=_sweep_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        return int(arguments.handler(arguments))
    except ApproximantError as error:
        parser.error(str(error))
    return 2
