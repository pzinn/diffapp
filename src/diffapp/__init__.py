"""Tools for constructing and analysing differential approximants."""

from .core import (
    default_specification,
    fit_default_differential_approximant,
    fit_differential_approximant,
)
from .io import LegacyDataset, LegacySweep, read_legacy_dataset, read_plain_coefficients
from .model import (
    ApproximantError,
    DifferentialApproximant,
    FitDiagnostics,
    PolynomialCancellation,
    SingularSystemError,
    Singularity,
)

__all__ = [
    "ApproximantError",
    "DifferentialApproximant",
    "FitDiagnostics",
    "LegacyDataset",
    "LegacySweep",
    "PolynomialCancellation",
    "SingularSystemError",
    "Singularity",
    "default_specification",
    "fit_default_differential_approximant",
    "fit_differential_approximant",
    "read_legacy_dataset",
    "read_plain_coefficients",
]
