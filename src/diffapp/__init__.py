"""Tools for constructing and analysing differential approximants."""

from .core import (
    default_specification,
    fit_default_differential_approximant,
    fit_differential_approximant,
)
from .extension import (
    CoefficientForecast,
    ExtensionEstimate,
    ExtensionSweepResult,
    extend_sweep,
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
from .sweep import (
    FitSweepResult,
    RootCluster,
    SweepConfig,
    SweepEstimate,
    SweepFit,
    SweepRejection,
    SweepResult,
    SweepSpecification,
    default_p_degrees,
    fit_sweep,
    generate_sweep_specifications,
    run_sweep,
)

__all__ = [
    "ApproximantError",
    "CoefficientForecast",
    "DifferentialApproximant",
    "FitDiagnostics",
    "FitSweepResult",
    "LegacyDataset",
    "LegacySweep",
    "ExtensionEstimate",
    "ExtensionSweepResult",
    "PolynomialCancellation",
    "RootCluster",
    "SingularSystemError",
    "Singularity",
    "SweepConfig",
    "SweepEstimate",
    "SweepFit",
    "SweepRejection",
    "SweepResult",
    "SweepSpecification",
    "default_specification",
    "default_p_degrees",
    "extend_sweep",
    "fit_default_differential_approximant",
    "fit_differential_approximant",
    "fit_sweep",
    "read_legacy_dataset",
    "read_plain_coefficients",
    "generate_sweep_specifications",
    "run_sweep",
]
