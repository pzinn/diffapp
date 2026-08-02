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
from .sweep import (
    RootCluster,
    SweepConfig,
    SweepEstimate,
    SweepRejection,
    SweepResult,
    SweepSpecification,
    default_p_degrees,
    generate_sweep_specifications,
    run_sweep,
)

__all__ = [
    "ApproximantError",
    "DifferentialApproximant",
    "FitDiagnostics",
    "LegacyDataset",
    "LegacySweep",
    "PolynomialCancellation",
    "RootCluster",
    "SingularSystemError",
    "Singularity",
    "SweepConfig",
    "SweepEstimate",
    "SweepRejection",
    "SweepResult",
    "SweepSpecification",
    "default_specification",
    "default_p_degrees",
    "fit_default_differential_approximant",
    "fit_differential_approximant",
    "read_legacy_dataset",
    "read_plain_coefficients",
    "generate_sweep_specifications",
    "run_sweep",
]
