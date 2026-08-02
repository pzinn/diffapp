"""Input formats, including the compact files used with NEWGRQD."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .model import ApproximantError


_INTEGER_SPLIT = re.compile(r"[\s,]+")


@dataclass(frozen=True)
class LegacySweep:
    minimum_order: int
    maximum_order: int
    minimum_degree: int
    maximum_degree: int
    minimum_q1_offset: int
    maximum_q1_offset: int
    minimum_q2_offset: int
    maximum_q2_offset: int
    minimum_p_degree: int
    maximum_p_degree: int
    minimum_terms: int = 0
    maximum_terms: int = 0

    @classmethod
    def from_values(cls, values: list[int]) -> "LegacySweep":
        padded = values + [0] * (12 - len(values))
        if len(values) < 10:
            raise ApproximantError(
                "legacy sweep line must contain at least ten integers"
            )
        return cls(*padded[:12])

    def specifications(self) -> tuple[tuple[tuple[int, ...], int], ...]:
        """Expand the sweep into ``(Q degrees, P degree)`` specifications."""

        result: list[tuple[tuple[int, ...], int]] = []
        for order in range(self.minimum_order, self.maximum_order + 1):
            for p_degree in range(self.minimum_p_degree, self.maximum_p_degree + 1):
                for degree in range(self.minimum_degree, self.maximum_degree + 1):
                    for delta in (-1, 0, 1):
                        q_degrees = [degree] * (order + 1)
                        if order >= 1:
                            q_degrees[1] = degree + delta
                        if order >= 2:
                            q_degrees[2] = degree + delta
                        if min(q_degrees) >= 0:
                            result.append((tuple(q_degrees), p_degree))
        return tuple(result)


@dataclass(frozen=True)
class LegacyDataset:
    output_flags: tuple[int, int, int]
    highest_series_order: int
    coefficients: tuple[Decimal, ...]
    sweep: LegacySweep
    terminator: int | None = None


def _integers(line: str) -> list[int]:
    try:
        return [int(value) for value in _INTEGER_SPLIT.split(line.strip()) if value]
    except ValueError as error:
        raise ApproximantError(f"invalid integer line in legacy input: {line!r}") from error


def read_legacy_dataset(path: str | Path) -> LegacyDataset:
    """Read the practical, one-coefficient-per-line NEWGRQD input variant."""

    source = Path(path)
    lines = [line.strip() for line in source.read_text().replace("\x00", "").splitlines()]
    lines = [line for line in lines if line]
    if len(lines) < 4:
        raise ApproximantError(f"legacy input {source} is incomplete")

    flags = _integers(lines[0])
    if len(flags) != 3:
        raise ApproximantError("legacy output flag line must contain three integers")
    try:
        highest_order = int(lines[1])
    except ValueError as error:
        raise ApproximantError("legacy series order is not an integer") from error
    coefficient_count = highest_order + 1
    if len(lines) < 3 + coefficient_count:
        raise ApproximantError(
            f"legacy input declares {coefficient_count} coefficients but is truncated"
        )
    try:
        coefficients = tuple(
            Decimal(lines[index]) for index in range(2, 2 + coefficient_count)
        )
    except InvalidOperation as error:
        raise ApproximantError("invalid series coefficient in legacy input") from error

    sweep_index = 2 + coefficient_count
    sweep = LegacySweep.from_values(_integers(lines[sweep_index]))
    terminator = None
    if len(lines) > sweep_index + 1:
        values = _integers(lines[sweep_index + 1])
        terminator = values[0] if values else None
    return LegacyDataset(
        output_flags=(flags[0], flags[1], flags[2]),
        highest_series_order=highest_order,
        coefficients=coefficients,
        sweep=sweep,
        terminator=terminator,
    )


def read_plain_coefficients(path: str | Path) -> tuple[Decimal, ...]:
    """Read whitespace- or comma-separated decimal coefficients.

    Blank lines and text following ``#`` are ignored.  Values are returned as
    ``Decimal`` objects so no precision is discarded by the input layer.
    """

    source = Path(path)
    tokens: list[str] = []
    for original_line in source.read_text().replace("\x00", "").splitlines():
        line = original_line.partition("#")[0].strip()
        if line:
            tokens.extend(value for value in _INTEGER_SPLIT.split(line) if value)
    if not tokens:
        raise ApproximantError(f"coefficient file {source} is empty")
    try:
        return tuple(Decimal(value) for value in tokens)
    except InvalidOperation as error:
        raise ApproximantError(f"invalid coefficient in {source}") from error
