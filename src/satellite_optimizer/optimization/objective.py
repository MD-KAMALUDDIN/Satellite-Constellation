from dataclasses import dataclass

from satellite_optimizer.optimization.reduction import (
    ScoredCandidate,
)


@dataclass(frozen=True)
class OptimizationResult:
    """Result of a classical optimization run."""

    selected_candidates: list[ScoredCandidate]
    objective_value: float
    solver_status: str


def calculate_objective_value(
    candidates: list[ScoredCandidate],
) -> float:
    """Calculate the total mission score."""

    return sum(
        candidate.score
        for candidate in candidates
    )