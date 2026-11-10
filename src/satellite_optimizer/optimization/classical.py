from ortools.sat.python import cp_model

from satellite_optimizer.optimization.objective import (
    OptimizationResult,
)
from satellite_optimizer.optimization.reduction import (
    ScoredCandidate,
)


_SCORE_SCALE = 1_000_000


def _scores_as_integers(
    candidates: list[ScoredCandidate],
) -> list[int]:
    """
    Convert floating-point scores into integer coefficients.

    CP-SAT works naturally with integer objective coefficients.
    """

    return [
        round(candidate.score * _SCORE_SCALE)
        for candidate in candidates
    ]


def _overlap(
    first: ScoredCandidate,
    second: ScoredCandidate,
) -> bool:
    """Return True if two candidates overlap in time."""

    first_opportunity = first.opportunity
    second_opportunity = second.opportunity

    return (
        first_opportunity.start_time
        < second_opportunity.end_time
        and second_opportunity.start_time
        < first_opportunity.end_time
    )


def solve_classical(
    candidates: list[ScoredCandidate],
    time_limit_seconds: float = 30.0,
) -> OptimizationResult:
    """
    Solve the reduced scheduling problem using CP-SAT.

    Constraints:
    - infeasible candidates cannot be selected
    - at most one candidate per target
    - overlapping observations on the same satellite
      cannot both be selected
    """

    model = cp_model.CpModel()

    variables = [
        model.NewBoolVar(f"x_{index}")
        for index in range(len(candidates))
    ]

    # Explicitly prevent infeasible candidates.
    for index, candidate in enumerate(candidates):
        if not candidate.opportunity.feasible:
            model.Add(
                variables[index] == 0
            )

    # At most one selected observation per target.
    target_to_indices: dict[str, list[int]] = {}

    for index, candidate in enumerate(candidates):
        target_id = candidate.opportunity.target_id

        target_to_indices.setdefault(
            target_id,
            [],
        ).append(index)

    for indices in target_to_indices.values():
        model.AddAtMostOne(
            variables[index]
            for index in indices
        )

    # Prevent overlapping observations on the same satellite.
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            first = candidates[i]
            second = candidates[j]

            if (
                first.opportunity.satellite_id
                != second.opportunity.satellite_id
            ):
                continue

            if _overlap(first, second):
                model.Add(
                    variables[i]
                    + variables[j]
                    <= 1
                )

    integer_scores = _scores_as_integers(
        candidates
    )

    model.Maximize(
        sum(
            integer_scores[index]
            * variables[index]
            for index in range(len(candidates))
        )
    )

    solver = cp_model.CpSolver()

    solver.parameters.max_time_in_seconds = (
        time_limit_seconds
    )

    status = solver.Solve(model)

    status_name = solver.StatusName(status)

    if status not in (
        cp_model.OPTIMAL,
        cp_model.FEASIBLE,
    ):
        return OptimizationResult(
            selected_candidates=[],
            objective_value=0.0,
            solver_status=status_name,
        )

    selected_candidates = [
        candidates[index]
        for index in range(len(candidates))
        if solver.Value(variables[index]) == 1
    ]

    objective_value = sum(
        candidates[index].score
        for index in range(len(candidates))
        if solver.Value(variables[index]) == 1
    )

    return OptimizationResult(
        selected_candidates=selected_candidates,
        objective_value=objective_value,
        solver_status=status_name,
    )