from dataclasses import dataclass
from datetime import datetime

from satellite_optimizer.models.mission import Mission
from satellite_optimizer.models.observation import ObservationOpportunity
from satellite_optimizer.models.target import Target


@dataclass(frozen=True)
class ScoredCandidate:
    """An observation opportunity with its mission score."""

    opportunity: ObservationOpportunity
    target_priority: float
    score: float


def _normalize(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    """Normalize a value into the range [0, 1]."""

    if maximum <= minimum:
        return 1.0

    normalized = (
        (value - minimum)
        / (maximum - minimum)
    )

    return max(
        0.0,
        min(1.0, normalized),
    )


def _deadline_value(
    opportunity: ObservationOpportunity,
    target: Target,
    mission_start: datetime,
) -> float:
    """
    Give higher value to observations scheduled earlier
    relative to the target deadline.
    """

    elapsed_hours = (
        opportunity.start_time - mission_start
    ).total_seconds() / 3600.0

    if target.deadline_hours <= 0.0:
        return 1.0

    value = (
        1.0
        - elapsed_hours / target.deadline_hours
    )

    return max(
        0.0,
        min(1.0, value),
    )


def score_opportunity(
    opportunity: ObservationOpportunity,
    target: Target,
    mission: Mission,
    mission_start: datetime,
) -> float:
    """
    Calculate the mission value of one observation opportunity.

    Higher score means a more desirable candidate.
    """

    # Mission priority.
    priority_score = _normalize(
        target.priority,
        minimum=0.0,
        maximum=10.0,
    )

    # Already normalized by the opportunity generator.
    image_quality_score = max(
        0.0,
        min(1.0, opportunity.image_quality),
    )

    # Earlier opportunities are generally more valuable.
    future_value_score = _deadline_value(
        opportunity=opportunity,
        target=target,
        mission_start=mission_start,
    )

    # Lower energy consumption is better.
    energy_score = 1.0 / (
        1.0 + opportunity.energy_cost_wh
    )

    # Lower slew time is better.
    slew_score = 1.0 / (
        1.0 + opportunity.slew_time_sec
    )

    # Smaller data volume is slightly preferable.
    data_score = 1.0 / (
        1.0 + opportunity.data_size_mb / 1000.0
    )

    weights = mission.scoring

    score = (
        weights.task_value_weight
        * priority_score
        + weights.image_quality_weight
        * image_quality_score
        + weights.future_value_weight
        * future_value_score
        + weights.energy_weight
        * energy_score
        + weights.slew_weight
        * slew_score
        + weights.data_weight
        * data_score
    )

    return score


def score_opportunities(
    opportunities: list[ObservationOpportunity],
    targets: list[Target],
    mission: Mission,
    mission_start: datetime,
) -> list[ScoredCandidate]:
    """Score all feasible observation opportunities."""

    targets_by_id = {
        target.id: target
        for target in targets
    }

    scored_candidates: list[ScoredCandidate] = []

    for opportunity in opportunities:
        if not opportunity.feasible:
            continue

        target = targets_by_id.get(
            opportunity.target_id
        )

        if target is None:
            raise ValueError(
                "Unknown target ID: "
                f"{opportunity.target_id}"
            )

        score = score_opportunity(
            opportunity=opportunity,
            target=target,
            mission=mission,
            mission_start=mission_start,
        )

        scored_candidates.append(
            ScoredCandidate(
                opportunity=opportunity,
                target_priority=target.priority,
                score=score,
            )
        )

    return scored_candidates


def _deduplicate_candidates(
    scored_candidates: list[ScoredCandidate],
) -> list[ScoredCandidate]:
    """
    Keep only the best-scoring candidate per (satellite, target) pair.

    The opportunity scanner fires every time_step_seconds, producing
    many overlapping windows for the same satellite-target pair within
    a single pass.  Feeding all of them into the QUBO creates dense
    conflict graphs where every pair of same-target candidates is
    penalised, making it nearly impossible for QAOA to find a feasible
    solution.  Keeping only the peak-score window per pair removes
    those structural conflicts before the QUBO is built.
    """
    best: dict[tuple[str, str], ScoredCandidate] = {}

    for candidate in scored_candidates:
        key = (
            candidate.opportunity.satellite_id,
            candidate.opportunity.target_id,
        )
        if key not in best or candidate.score > best[key].score:
            best[key] = candidate

    return list(best.values())


def reduce_candidates(
    opportunities: list[ObservationOpportunity],
    targets: list[Target],
    mission: Mission,
    mission_start: datetime,
) -> list[ScoredCandidate]:
    """
    Reduce raw opportunities to the highest-value candidates.

    The maximum number is controlled by:
        mission.quantum.max_candidates
    """

    scored_candidates = score_opportunities(
        opportunities=opportunities,
        targets=targets,
        mission=mission,
        mission_start=mission_start,
    )

    # One best window per (satellite, target) pair — eliminates the
    # dense same-target conflict clusters that break QAOA feasibility.
    scored_candidates = _deduplicate_candidates(scored_candidates)

    scored_candidates.sort(
        key=lambda candidate: candidate.score,
        reverse=True,
    )

    max_candidates = mission.quantum.max_candidates

    return scored_candidates[:max_candidates]