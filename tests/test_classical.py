from datetime import datetime, timedelta, timezone

from satellite_optimizer.models.observation import (
    ObservationOpportunity,
)
from satellite_optimizer.optimization.classical import (
    solve_classical,
)
from satellite_optimizer.optimization.reduction import (
    ScoredCandidate,
)


def make_candidate(
    candidate_id: int,
    satellite_id: str,
    target_id: str,
    score: float,
    start_seconds: int,
    duration_seconds: int = 10,
) -> ScoredCandidate:
    mission_start = datetime(
        2026,
        1,
        1,
        tzinfo=timezone.utc,
    )

    start_time = (
        mission_start
        + timedelta(seconds=start_seconds)
    )

    end_time = (
        start_time
        + timedelta(seconds=duration_seconds)
    )

    opportunity = ObservationOpportunity(
        satellite_id=satellite_id,
        target_id=target_id,
        start_time=start_time,
        end_time=end_time,
        elevation_deg=70.0,
        off_nadir_deg=10.0,
        slew_angle_deg=10.0,
        slew_time_sec=5.0,
        image_quality=0.8,
        energy_cost_wh=0.3,
        data_size_mb=500.0,
        feasible=True,
    )

    return ScoredCandidate(
        opportunity=opportunity,
        target_priority=10.0,
        score=score + candidate_id * 0.000001,
    )


def test_classical_solver_selects_candidates():
    candidates = [
        make_candidate(
            candidate_id=0,
            satellite_id="SAT-01",
            target_id="TGT-01",
            score=0.9,
            start_seconds=0,
        ),
        make_candidate(
            candidate_id=1,
            satellite_id="SAT-02",
            target_id="TGT-02",
            score=0.8,
            start_seconds=0,
        ),
    ]

    result = solve_classical(candidates)

    assert result.solver_status in {
        "OPTIMAL",
        "FEASIBLE",
    }

    assert len(result.selected_candidates) == 2


def test_at_most_one_candidate_per_target():
    candidates = [
        make_candidate(
            candidate_id=0,
            satellite_id="SAT-01",
            target_id="TGT-01",
            score=0.9,
            start_seconds=0,
        ),
        make_candidate(
            candidate_id=1,
            satellite_id="SAT-02",
            target_id="TGT-01",
            score=0.8,
            start_seconds=60,
        ),
    ]

    result = solve_classical(candidates)

    assert len(result.selected_candidates) == 1
    assert (
        result.selected_candidates[0]
        .opportunity
        .satellite_id
        == "SAT-01"
    )


def test_overlapping_same_satellite_candidates_conflict():
    candidates = [
        make_candidate(
            candidate_id=0,
            satellite_id="SAT-01",
            target_id="TGT-01",
            score=0.9,
            start_seconds=0,
            duration_seconds=30,
        ),
        make_candidate(
            candidate_id=1,
            satellite_id="SAT-01",
            target_id="TGT-02",
            score=0.8,
            start_seconds=10,
            duration_seconds=30,
        ),
    ]

    result = solve_classical(candidates)

    assert len(result.selected_candidates) == 1


def test_non_overlapping_same_satellite_candidates_can_both_be_selected():
    candidates = [
        make_candidate(
            candidate_id=0,
            satellite_id="SAT-01",
            target_id="TGT-01",
            score=0.9,
            start_seconds=0,
            duration_seconds=10,
        ),
        make_candidate(
            candidate_id=1,
            satellite_id="SAT-01",
            target_id="TGT-02",
            score=0.8,
            start_seconds=60,
            duration_seconds=10,
        ),
    ]

    result = solve_classical(candidates)

    assert len(result.selected_candidates) == 2


def test_infeasible_candidate_is_not_selected():
    candidate = make_candidate(
        candidate_id=0,
        satellite_id="SAT-01",
        target_id="TGT-01",
        score=0.9,
        start_seconds=0,
    )

    infeasible_opportunity = ObservationOpportunity(
        satellite_id=candidate.opportunity.satellite_id,
        target_id=candidate.opportunity.target_id,
        start_time=candidate.opportunity.start_time,
        end_time=candidate.opportunity.end_time,
        elevation_deg=candidate.opportunity.elevation_deg,
        off_nadir_deg=candidate.opportunity.off_nadir_deg,
        slew_angle_deg=candidate.opportunity.slew_angle_deg,
        slew_time_sec=candidate.opportunity.slew_time_sec,
        image_quality=candidate.opportunity.image_quality,
        energy_cost_wh=candidate.opportunity.energy_cost_wh,
        data_size_mb=candidate.opportunity.data_size_mb,
        feasible=False,
    )

    infeasible_candidate = ScoredCandidate(
        opportunity=infeasible_opportunity,
        target_priority=candidate.target_priority,
        score=candidate.score,
    )

    result = solve_classical(
        [infeasible_candidate]
    )

    assert len(result.selected_candidates) == 0