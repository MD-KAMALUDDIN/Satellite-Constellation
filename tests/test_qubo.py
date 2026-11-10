from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from satellite_optimizer.models.observation import (
    ObservationOpportunity,
)
from satellite_optimizer.optimization.qubo import (
    build_qubo,
    evaluate_qubo,
)
from satellite_optimizer.optimization.reduction import (
    ScoredCandidate,
)


def make_candidate(
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
        score=score,
    )


def test_qubo_dimensions():
    candidates = [
        make_candidate(
            "SAT-01",
            "TGT-01",
            0.9,
            0,
        ),
        make_candidate(
            "SAT-02",
            "TGT-02",
            0.8,
            0,
        ),
        make_candidate(
            "SAT-03",
            "TGT-03",
            0.7,
            0,
        ),
    ]

    problem = build_qubo(candidates)

    assert problem.num_variables == 3
    assert problem.num_qubits == 3
    assert problem.matrix.shape == (3, 3)


def test_qubo_diagonal_contains_negative_scores():
    candidates = [
        make_candidate(
            "SAT-01",
            "TGT-01",
            0.9,
            0,
        ),
        make_candidate(
            "SAT-02",
            "TGT-02",
            0.8,
            0,
        ),
    ]

    problem = build_qubo(candidates)

    assert problem.matrix[0, 0] == pytest.approx(-0.9)
    assert problem.matrix[1, 1] == pytest.approx(-0.8)


def test_same_target_creates_conflict():
    candidates = [
        make_candidate(
            "SAT-01",
            "TGT-01",
            0.9,
            0,
        ),
        make_candidate(
            "SAT-02",
            "TGT-01",
            0.8,
            60,
        ),
    ]

    problem = build_qubo(
        candidates,
        conflict_penalty=2.0,
    )

    assert problem.matrix[0, 1] == pytest.approx(2.0)
    assert problem.matrix[1, 0] == pytest.approx(2.0)


def test_overlapping_same_satellite_creates_conflict():
    candidates = [
        make_candidate(
            "SAT-01",
            "TGT-01",
            0.9,
            0,
            duration_seconds=30,
        ),
        make_candidate(
            "SAT-01",
            "TGT-02",
            0.8,
            10,
            duration_seconds=30,
        ),
    ]

    problem = build_qubo(
        candidates,
        conflict_penalty=2.0,
    )

    assert problem.matrix[0, 1] == pytest.approx(2.0)
    assert problem.matrix[1, 0] == pytest.approx(2.0)


def test_non_conflicting_candidates_have_zero_off_diagonal():
    candidates = [
        make_candidate(
            "SAT-01",
            "TGT-01",
            0.9,
            0,
        ),
        make_candidate(
            "SAT-02",
            "TGT-02",
            0.8,
            0,
        ),
    ]

    problem = build_qubo(
        candidates,
        conflict_penalty=2.0,
    )

    assert problem.matrix[0, 1] == pytest.approx(0.0)
    assert problem.matrix[1, 0] == pytest.approx(0.0)


def test_evaluate_empty_selection():
    candidates = [
        make_candidate(
            "SAT-01",
            "TGT-01",
            0.9,
            0,
        ),
    ]

    problem = build_qubo(candidates)

    value = evaluate_qubo(
        problem,
        np.array([0]),
    )

    assert value == pytest.approx(0.0)


def test_evaluate_single_selection():
    candidates = [
        make_candidate(
            "SAT-01",
            "TGT-01",
            0.9,
            0,
        ),
    ]

    problem = build_qubo(candidates)

    value = evaluate_qubo(
        problem,
        np.array([1]),
    )

    assert value == pytest.approx(-0.9)


def test_conflicting_selection_is_penalized():
    candidates = [
        make_candidate(
            "SAT-01",
            "TGT-01",
            0.9,
            0,
        ),
        make_candidate(
            "SAT-02",
            "TGT-01",
            0.8,
            60,
        ),
    ]

    problem = build_qubo(
        candidates,
        conflict_penalty=2.0,
    )

    valid_value = evaluate_qubo(
        problem,
        np.array([1, 0]),
    )

    invalid_value = evaluate_qubo(
        problem,
        np.array([1, 1]),
    )

    assert valid_value == pytest.approx(-0.9)

    # The symmetric QUBO matrix stores the conflict
    # penalty in both Q[0,1] and Q[1,0].
    expected_invalid_value = (
        -0.9
        - 0.8
        + 2.0
        + 2.0
    )

    assert invalid_value == pytest.approx(
        expected_invalid_value
    )

    assert invalid_value > valid_value 


def test_wrong_bitstring_length_raises():
    candidates = [
        make_candidate(
            "SAT-01",
            "TGT-01",
            0.9,
            0,
        ),
    ]

    problem = build_qubo(candidates)

    with pytest.raises(ValueError):
        evaluate_qubo(
            problem,
            np.array([1, 0]),
        )


def test_non_binary_bitstring_raises():
    candidates = [
        make_candidate(
            "SAT-01",
            "TGT-01",
            0.9,
            0,
        ),
    ]

    problem = build_qubo(candidates)

    with pytest.raises(ValueError):
        evaluate_qubo(
            problem,
            np.array([2]),
        )