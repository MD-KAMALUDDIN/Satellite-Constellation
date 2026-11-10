from datetime import datetime, timedelta, timezone

import numpy as np

from satellite_optimizer.hybrid.evaluator import (
    compare_results,
    validate_qaoa_selection,
)
from satellite_optimizer.models.observation import ObservationOpportunity
from satellite_optimizer.optimization.reduction import ScoredCandidate


def make_candidate(
    satellite_id: str,
    target_id: str,
    score: float,
    offset_seconds: int,
) -> ScoredCandidate:
    start = datetime(
        2026,
        1,
        1,
        tzinfo=timezone.utc,
    ) + timedelta(seconds=offset_seconds)

    opportunity = ObservationOpportunity(
        satellite_id=satellite_id,
        target_id=target_id,
        start_time=start,
        end_time=start + timedelta(seconds=10),
        elevation_deg=80.0,
        off_nadir_deg=5.0,
        slew_angle_deg=5.0,
        slew_time_sec=2.5,
        image_quality=0.9,
        energy_cost_wh=0.3,
        data_size_mb=500.0,
        feasible=True,
    )

    return ScoredCandidate(
        opportunity=opportunity,
        target_priority=8.0,
        score=score,
    )


def test_qaoa_bitstring_decodes_in_qiskit_order():
    candidates = [
        make_candidate("SAT-01", "TGT-01", 0.9, 0),
        make_candidate("SAT-02", "TGT-02", 0.8, 60),
        make_candidate("SAT-03", "TGT-03", 0.7, 120),
    ]

    validation = validate_qaoa_selection(
        bitstring="001",
        candidates=candidates,
    )

    assert validation.feasible
    assert len(validation.selected_candidates) == 1
    assert validation.selected_candidates[0] == candidates[0]


def test_valid_qaoa_selection_is_accepted():
    candidates = [
        make_candidate("SAT-01", "TGT-01", 0.9, 0),
        make_candidate("SAT-02", "TGT-02", 0.8, 60),
        make_candidate("SAT-03", "TGT-03", 0.7, 120),
    ]

    validation = validate_qaoa_selection(
        bitstring="101",
        candidates=candidates,
    )

    assert validation.feasible
    assert validation.violations == []
    assert len(validation.selected_candidates) == 2
    assert np.isclose(validation.objective_value, 1.6)


def test_duplicate_target_is_rejected():
    candidates = [
        make_candidate("SAT-01", "TGT-01", 0.9, 0),
        make_candidate("SAT-02", "TGT-01", 0.8, 60),
        make_candidate("SAT-03", "TGT-02", 0.7, 120),
    ]

    validation = validate_qaoa_selection(
        bitstring="011",
        candidates=candidates,
    )

    assert not validation.feasible
    assert validation.objective_value == 0.0
    assert any(
        "Multiple observations selected for target TGT-01"
        in violation
        for violation in validation.violations
    )


def test_overlapping_same_satellite_is_rejected():
    candidates = [
        make_candidate("SAT-01", "TGT-01", 0.9, 0),
        make_candidate("SAT-01", "TGT-02", 0.8, 5),
        make_candidate("SAT-03", "TGT-03", 0.7, 120),
    ]

    validation = validate_qaoa_selection(
        bitstring="011",
        candidates=candidates,
    )

    assert not validation.feasible
    assert any(
        "Overlapping observations selected for satellite SAT-01"
        in violation
        for violation in validation.violations
    )


def test_same_solution_is_detected():
    candidates = [
        make_candidate("SAT-01", "TGT-01", 0.9, 0),
        make_candidate("SAT-02", "TGT-02", 0.8, 60),
        make_candidate("SAT-03", "TGT-03", 0.7, 120),
    ]

    validation = validate_qaoa_selection(
        bitstring="001",
        candidates=candidates,
    )

    comparison = compare_results(
        classical_candidates=[candidates[0]],
        qaoa_validation=validation,
    )

    assert comparison.same_solution
    assert np.isclose(comparison.classical_score, 0.9)
    assert np.isclose(comparison.qaoa_score, 0.9)
    assert np.isclose(comparison.score_gap, 0.0)
    assert np.isclose(comparison.qaoa_relative_gap_percent, 0.0)


def test_lower_qaoa_score_is_reported():
    candidates = [
        make_candidate("SAT-01", "TGT-01", 0.9, 0),
        make_candidate("SAT-02", "TGT-02", 0.8, 60),
        make_candidate("SAT-03", "TGT-03", 0.7, 120),
    ]

    validation = validate_qaoa_selection(
        bitstring="100",
        candidates=candidates,
    )

    comparison = compare_results(
        classical_candidates=[candidates[0], candidates[1]],
        qaoa_validation=validation,
    )

    assert not comparison.same_solution
    assert np.isclose(comparison.classical_score, 1.7)
    assert np.isclose(comparison.qaoa_score, 0.7)
    assert np.isclose(comparison.score_gap, 1.0)
    assert np.isclose(
        comparison.qaoa_relative_gap_percent,
        100.0 / 1.7,
    )