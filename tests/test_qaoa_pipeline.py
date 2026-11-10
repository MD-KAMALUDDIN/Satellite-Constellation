"""
Tests for the QAOA evaluation pipeline.

Covers:
  - slack variables are created when satellites are provided
  - total QUBO variables > candidate variables when slack is required
  - variable metadata correctly identifies observation/slack variables
  - QUBO energy matches the mathematical formulation
  - feasible energy allocation passes validation
  - energy-over-budget allocation fails validation
  - QAOA ignores infeasible lowest-energy samples
  - QAOA selects the best feasible sampled state
  - zero feasible samples produces None score
  - Qiskit bitstring ordering is decoded correctly
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from satellite_optimizer.models.observation import ObservationOpportunity
from satellite_optimizer.models.satellite import Satellite
from satellite_optimizer.optimization.qubo import (
    QUBOProblem,
    VariableMeta,
    build_qubo,
    evaluate_qubo,
)
from satellite_optimizer.optimization.reduction import ScoredCandidate
from satellite_optimizer.hybrid.evaluator import (
    ValidationResult,
    _decode_qaoa_bitstring,
    validate_qaoa_selection,
    _best_feasible_validation,
)
from satellite_optimizer.quantum.qaoa import QAOAResult, ranked_samples
from satellite_optimizer.quantum.circuit import QAOACircuit


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _make_candidate(
    sat_id: str,
    tgt_id: str,
    score: float,
    offset_s: int = 0,
    duration_s: int = 60,
    energy_wh: float = 0.5,
) -> ScoredCandidate:
    start = _T0 + timedelta(seconds=offset_s)
    opp = ObservationOpportunity(
        satellite_id=sat_id,
        target_id=tgt_id,
        start_time=start,
        end_time=start + timedelta(seconds=duration_s),
        elevation_deg=70.0,
        off_nadir_deg=10.0,
        slew_angle_deg=10.0,
        slew_time_sec=5.0,
        image_quality=0.8,
        energy_cost_wh=energy_wh,
        data_size_mb=500.0,
        feasible=True,
    )
    return ScoredCandidate(opportunity=opp, target_priority=8.0, score=score)


def _make_satellite(
    sat_id: str,
    capacity_wh: float = 1000.0,
    initial_wh: float = 850.0,
) -> Satellite:
    return Satellite(
        id=sat_id,
        name=sat_id,
        altitude_km=550.0,
        inclination_deg=53.0,
        raan_deg=0.0,
        phase_deg=0.0,
        battery_capacity_wh=capacity_wh,
        initial_battery_wh=initial_wh,
        storage_capacity_gb=256.0,
        initial_storage_gb=40.0,
        slew_rate_deg_per_sec=2.0,
    )


def _make_fake_qaoa_result(
    counts: dict[str, int],
    qubo: QUBOProblem,
) -> QAOAResult:
    """Build a minimal QAOAResult with controlled counts for testing."""
    n = qubo.num_variables
    raw_bs = min(counts, key=lambda bs: float("inf"))  # just pick first
    # pick lowest energy
    best_bs = min(
        counts,
        key=lambda bs: float(
            np.asarray([int(b) for b in reversed(bs)], dtype=float)
            @ qubo.matrix
            @ np.asarray([int(b) for b in reversed(bs)], dtype=float)
        ),
    )
    raw_energy = float(
        np.asarray([int(b) for b in reversed(best_bs)], dtype=float)
        @ qubo.matrix
        @ np.asarray([int(b) for b in reversed(best_bs)], dtype=float)
    )
    dummy_circuit = QAOACircuit(
        linear=np.zeros(n),
        quadratic=np.zeros((n, n)),
        depth=1,
        num_qubits=n,
    )
    return QAOAResult(
        best_parameters=np.zeros(2),
        best_energy=raw_energy,
        counts=counts,
        best_bitstring=best_bs,
        raw_best_bitstring=best_bs,
        raw_best_energy=raw_energy,
        feasible_sample_count=0,
        circuit=dummy_circuit,
        circuit_depth=1,
        gate_count=1,
        two_qubit_gate_count=0,
        shots=sum(counts.values()),
        backend_name="test",
    )


# ------------------------------------------------------------------ #
# 1. Slack variables are created when satellites are provided
# ------------------------------------------------------------------ #

def test_slack_variables_created_with_satellites():
    candidates = [
        _make_candidate("SAT-01", "TGT-01", 0.9, energy_wh=1.0),
        _make_candidate("SAT-01", "TGT-02", 0.8, offset_s=120, energy_wh=1.0),
    ]
    sats = [_make_satellite("SAT-01", capacity_wh=100.0, initial_wh=50.0)]
    qubo = build_qubo(candidates, satellites=sats)
    assert qubo.num_slack_variables > 0


# ------------------------------------------------------------------ #
# 2. Total QUBO variables > candidate variables when slack is required
# ------------------------------------------------------------------ #

def test_total_variables_greater_than_candidates_with_slack():
    candidates = [_make_candidate("SAT-01", "TGT-01", 0.9, energy_wh=2.0)]
    sats = [_make_satellite("SAT-01", capacity_wh=100.0, initial_wh=50.0)]
    qubo = build_qubo(candidates, satellites=sats)
    assert qubo.num_variables > qubo.num_observation_variables


# ------------------------------------------------------------------ #
# 3. Variable metadata correctly identifies observation/slack variables
# ------------------------------------------------------------------ #

def test_variable_metadata_kinds():
    candidates = [
        _make_candidate("SAT-01", "TGT-01", 0.9),
        _make_candidate("SAT-02", "TGT-02", 0.8),
    ]
    sats = [
        _make_satellite("SAT-01", capacity_wh=100.0, initial_wh=50.0),
        _make_satellite("SAT-02", capacity_wh=100.0, initial_wh=50.0),
    ]
    qubo = build_qubo(candidates, satellites=sats)

    obs = [v for v in qubo.variable_meta if v.kind == "observation"]
    slack = [v for v in qubo.variable_meta if v.kind == "slack"]

    assert len(obs) == 2
    assert len(slack) > 0
    for v in obs:
        assert v.candidate_index is not None
    for v in slack:
        assert v.satellite_id is not None
        assert v.slack_bit is not None


# ------------------------------------------------------------------ #
# 4. QUBO energy matches the mathematical formulation
# ------------------------------------------------------------------ #

def test_qubo_energy_no_conflict_no_slack():
    """x^T Q x for a single selected candidate = -score."""
    candidates = [_make_candidate("SAT-01", "TGT-01", 0.75)]
    qubo = build_qubo(candidates)  # no satellites → no slack
    energy = evaluate_qubo(qubo, [1])
    assert energy == pytest.approx(-0.75)


def test_qubo_energy_conflict_penalised():
    candidates = [
        _make_candidate("SAT-01", "TGT-01", 0.9),
        _make_candidate("SAT-02", "TGT-01", 0.8),  # same target → conflict
    ]
    qubo = build_qubo(candidates, conflict_penalty=5.0)
    # selecting both: -0.9 - 0.8 + 5 + 5 = 8.3
    energy_both = evaluate_qubo(qubo, [1, 1])
    energy_one = evaluate_qubo(qubo, [1, 0])
    assert energy_both > energy_one


# ------------------------------------------------------------------ #
# 5. Feasible energy allocation passes validation
# ------------------------------------------------------------------ #

def test_energy_within_budget_passes():
    candidates = [_make_candidate("SAT-01", "TGT-01", 0.9, energy_wh=1.0)]
    sats = [_make_satellite("SAT-01", capacity_wh=100.0, initial_wh=50.0)]
    # budget = 50 - 0.20*100 = 30 Wh; used = 1.0 Wh → OK
    qubo = build_qubo(candidates, satellites=sats)
    # bitstring covers only observation qubits (num_qubits)
    # "1" reversed = qubit 0 selected = candidate 0
    bs = "1"
    result = validate_qaoa_selection(bs, qubo, sats)
    assert result.feasible
    assert result.objective_value == pytest.approx(0.9)


# ------------------------------------------------------------------ #
# 6. Energy over budget fails validation
# ------------------------------------------------------------------ #

def test_energy_over_budget_fails():
    # budget = 50 - 0.20*100 = 30 Wh; two obs each 20 Wh → 40 Wh > 30
    candidates = [
        _make_candidate("SAT-01", "TGT-01", 0.9, offset_s=0, energy_wh=20.0),
        _make_candidate("SAT-01", "TGT-02", 0.8, offset_s=300, energy_wh=20.0),
    ]
    sats = [_make_satellite("SAT-01", capacity_wh=100.0, initial_wh=50.0)]
    qubo = build_qubo(candidates, satellites=sats)
    # select both observation bits: "11" reversed → qubit0=1, qubit1=1
    bs = "11"
    result = validate_qaoa_selection(bs, qubo, sats)
    assert not result.feasible
    assert any("Energy budget exceeded" in v for v in result.violations)


# ------------------------------------------------------------------ #
# 7 & 8. QAOA ignores infeasible lowest-energy sample, picks best feasible
# ------------------------------------------------------------------ #

def test_best_feasible_sample_chosen_over_lowest_energy():
    """
    Two candidates on same target → selecting both is infeasible.
    counts has:
      "11" → lowest energy but infeasible (both selected)
      "10" → higher energy but feasible (only candidate 0)
      "01" → feasible (only candidate 1)
    Expected: best feasible = "10" (score 0.9 > 0.8)
    """
    candidates = [
        _make_candidate("SAT-01", "TGT-01", 0.9, offset_s=0),
        _make_candidate("SAT-02", "TGT-01", 0.8, offset_s=300),
    ]
    qubo = build_qubo(candidates, conflict_penalty=0.01)  # tiny penalty so "11" has lowest energy
    # Force "11" to have lowest QUBO energy by using tiny penalty
    # Manually verify: energy("11") = -0.9 - 0.8 + 0.01 + 0.01 = -1.68
    # energy("10") = -0.9,  energy("01") = -0.8
    counts = {"11": 100, "10": 50, "01": 30}
    fake_result = _make_fake_qaoa_result(counts, qubo)

    validation = _best_feasible_validation(fake_result, qubo)
    assert validation.feasible
    assert validation.objective_value == pytest.approx(0.9)
    assert validation.feasible_sample_count == 2  # "10" and "01" are both feasible


# ------------------------------------------------------------------ #
# 9. Zero feasible samples produces None score
# ------------------------------------------------------------------ #

def test_zero_feasible_samples_gives_none_score():
    """All sampled bitstrings select conflicting candidates."""
    candidates = [
        _make_candidate("SAT-01", "TGT-01", 0.9),
        _make_candidate("SAT-02", "TGT-01", 0.8),
    ]
    qubo = build_qubo(candidates, conflict_penalty=0.01)
    # Only "11" in counts → always infeasible (same target)
    counts = {"11": 512}
    fake_result = _make_fake_qaoa_result(counts, qubo)

    validation = _best_feasible_validation(fake_result, qubo)
    assert not validation.feasible
    assert validation.objective_value is None
    assert validation.feasible_sample_count == 0


# ------------------------------------------------------------------ #
# 10. Qiskit bitstring ordering decoded correctly
# ------------------------------------------------------------------ #

def test_bitstring_ordering_qubit0_is_rightmost():
    """
    Qiskit convention: rightmost character = qubit 0.
    Bitstring "01" means qubit 0 = 1, qubit 1 = 0.
    So only candidate at index 0 should be selected.
    """
    candidates = [
        _make_candidate("SAT-01", "TGT-01", 0.9),
        _make_candidate("SAT-02", "TGT-02", 0.8),
    ]
    qubo = build_qubo(candidates)
    # "01" → reversed → ['1', '0'] → index 0 selected, index 1 not
    selected = _decode_qaoa_bitstring("01", qubo)
    assert len(selected) == 1
    assert selected[0].opportunity.target_id == "TGT-01"


def test_bitstring_ordering_qubit1_is_leftmost():
    """
    Bitstring "10" means qubit 0 = 0, qubit 1 = 1.
    So only candidate at index 1 should be selected.
    """
    candidates = [
        _make_candidate("SAT-01", "TGT-01", 0.9),
        _make_candidate("SAT-02", "TGT-02", 0.8),
    ]
    qubo = build_qubo(candidates)
    selected = _decode_qaoa_bitstring("10", qubo)
    assert len(selected) == 1
    assert selected[0].opportunity.target_id == "TGT-02"
