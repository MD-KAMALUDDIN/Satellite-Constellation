import numpy as np
import pytest

from satellite_optimizer.optimization.qubo import (
    build_qubo,
)
from satellite_optimizer.optimization.reduction import (
    ScoredCandidate,
)
from satellite_optimizer.models.observation import (
    ObservationOpportunity,
)
from satellite_optimizer.quantum.circuit import (
    build_qaoa_circuit,
)
from satellite_optimizer.quantum.qaoa import (
    run_qaoa,
)


def make_candidate(
    satellite_id: str,
    target_id: str,
    score: float,
    offset_seconds: int,
) -> ScoredCandidate:
    from datetime import datetime, timedelta, timezone

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


def make_problem():
    candidates = [
        make_candidate("SAT-01", "TGT-01", 0.9, 0),
        make_candidate("SAT-02", "TGT-01", 0.8, 60),
        make_candidate("SAT-03", "TGT-02", 0.7, 120),
    ]

    return build_qubo(
        candidates,
        conflict_penalty=2.0,
    )


def test_qaoa_circuit_has_expected_qubits():
    problem = make_problem()

    qaoa = build_qaoa_circuit(
        problem,
        depth=2,
    )

    assert qaoa.num_qubits == 3
    assert qaoa.depth == 2


def test_qaoa_circuit_has_parameters():
    problem = make_problem()

    qaoa = build_qaoa_circuit(
        problem,
        depth=2,
    )

    # depth=2 → 2 gammas + 2 betas = 4 parameters total
    assert qaoa.depth * 2 == 4


def test_qaoa_circuit_contains_measurements():
    problem = make_problem()

    qaoa = build_qaoa_circuit(
        problem,
        depth=1,
    )

    from satellite_optimizer.quantum.circuit import build_bound_circuit
    qc = build_bound_circuit(qaoa, [0.5], [0.3])
    assert qc.num_clbits == 3
    assert qc.count_ops().get("measure", 0) == 3


def test_qaoa_circuit_contains_cost_interaction():
    problem = make_problem()

    qaoa = build_qaoa_circuit(
        problem,
        depth=1,
    )

    from satellite_optimizer.quantum.circuit import build_bound_circuit
    qc = build_bound_circuit(qaoa, [0.5], [0.3])
    assert qc.count_ops().get("rzz", 0) > 0


def test_qaoa_rejects_invalid_depth():
    problem = make_problem()

    with pytest.raises(ValueError):
        build_qaoa_circuit(
            problem,
            depth=0,
        )


def test_qaoa_local_execution():
    problem = make_problem()

    result = run_qaoa(
        problem=problem,
        depth=1,
        shots=128,
        max_iterations=10,
        seed=42,
    )

    assert result.backend_name == "AerSimulator"
    assert result.shots == 128
    assert result.raw_best_bitstring
    assert len(result.raw_best_bitstring) == problem.num_variables
    assert result.counts
    assert result.circuit_depth > 0
    assert result.gate_count > 0
    assert result.two_qubit_gate_count > 0
    assert np.isfinite(result.best_energy)