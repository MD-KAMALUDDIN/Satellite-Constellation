from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from satellite_optimizer.optimization.classical import solve_classical
from satellite_optimizer.optimization.objective import calculate_objective_value
from satellite_optimizer.optimization.qubo import QUBOProblem, build_qubo
from satellite_optimizer.optimization.reduction import ScoredCandidate
from satellite_optimizer.quantum.qaoa import QAOAResult, ranked_samples, run_qaoa

_ENERGY_TOLERANCE_WH = 1e-9


@dataclass(frozen=True)
class ValidationResult:
    """Classical validation result for a QAOA-selected bitstring."""

    selected_candidates: list[ScoredCandidate]
    # None means QAOA produced no feasible sample — never fake 0.0
    objective_value: Optional[float]
    feasible: bool
    violations: list[str]
    feasible_sample_count: int


@dataclass(frozen=True)
class ComparisonResult:
    """Comparison between the classical baseline and QAOA."""

    classical_score: float
    # None when QAOA found no feasible sample
    qaoa_score: Optional[float]
    score_gap: Optional[float]
    qaoa_relative_gap_percent: Optional[float]
    same_solution: bool


@dataclass(frozen=True)
class HybridEvaluationResult:
    """Complete result of classical + QUBO + QAOA optimization."""

    classical_result: object
    qubo: QUBOProblem
    qaoa_result: QAOAResult
    qaoa_validation: ValidationResult
    comparison: ComparisonResult
    final_selected_candidates: list[ScoredCandidate]
    final_method: str


def _overlap(first: ScoredCandidate, second: ScoredCandidate) -> bool:
    """Return True when two observation opportunities overlap in time."""
    first_opportunity = first.opportunity
    second_opportunity = second.opportunity

    return (
        first_opportunity.start_time < second_opportunity.end_time
        and second_opportunity.start_time < first_opportunity.end_time
    )


def _decode_qaoa_bitstring(
    bitstring: str,
    qubo: QUBOProblem,
) -> list[ScoredCandidate]:
    """
    Decode a Qiskit-style bitstring into selected observation candidates.

    Qiskit stores the highest-index qubit on the left, so the bitstring
    is reversed before mapping bits to QUBO variable indices.
    Slack variables are ignored — they are never returned as tasks.
    """
    if len(bitstring) != qubo.num_qubits:
        raise ValueError(
            "QAOA bitstring length must equal the number of observation qubits."
        )

    variable_bits = list(reversed(bitstring))
    selected: list[ScoredCandidate] = []

    for meta in qubo.variable_meta:
        if meta.kind == "observation" and variable_bits[meta.index] == "1":
            selected.append(qubo.candidates[meta.candidate_index])

    return selected


def validate_qaoa_selection(
    bitstring: str,
    qubo: QUBOProblem,
    satellites: list | None = None,
    battery_reserve_fraction: float = 0.20,
) -> ValidationResult:
    """
    Validate a QAOA solution against all mission constraints:
      - infeasible opportunity
      - duplicate target
      - same-satellite temporal overlap
      - energy budget per satellite
    """
    selected_candidates = _decode_qaoa_bitstring(bitstring, qubo)
    violations: list[str] = []

    # infeasible opportunity
    for c in selected_candidates:
        if not c.opportunity.feasible:
            violations.append(
                f"Infeasible opportunity: "
                f"{c.opportunity.satellite_id} -> {c.opportunity.target_id}"
            )

    # duplicate target
    for i in range(len(selected_candidates)):
        for j in range(i + 1, len(selected_candidates)):
            a = selected_candidates[i]
            b = selected_candidates[j]
            if a.opportunity.target_id == b.opportunity.target_id:
                violations.append(
                    f"Duplicate target: {a.opportunity.target_id}"
                )
            if (
                a.opportunity.satellite_id == b.opportunity.satellite_id
                and _overlap(a, b)
            ):
                violations.append(
                    f"Satellite overlap: {a.opportunity.satellite_id}"
                )

    # energy budget per satellite
    if satellites:
        sat_map = {s.id: s for s in satellites}
        energy_used: dict[str, float] = {}
        for c in selected_candidates:
            sid = c.opportunity.satellite_id
            energy_used[sid] = (
                energy_used.get(sid, 0.0) + c.opportunity.energy_cost_wh
            )
        for sid, used in energy_used.items():
            sat = sat_map.get(sid)
            if sat is None:
                continue
            budget = (
                sat.initial_battery_wh
                - battery_reserve_fraction * sat.battery_capacity_wh
            )
            if used > budget + _ENERGY_TOLERANCE_WH:
                violations.append(
                    f"Energy budget exceeded: {sid} "
                    f"(used {used:.3f} Wh, budget {budget:.3f} Wh)"
                )

    feasible = len(violations) == 0
    objective_value: Optional[float] = (
        calculate_objective_value(selected_candidates) if feasible else None
    )

    return ValidationResult(
        selected_candidates=selected_candidates,
        objective_value=objective_value,
        feasible=feasible,
        violations=violations,
        feasible_sample_count=0,
    )


def _best_feasible_validation(
    qaoa_result: QAOAResult,
    qubo: QUBOProblem,
    satellites: list | None = None,
) -> ValidationResult:
    """
    Scan every sampled bitstring and return the ValidationResult for the
    highest-mission-score feasible sample.
    If none is feasible, returns infeasible with objective_value=None.
    """
    samples = ranked_samples(qaoa_result.counts, qubo)

    best_validation: Optional[ValidationResult] = None
    best_score: float = -1.0
    feasible_count = 0

    for _energy, bitstring in samples:
        v = validate_qaoa_selection(bitstring, qubo, satellites)
        if v.feasible:
            feasible_count += 1
            score = v.objective_value if v.objective_value is not None else 0.0
            if best_validation is None or score > best_score:
                best_score = score
                best_validation = v

    if best_validation is not None:
        return ValidationResult(
            selected_candidates=best_validation.selected_candidates,
            objective_value=best_validation.objective_value,
            feasible=True,
            violations=[],
            feasible_sample_count=feasible_count,
        )

    raw_v = validate_qaoa_selection(
        qaoa_result.raw_best_bitstring, qubo, satellites
    )
    return ValidationResult(
        selected_candidates=raw_v.selected_candidates,
        objective_value=None,
        feasible=False,
        violations=raw_v.violations,
        feasible_sample_count=0,
    )


def _candidate_identity(
    candidates: list[ScoredCandidate],
) -> set[tuple[str, str, object]]:
    """Create stable identities for comparing selected opportunities."""
    return {
        (
            candidate.opportunity.satellite_id,
            candidate.opportunity.target_id,
            candidate.opportunity.start_time,
        )
        for candidate in candidates
    }


def compare_results(
    classical_candidates: list[ScoredCandidate],
    qaoa_validation: ValidationResult,
) -> ComparisonResult:
    """Compare the validated QAOA result against CP-SAT."""
    classical_score = calculate_objective_value(classical_candidates)
    qaoa_score = qaoa_validation.objective_value  # None if no feasible sample

    if qaoa_score is not None:
        score_gap: Optional[float] = classical_score - qaoa_score
        relative_gap: Optional[float] = (
            (score_gap / classical_score) * 100.0
            if classical_score > 0.0
            else 0.0
        )
    else:
        score_gap = None
        relative_gap = None

    same_solution = (
        qaoa_validation.feasible
        and _candidate_identity(classical_candidates)
        == _candidate_identity(qaoa_validation.selected_candidates)
    )

    return ComparisonResult(
        classical_score=classical_score,
        qaoa_score=qaoa_score,
        score_gap=score_gap,
        qaoa_relative_gap_percent=relative_gap,
        same_solution=same_solution,
    )


def run_hybrid_evaluation(
    candidates: list[ScoredCandidate],
    satellites: list | None = None,
    qaoa_depth: int = 2,
    shots: int = 2048,
    max_iterations: int = 50,
    seed: int = 42,
    classical_time_limit_seconds: float = 30.0,
) -> HybridEvaluationResult:
    """
    Run the complete classical + QUBO + QAOA evaluation pipeline.
    """
    if not candidates:
        raise ValueError(
            "Hybrid evaluation requires at least one candidate."
        )

    classical_result = solve_classical(
        candidates=candidates,
        time_limit_seconds=classical_time_limit_seconds,
    )

    qubo = build_qubo(candidates, satellites=satellites)

    qaoa_result = run_qaoa(
        problem=qubo,
        depth=qaoa_depth,
        shots=shots,
        max_iterations=max_iterations,
        seed=seed,
    )

    qaoa_validation = _best_feasible_validation(qaoa_result, qubo, satellites)

    comparison = compare_results(
        classical_candidates=classical_result.selected_candidates,
        qaoa_validation=qaoa_validation,
    )

    if qaoa_validation.feasible and qaoa_validation.objective_value >= classical_result.objective_value:
        final_selected_candidates = qaoa_validation.selected_candidates
        final_method = "QAOA"
    else:
        final_selected_candidates = classical_result.selected_candidates
        final_method = "CP-SAT"

    return HybridEvaluationResult(
        classical_result=classical_result,
        qubo=qubo,
        qaoa_result=qaoa_result,
        qaoa_validation=qaoa_validation,
        comparison=comparison,
        final_selected_candidates=final_selected_candidates,
        final_method=final_method,
    )