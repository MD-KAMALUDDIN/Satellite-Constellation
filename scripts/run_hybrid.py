from __future__ import annotations

from datetime import datetime, timezone

from satellite_optimizer.hybrid.evaluator import run_hybrid_evaluation
from satellite_optimizer.optimization.reduction import reduce_candidates
from satellite_optimizer.scheduling.opportunities import generate_opportunities
from satellite_optimizer.utils.config import ConfigLoader


def main() -> None:
    """Run the complete classical + quantum hybrid optimization pipeline."""
    print()
    print("=" * 72)
    print("SATELLITE HYBRID OPTIMIZATION")
    print("=" * 72)

    loader = ConfigLoader()
    satellites, targets, mission = loader.load_all()

    mission_start = datetime.now(timezone.utc)

    print()
    print("Mission")
    print("-" * 72)
    print(f"Satellites:        {len(satellites)}")
    print(f"Targets:           {len(targets)}")
    print(f"Duration:          {mission.simulation.duration_hours} hours")
    print(
        f"QAOA depth:        {mission.quantum.default_depth}"
    )
    print(
        f"QAOA shots:        {mission.quantum.shots}"
    )

    print()
    print("1. Classical physics")
    print("-" * 72)

    opportunities = generate_opportunities(
        satellites=satellites,
        targets=targets,
        mission=mission,
        start_time=mission_start,
    )

    feasible_opportunities = [
        opportunity
        for opportunity in opportunities
        if opportunity.feasible
    ]

    print(f"Raw opportunities:  {len(opportunities)}")
    print(f"Feasible:           {len(feasible_opportunities)}")

    print()
    print("2. Candidate reduction")
    print("-" * 72)

    candidates = reduce_candidates(
        opportunities=opportunities,
        targets=targets,
        mission=mission,
        mission_start=mission_start,
    )

    print(f"Candidates:         {len(candidates)}")
    print(f"Qubits required:    {len(candidates)}")

    if not candidates:
        raise RuntimeError(
            "No candidates were produced. "
            "Cannot run optimization."
        )

    print()
    print("3. Hybrid optimization")
    print("-" * 72)
    print("Running CP-SAT baseline...")
    print("Running QAOA...")
    print(
        "Note: local QAOA uses exact statevector expectation "
        "during parameter optimization."
    )

    result = run_hybrid_evaluation(
        candidates=candidates,
        satellites=satellites,
        qaoa_depth=mission.quantum.default_depth,
        shots=mission.quantum.shots,
        max_iterations=50,
        seed=42,
        classical_time_limit_seconds=30.0,
    )

    classical = result.classical_result
    qaoa = result.qaoa_result
    validation = result.qaoa_validation
    comparison = result.comparison

    print()
    print("4. Classical baseline")
    print("-" * 72)
    print(f"Status:             {classical.solver_status}")
    print(f"Selected tasks:     {len(classical.selected_candidates)}")
    print(f"Objective:          {classical.objective_value:.6f}")

    print()
    print("5. QAOA")
    print("-" * 72)
    print(f"Backend:             {qaoa.backend_name}")
    print(f"Observation vars:    {result.qubo.num_observation_variables}")
    print(f"Slack vars:          {result.qubo.num_slack_variables}")
    print(f"Total qubits:        {result.qubo.num_qubits}")
    print(f"QAOA depth:          {qaoa.circuit.depth}")
    print(f"Circuit depth:       {qaoa.circuit_depth}")
    print(f"Total gates:         {qaoa.gate_count}")
    print(f"2-qubit gates:       {qaoa.two_qubit_gate_count}")
    print(f"Shots:               {qaoa.shots}")
    print(f"Raw best bitstring:  {qaoa.raw_best_bitstring}")
    print(f"Raw best energy:     {qaoa.raw_best_energy:.6f}")
    print(f"Expectation energy:  {qaoa.best_energy:.6f}")

    print()
    print("6. Classical validation")
    print("-" * 72)
    total_samples = len(qaoa.counts)
    print(f"Total unique samples: {total_samples}")
    print(f"Feasible samples:     {validation.feasible_sample_count}")
    print(f"Feasible:             {validation.feasible}")
    print(f"Selected tasks:       {len(validation.selected_candidates)}")
    score_str = (
        f"{validation.objective_value:.6f}"
        if validation.objective_value is not None
        else "N/A"
    )
    print(f"Mission score:        {score_str}")

    if validation.violations:
        print("Violations:")
        for v in validation.violations:
            print(f"  - {v}")

    print()
    print("7. Comparison")
    print("-" * 72)
    print(f"CP-SAT score:        {comparison.classical_score:.6f}")
    qaoa_score_str = (
        f"{comparison.qaoa_score:.6f}"
        if comparison.qaoa_score is not None
        else "N/A"
    )
    gap_str = (
        f"{comparison.score_gap:.6f}"
        if comparison.score_gap is not None
        else "N/A"
    )
    rel_str = (
        f"{comparison.qaoa_relative_gap_percent:.2f}%"
        if comparison.qaoa_relative_gap_percent is not None
        else "N/A"
    )
    print(f"QAOA score:          {qaoa_score_str}")
    print(f"Score gap:           {gap_str}")
    print(f"QAOA relative gap:   {rel_str}")
    print(f"Same solution:       {comparison.same_solution}")

    print()
    print("8. Final recommendation")
    print("-" * 72)
    print(f"Method:              {result.final_method}")
    print(
        f"Selected tasks:      "
        f"{len(result.final_selected_candidates)}"
    )

    for index, candidate in enumerate(
        result.final_selected_candidates,
        start=1,
    ):
        opportunity = candidate.opportunity

        print(
            f"  {index:02d}. "
            f"{opportunity.satellite_id} -> "
            f"{opportunity.target_id} | "
            f"score={candidate.score:.4f} | "
            f"start={opportunity.start_time.isoformat()}"
        )

    print()
    print("=" * 72)
    print("HYBRID OPTIMIZATION COMPLETE")
    print("=" * 72)
    print()
    print(
        "The quantum circuit is available to the dashboard "
        "through result.qaoa_result.circuit."
    )
    print()


if __name__ == "__main__":
    main()