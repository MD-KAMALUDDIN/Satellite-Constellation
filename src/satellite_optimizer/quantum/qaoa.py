from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize

from satellite_optimizer.optimization.qubo import QUBOProblem
from satellite_optimizer.quantum.circuit import (
    QAOACircuit,
    build_bound_circuit,
    build_qaoa_circuit,
)


@dataclass(frozen=True)
class QAOAResult:
    """Result returned by a local QAOA execution."""

    best_parameters: np.ndarray
    best_energy: float
    counts: dict[str, int]
    best_bitstring: str
    # raw_best_* = lowest QUBO energy regardless of feasibility
    raw_best_bitstring: str
    raw_best_energy: float
    feasible_sample_count: int
    circuit: QAOACircuit
    circuit_depth: int
    gate_count: int
    two_qubit_gate_count: int
    shots: int
    backend_name: str


def _energy_from_bitstring(
    problem: QUBOProblem,
    bitstring: str,
) -> float:
    """Evaluate x^T Q_obs x using only the observation submatrix."""
    n = problem.num_qubits  # observation variables only
    bits = np.asarray(
        [int(b) for b in reversed(bitstring)],
        dtype=float,
    )
    # slice the top-left n×n block (observation variables only)
    Q_obs = problem.matrix[:n, :n]
    return float(bits @ Q_obs @ bits)


def _objective(
    parameters: np.ndarray,
    qaoa_circuit: QAOACircuit,
    problem: QUBOProblem,
) -> float:
    depth = qaoa_circuit.depth
    gammas = parameters[:depth].tolist()
    betas = parameters[depth:].tolist()

    qc = build_bound_circuit(qaoa_circuit, gammas, betas)
    qc_no_meas = qc.remove_final_measurements(inplace=False)

    sv = Statevector.from_instruction(qc_no_meas)
    probs = sv.probabilities()

    energy = 0.0
    for index, prob in enumerate(probs):
        if prob == 0.0:
            continue
        bs = format(index, f"0{problem.num_qubits}b")
        energy += prob * _energy_from_bitstring(problem, bs)

    return float(energy)


def _sample(
    qaoa_circuit: QAOACircuit,
    parameters: np.ndarray,
    shots: int,
) -> dict[str, int]:
    depth = qaoa_circuit.depth
    gammas = parameters[:depth].tolist()
    betas = parameters[depth:].tolist()

    qc = build_bound_circuit(qaoa_circuit, gammas, betas)
    backend = AerSimulator()
    transpiled = transpile(qc, backend=backend)
    result = backend.run(transpiled, shots=shots).result()
    return dict(result.get_counts())


def _best_sample(
    counts: dict[str, int],
    problem: QUBOProblem,
) -> tuple[str, float]:
    """Return (bitstring, energy) for the lowest-QUBO-energy sample."""
    best_bitstring = ""
    best_energy = float("inf")

    for bitstring in counts:
        energy = _energy_from_bitstring(problem, bitstring)
        if energy < best_energy:
            best_energy = energy
            best_bitstring = bitstring

    return best_bitstring, best_energy


def ranked_samples(
    counts: dict[str, int],
    problem: QUBOProblem,
) -> list[tuple[float, str]]:
    """Return all sampled bitstrings sorted by QUBO energy ascending."""
    return sorted(
        (
            (_energy_from_bitstring(problem, bs), bs)
            for bs in counts
        ),
        key=lambda t: t[0],
    )


def _count_two_qubit_gates(circuit) -> int:
    return sum(
        1 for inst in circuit.data
        if inst.operation.num_qubits == 2
    )


def run_qaoa(
    problem: QUBOProblem,
    depth: int = 2,
    shots: int = 2048,
    max_iterations: int = 100,
    seed: int = 42,
) -> QAOAResult:
    if problem.num_variables <= 0:
        raise ValueError("QAOA requires a non-empty QUBO.")
    if depth <= 0:
        raise ValueError("QAOA depth must be greater than zero.")
    if shots <= 0:
        raise ValueError("shots must be greater than zero.")

    qaoa_circuit = build_qaoa_circuit(problem=problem, depth=depth)

    rng = np.random.default_rng(seed)
    initial_parameters = np.concatenate([
        rng.uniform(0.0, np.pi, depth),
        rng.uniform(0.0, np.pi / 2.0, depth),
    ])

    optimization = minimize(
        _objective,
        initial_parameters,
        args=(qaoa_circuit, problem),
        method="COBYLA",
        options={"maxiter": max_iterations},
    )

    best_parameters = np.asarray(optimization.x, dtype=float)
    best_energy = _objective(best_parameters, qaoa_circuit, problem)

    counts = _sample(qaoa_circuit, best_parameters, shots)
    raw_best_bitstring, raw_best_energy = _best_sample(counts, problem)

    # build a final transpiled circuit for stats + SVG rendering
    depth_val = qaoa_circuit.depth
    gammas = best_parameters[:depth_val].tolist()
    betas = best_parameters[depth_val:].tolist()
    final_qc = build_bound_circuit(qaoa_circuit, gammas, betas)
    backend = AerSimulator()
    transpiled = transpile(final_qc, backend=backend)

    return QAOAResult(
        best_parameters=best_parameters,
        best_energy=min(best_energy, raw_best_energy),
        counts=counts,
        # best_bitstring is set to raw for now; evaluator will override
        # with the best feasible sample after classical validation.
        best_bitstring=raw_best_bitstring,
        raw_best_bitstring=raw_best_bitstring,
        raw_best_energy=raw_best_energy,
        feasible_sample_count=0,  # filled in by run_hybrid_evaluation
        circuit=qaoa_circuit,
        circuit_depth=transpiled.depth(),
        gate_count=len(transpiled.data),
        two_qubit_gate_count=_count_two_qubit_gates(transpiled),
        shots=shots,
        backend_name="AerSimulator",
    )
