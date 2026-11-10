from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit

from satellite_optimizer.optimization.qubo import QUBOProblem


@dataclass(frozen=True)
class QAOACircuit:
    """QAOA problem structure — Ising coefficients + metadata."""

    linear: np.ndarray
    quadratic: np.ndarray
    depth: int
    num_qubits: int


def _ising_coefficients(
    problem: QUBOProblem,
) -> tuple[np.ndarray, np.ndarray]:
    # Use only the observation submatrix — slack qubits are not in the circuit
    n = problem.num_qubits
    matrix = np.asarray(problem.matrix[:n, :n], dtype=float)

    linear = np.zeros(n, dtype=float)
    quadratic = np.zeros((n, n), dtype=float)

    for i in range(n):
        linear[i] = -0.5 * matrix[i, i]

    for i in range(n):
        for j in range(i + 1, n):
            qij = matrix[i, j]
            if qij == 0.0:
                continue
            quadratic[i, j] = 0.5 * qij
            linear[i] -= 0.5 * qij
            linear[j] -= 0.5 * qij

    return linear, quadratic


def build_qaoa_circuit(
    problem: QUBOProblem,
    depth: int = 2,
) -> QAOACircuit:
    if problem.num_variables <= 0:
        raise ValueError("QAOA requires at least one QUBO variable.")
    if depth <= 0:
        raise ValueError("QAOA depth must be greater than zero.")

    linear, quadratic = _ising_coefficients(problem)

    return QAOACircuit(
        linear=linear,
        quadratic=quadratic,
        depth=depth,
        num_qubits=problem.num_qubits,
    )


def build_bound_circuit(
    qaoa_circuit: QAOACircuit,
    gammas: list[float],
    betas: list[float],
) -> QuantumCircuit:
    """Build a concrete QuantumCircuit with all angles bound to floats."""
    n = qaoa_circuit.num_qubits
    linear = qaoa_circuit.linear
    quadratic = qaoa_circuit.quadratic

    qc = QuantumCircuit(n, n)
    qc.h(range(n))

    for layer in range(qaoa_circuit.depth):
        g = float(gammas[layer])
        b = float(betas[layer])

        for qubit in range(n):
            c = float(linear[qubit])
            if c != 0.0:
                qc.rz(2.0 * g * c, qubit)

        for i in range(n):
            for j in range(i + 1, n):
                c = float(quadratic[i, j])
                if c != 0.0:
                    qc.rzz(2.0 * g * c, i, j)

        for qubit in range(n):
            qc.rx(2.0 * b, qubit)

    qc.measure(range(n), range(n))
    return qc
