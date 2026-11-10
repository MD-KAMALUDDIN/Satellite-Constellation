from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from satellite_optimizer.optimization.reduction import (
    ScoredCandidate,
)


@dataclass(frozen=True)
class VariableMeta:
    """Metadata for one QUBO variable (qubit)."""

    index: int
    kind: Literal["observation", "slack"]
    # observation → index into candidates list
    # slack       → (satellite_id, bit_position)
    candidate_index: int | None = None
    satellite_id: str | None = None
    slack_bit: int | None = None


@dataclass(frozen=True)
class QUBOProblem:
    """QUBO representation of the scheduling problem."""

    matrix: np.ndarray
    candidates: list[ScoredCandidate]
    conflict_penalty: float
    energy_penalty: float
    # one entry per qubit in matrix order
    variable_meta: list[VariableMeta]
    # constant offset from slack expansion (does not affect optimisation)
    constant_offset: float

    # keep old name as alias so existing callers don't break
    @property
    def penalty(self) -> float:
        return self.conflict_penalty

    @property
    def num_observation_variables(self) -> int:
        return len(self.candidates)

    @property
    def num_slack_variables(self) -> int:
        return sum(
            1 for v in self.variable_meta if v.kind == "slack"
        )

    @property
    def num_variables(self) -> int:
        return len(self.variable_meta)

    @property
    def num_qubits(self) -> int:
        # QAOA circuit only operates on observation qubits.
        # Slack variables are tracked in metadata for classical
        # validation but are never mapped to quantum gates.
        return self.num_observation_variables


def _overlap(
    first: ScoredCandidate,
    second: ScoredCandidate,
) -> bool:
    """Return True when two observation windows overlap."""

    first_opportunity = first.opportunity
    second_opportunity = second.opportunity

    return (
        first_opportunity.start_time
        < second_opportunity.end_time
        and second_opportunity.start_time
        < first_opportunity.end_time
    )


def _same_target_conflict(
    first: ScoredCandidate,
    second: ScoredCandidate,
) -> bool:
    """Return True when two candidates observe the same target."""

    return (
        first.opportunity.target_id
        == second.opportunity.target_id
    )


def _same_satellite_overlap(
    first: ScoredCandidate,
    second: ScoredCandidate,
) -> bool:
    """Return True when candidates conflict on one satellite."""

    if (
        first.opportunity.satellite_id
        != second.opportunity.satellite_id
    ):
        return False

    return _overlap(first, second)


def _slack_bits_needed(max_value: float) -> int:
    """
    Number of binary slack bits needed to represent [0, max_value].

    We use a unary / power-of-two encoding:
        s = sum_{k=0}^{B-1} 2^k * s_k
    so B = ceil(log2(max_value + 1)), minimum 1.
    """
    if max_value <= 0.0:
        return 1
    import math
    return max(1, math.ceil(math.log2(max_value + 1)))


def build_qubo(
    candidates: list[ScoredCandidate],
    satellites: list | None = None,
    conflict_penalty: float | None = None,
    energy_penalty: float | None = None,
    battery_reserve_fraction: float = 0.20,
) -> QUBOProblem:
    """
    Build a QUBO matrix for the candidate scheduling problem.

    Variables
    ---------
    x_i  : 1 if observation candidate i is selected  (observation variables)
    s_k_b: binary slack bits for satellite k energy budget  (slack variables)

    Objective (minimise)
    --------------------
    H = -sum_i(w_i * x_i)                          # reward
      + M * sum_{conflicts}(x_i * x_j)             # conflict penalty
      + lam/2 * sum_k(sum_i(e_i*x_i) + S_k - E_max_k)^2  # energy penalty

    where S_k = sum_b(2^b * s_k_b)  is the slack for satellite k.

    Convention
    ----------
    energy = x^T Q x   (symmetric matrix, both Q[i,j] and Q[j,i] are set)
    so a pairwise term  a*x_i*x_j  contributes  a/2 to Q[i,j] and Q[j,i].
    A diagonal term     a*x_i      contributes  a   to Q[i,i]  (x_i^2=x_i).
    """

    num_candidates = len(candidates)

    if num_candidates == 0:
        raise ValueError(
            "Cannot build a QUBO with zero candidates."
        )

    # ------------------------------------------------------------------ #
    # Conflict penalty
    # ------------------------------------------------------------------ #
    if conflict_penalty is None:
        max_score = max(c.score for c in candidates)
        conflict_penalty = max(10.0 * max_score, 1.0)

    # ------------------------------------------------------------------ #
    # Energy budget per satellite
    # ------------------------------------------------------------------ #
    # Build satellite lookup: id -> (budget_wh, list of candidate indices)
    sat_budget: dict[str, float] = {}
    sat_candidates: dict[str, list[int]] = {}

    if satellites:
        for sat in satellites:
            budget = (
                sat.initial_battery_wh
                - battery_reserve_fraction * sat.battery_capacity_wh
            )
            sat_budget[sat.id] = max(budget, 0.0)

    for idx, c in enumerate(candidates):
        sid = c.opportunity.satellite_id
        sat_candidates.setdefault(sid, []).append(idx)
        if sid not in sat_budget:
            # fallback: no budget constraint for this satellite
            sat_budget[sid] = float("inf")

    # Only add energy slack for satellites with a finite budget
    constrained_sats = [
        sid for sid, budget in sat_budget.items()
        if budget < float("inf")
    ]
    constrained_sats.sort()  # deterministic ordering

    # ------------------------------------------------------------------ #
    # Variable layout
    # ------------------------------------------------------------------ #
    variable_meta: list[VariableMeta] = []

    # observation variables first
    for idx in range(num_candidates):
        variable_meta.append(
            VariableMeta(
                index=idx,
                kind="observation",
                candidate_index=idx,
            )
        )

    # slack variables per constrained satellite
    slack_start: dict[str, int] = {}  # satellite_id -> first slack qubit index
    slack_bits_count: dict[str, int] = {}

    for sid in constrained_sats:
        budget = sat_budget[sid]
        # max slack = budget (worst case: zero energy used)
        n_bits = _slack_bits_needed(budget)
        slack_start[sid] = len(variable_meta)
        slack_bits_count[sid] = n_bits
        for b in range(n_bits):
            variable_meta.append(
                VariableMeta(
                    index=len(variable_meta),
                    kind="slack",
                    satellite_id=sid,
                    slack_bit=b,
                )
            )

    n_total = len(variable_meta)
    matrix = np.zeros((n_total, n_total), dtype=float)

    # ------------------------------------------------------------------ #
    # Objective: -w_i * x_i  (diagonal)
    # ------------------------------------------------------------------ #
    for idx, candidate in enumerate(candidates):
        matrix[idx, idx] -= candidate.score

    # ------------------------------------------------------------------ #
    # Conflict penalties: M * x_i * x_j
    # pairwise term a*x_i*x_j -> Q[i,j] += a/2, Q[j,i] += a/2
    # but our convention stores full penalty in both entries already
    # (evaluate_qubo uses x^T Q x which double-counts off-diagonal)
    # ------------------------------------------------------------------ #
    for i in range(num_candidates):
        for j in range(i + 1, num_candidates):
            first = candidates[i]
            second = candidates[j]

            if _same_target_conflict(first, second) or _same_satellite_overlap(first, second):
                matrix[i, j] += conflict_penalty
                matrix[j, i] += conflict_penalty

    # ------------------------------------------------------------------ #
    # Energy penalty: lam/2 * (sum_i e_i x_i + S_k - E_max_k)^2
    #
    # Expand (A + S - E)^2 = A^2 + S^2 + E^2 + 2AS - 2AE - 2SE
    # where A = sum_i e_i x_i,  S = sum_b 2^b s_b
    #
    # Binary variable identities: x_i^2 = x_i, s_b^2 = s_b
    # ------------------------------------------------------------------ #
    if energy_penalty is None:
        energy_penalty = conflict_penalty  # same order of magnitude

    constant_offset = 0.0

    for sid in constrained_sats:
        budget = sat_budget[sid]
        indices = sat_candidates.get(sid, [])
        energies = [
            candidates[i].opportunity.energy_cost_wh for i in indices
        ]
        s_start = slack_start[sid]
        n_bits = slack_bits_count[sid]
        lam = energy_penalty

        # constant: lam/2 * E_max^2
        constant_offset += 0.5 * lam * budget ** 2

        # diagonal: lam/2 * e_i^2 * x_i  (from A^2, x_i^2=x_i)
        for local_i, global_i in enumerate(indices):
            e_i = energies[local_i]
            matrix[global_i, global_i] += 0.5 * lam * e_i ** 2

        # diagonal: lam/2 * (2^b)^2 * s_b  (from S^2, s_b^2=s_b)
        for b in range(n_bits):
            s_idx = s_start + b
            matrix[s_idx, s_idx] += 0.5 * lam * (2 ** b) ** 2

        # pairwise A*A cross terms: lam * e_i * e_j * x_i * x_j
        for li in range(len(indices)):
            for lj in range(li + 1, len(indices)):
                gi = indices[li]
                gj = indices[lj]
                coeff = lam * energies[li] * energies[lj]
                matrix[gi, gj] += coeff
                matrix[gj, gi] += coeff

        # pairwise S*S cross terms: lam * 2^b * 2^c * s_b * s_c
        for b in range(n_bits):
            for c in range(b + 1, n_bits):
                sb = s_start + b
                sc = s_start + c
                coeff = lam * (2 ** b) * (2 ** c)
                matrix[sb, sc] += coeff
                matrix[sc, sb] += coeff

        # pairwise A*S cross terms: lam * e_i * 2^b * x_i * s_b
        for local_i, global_i in enumerate(indices):
            e_i = energies[local_i]
            for b in range(n_bits):
                s_idx = s_start + b
                coeff = lam * e_i * (2 ** b)
                matrix[global_i, s_idx] += coeff
                matrix[s_idx, global_i] += coeff

        # linear A*(-E_max): lam * (-E_max) * e_i * x_i  -> diagonal
        for local_i, global_i in enumerate(indices):
            e_i = energies[local_i]
            matrix[global_i, global_i] += lam * (-budget) * e_i

        # linear S*(-E_max): lam * (-E_max) * 2^b * s_b  -> diagonal
        for b in range(n_bits):
            s_idx = s_start + b
            matrix[s_idx, s_idx] += lam * (-budget) * (2 ** b)

    return QUBOProblem(
        matrix=matrix,
        candidates=candidates,
        conflict_penalty=conflict_penalty,
        energy_penalty=energy_penalty,
        variable_meta=variable_meta,
        constant_offset=constant_offset,
    )


def evaluate_qubo(
    problem: QUBOProblem,
    bitstring: list[int] | np.ndarray,
) -> float:
    """
    Evaluate x^T Q x for a full binary assignment (obs + slack variables).
    """

    x = np.asarray(bitstring, dtype=float)

    if len(x) != problem.num_variables:
        raise ValueError(
            "Bitstring length does not match "
            "the number of QUBO variables."
        )

    if not np.all(np.logical_or(x == 0, x == 1)):
        raise ValueError(
            "QUBO bitstring must contain only 0 and 1."
        )

    return float(x @ problem.matrix @ x)