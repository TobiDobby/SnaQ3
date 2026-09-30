"""Two-qubit Bell conflict resource on QuickQudits d=2 backends."""

from math import atan2, cos, pi

import numpy as np
import quickqudits as qq
from quickqudits import gates as qq_gates
from quickqudits.gate_class import Gate

ALICE_ANGLES = (0.0, pi / 4, 0.0, pi / 4)
BOB_ANGLES = (pi / 8, -pi / 8, pi / 8, -pi / 8)
P_QUANTUM = cos(pi / 8) ** 2

# QuickQudits 1.0.1 has Clifford S but no non-Clifford T gate.
T_GATE = Gate("T", 1, lambda d: np.diag([1, np.exp(1j*pi/4)]) if d == 2 else None,
              lambda d: np.diag([1, np.exp(-1j*pi/4)]) if d == 2 else None)


def append_rotation(circuit: qq.QuantumCircuit, qubit: int, theta: float) -> None:
    """Apply R(-theta), so computational measurement uses basis theta.

    Sequence S† H T^k H S, k=-8 theta/pi, is exact up to global phase.
    """
    if abs(theta) < 1e-14:
        return
    exponent = round(-8 * theta / pi)
    if abs(theta + exponent * pi / 8) > 1e-12:
        raise ValueError("angle requires a different gate synthesis")
    circuit.Sdag(qubit)
    circuit.H(qubit)
    for _ in range(abs(exponent)):
        circuit.append(T_GATE, qubit, dagger=exponent < 0)
    circuit.H(qubit)
    circuit.S(qubit)


def exact_circuit(x_a: int, x_b: int) -> qq.QuantumCircuit:
    circuit = qq.QuantumCircuit(2, 2)
    circuit.H(0)
    circuit.CX(0, 1)
    append_rotation(circuit, 0, ALICE_ANGLES[x_a])
    append_rotation(circuit, 1, BOB_ANGLES[x_b])
    return circuit


def exact_probabilities(x_a: int, x_b: int) -> np.ndarray:
    state, _ = exact_circuit(x_a, x_b).execute()
    return abs(state.reshape(2, 2)) ** 2


def exact_sample(x_a: int, x_b: int, seed: int) -> tuple[int, int]:
    _, digits, _ = exact_circuit(x_a, x_b).sample(1, seed=seed)
    return tuple(map(int, digits[0]))


def _distance(a: float, b: float) -> float:
    return abs((a - b + pi / 2) % pi - pi / 2)


def clifford_mapping() -> dict:
    """Search real X/Z Clifford axes under QQ's H and S conventions."""
    # Derive each basis from the actual QuickQudits gate matrices. U maps
    # the chosen basis to the computational basis before measurement.
    h, x = qq_gates.H(2), qq_gates.X(2)
    candidates = []
    for observable, transform in (("Z", np.eye(2)), ("X", h),
                                  ("minus_X", x @ h)):
        basis_zero = transform.conj().T[:, 0]
        angle = atan2(float(basis_zero[1].real), float(basis_zero[0].real))
        candidates.append((angle, observable))
    result = {}
    for player, targets in (("Alice", ALICE_ANGLES), ("Bob", BOB_ANGLES)):
        for x, target in enumerate(targets):
            angle, observable = min(candidates, key=lambda pair: (_distance(target, pair[0]), abs(pair[0])))
            error = _distance(target, angle)
            result[f"{player}_{x}"] = {"target_theta": target,
                "implemented_theta": angle, "angular_error": error,
                "projector_fidelity": cos(error) ** 2, "observable": observable}
    return result


def tableau_circuit(x_a: int, x_b: int) -> qq.QuantumCircuit:
    mapping = clifford_mapping()
    circuit = qq.QuantumCircuit(2, 2)
    circuit.H(0)
    circuit.CX(0, 1)
    for q, party, x in ((0, "Alice", x_a), (1, "Bob", x_b)):
        observable = mapping[f"{party}_{x}"]["observable"]
        if observable == "X":
            circuit.H(q)
        elif observable == "minus_X":
            circuit.H(q)
            circuit.X(q)
    return circuit


def tableau_sample(x_a: int, x_b: int, seed: int) -> tuple[int, int]:
    tableau = qq.Tableau(2, 2)
    tableau.apply_circuit(tableau_circuit(x_a, x_b), engine="python")
    return tuple(map(int, tableau.sample_measurements(1, seed=seed)[0]))
