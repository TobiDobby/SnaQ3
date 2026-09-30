"""Canonical direction phases and independent movement registers."""

from math import pi

import quickqudits as qq

PHASES = (0.0, pi / 2, pi, 3 * pi / 2)
DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
DIRECTION_NAMES = ("UP", "RIGHT", "DOWN", "LEFT")


def phase_to_input(phi: float) -> int:
    for x, known in enumerate(PHASES):
        if abs((phi - known + pi) % (2 * pi) - pi) < 1e-12:
            return x
    raise ValueError("phase must be a canonical direction phase")


def prepare_movement(x: int):
    """H then S^x implements Rz(x*pi/2), up to a global phase."""
    if x not in range(4):
        raise ValueError("direction must be 0..3")
    circuit = qq.QuantumCircuit(1, 2)
    circuit.H(0)
    for _ in range(x):
        circuit.S(0)
    return circuit.execute()[0]


def prepare_movement_tableau(x: int) -> qq.Tableau:
    """Prepare the same movement phase on QuickQudits' qubit tableau."""
    if x not in range(4):
        raise ValueError("direction must be 0..3")
    circuit = qq.QuantumCircuit(1, 2)
    circuit.H(0)
    for _ in range(x):
        circuit.S(0)
    tableau = qq.Tableau(1, 2)
    tableau.apply_circuit(circuit, engine="python")
    return tableau
