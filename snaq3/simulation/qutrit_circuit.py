"""Two genuine full-grid QuickQudits d=3 statevectors."""

import quickqudits as qq

from ..model import run_game
from ..resolver import ALICE_ANGLES, BOB_ANGLES, exact_sample
from ._memory import require_two_dense_grids


class CircuitGrid:
    def __init__(self, digits):
        self.digits = [0] * len(digits)
        self.circuit = qq.QuantumCircuit(len(digits), 3)
        require_two_dense_grids(len(digits), 3)
        for i, desired in enumerate(digits):
            for _ in range(desired):
                self.circuit.X(i)
        self.state, _ = self.circuit.execute()
        self.digits = list(digits)

    def set_digits(self, digits):
        circuit = qq.QuantumCircuit(len(self.digits), 3)
        for i, desired in enumerate(digits):
            for _ in range((desired - self.digits[i]) % 3):
                circuit.X(i)
        if circuit.ops:
            self.state, _ = circuit.execute(psi0=self.state)
        self.circuit = circuit
        self.digits = list(digits)


class QutritCircuit:
    name = "qutrit_circuit_exact"
    metadata = {"grid_backend": "two full QuickQudits QuantumCircuit d=3 statevectors",
                "memory_guard": "four dense vectors within configured ceiling and half available memory",
                "bell_backend": "QuickQudits QuantumCircuit d=2",
                "gate_sequence": "grid qutrit X updates; exact Bell H(0) CX(0,1); observable Sdag H T^k H S",
                "observable_angles": {"Alice": ALICE_ANGLES, "Bob": BOB_ANGLES}}

    def make_grid(self, digits):
        return CircuitGrid(digits)

    def resolve(self, x_a, x_b, seed):
        return exact_sample(x_a, x_b, seed)

    def run(self, n, inputs, seed):
        return run_game(self, n, inputs, seed)
