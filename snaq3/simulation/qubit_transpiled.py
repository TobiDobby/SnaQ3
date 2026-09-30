"""Full-grid binary embedding executed as QuickQudits d=2 statevectors."""

import quickqudits as qq

from ..model import run_game
from ..resolver import exact_sample
from ._memory import require_two_dense_grids
from .encoding import append_encoded_x


class EncodedGrid:
    def __init__(self, digits):
        self.digits = [0] * len(digits)
        self.circuit = qq.QuantumCircuit(2 * len(digits), 2)
        require_two_dense_grids(len(digits), 2, qudits_per_cell=2)
        for i, desired in enumerate(digits):
            for _ in range(desired):
                append_encoded_x(self.circuit, i)
        self.state, _ = self.circuit.execute()
        self.digits = list(digits)

    def set_digits(self, digits):
        circuit = qq.QuantumCircuit(2 * len(self.digits), 2)
        for i, desired in enumerate(digits):
            for _ in range((desired - self.digits[i]) % 3):
                append_encoded_x(circuit, i)
        if circuit.ops:
            self.state, _ = circuit.execute(psi0=self.state)
        self.circuit = circuit
        self.digits = list(digits)


class QubitTranspiled:
    name = "qubit_transpiled"
    metadata = {"grid_backend": "two full QuickQudits QuantumCircuit d=2 statevectors, 2m qubits each",
                "memory_guard": "four dense vectors within configured ceiling and half available memory",
                "encoding": {"0": "00", "1": "01", "2": "10", "invalid": "11"},
                "transpiled_gates": "qutrit X -> X(q0), CX(q0,q1), CX(q1,q0); invalid 11 fixed",
                "bell_backend": "QuickQudits QuantumCircuit d=2, same exact Bell sequence"}

    def make_grid(self, digits):
        return EncodedGrid(digits)

    def resolve(self, x_a, x_b, seed):
        return exact_sample(x_a, x_b, seed)

    def run(self, n, inputs, seed):
        return run_game(self, n, inputs, seed)
