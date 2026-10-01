"""Full 2m-qubit Clifford tableaus for the binary-encoded player grids."""

import quickqudits as qq

from ..model import run_game
from ..movement import prepare_movement_tableau
from ..resolver import clifford_mapping, tableau_sample
from .encoding import append_encoded_x


class EncodedTableauGrid:
    def __init__(self, digits):
        self.digits = [0] * len(digits)
        self.tableau = qq.Tableau(2 * len(digits), 2)
        self.set_digits(digits)

    def set_digits(self, digits):
        circuit = qq.QuantumCircuit(2 * len(self.digits), 2)
        for i, desired in enumerate(digits):
            for _ in range((desired - self.digits[i]) % 3):
                append_encoded_x(circuit, i)
        if circuit.ops:
            self.tableau.apply_circuit(circuit, engine="python")
        self.digits = list(digits)


class QubitTableauTranspiled:
    name = "qubit_tableau_transpiled"
    metadata = {
        "grid_backend": "two full QuickQudits Tableau d=2 registers, 2m qubits each",
        "bell_backend": "QuickQudits Tableau d=2; same Clifford Bell observables as qutrit_tableau_clifford",
        "encoding": {"0": "00", "1": "01", "2": "10", "invalid": "11"},
        "transpiled_gates": "qutrit X -> X(q0), CX(q0,q1), CX(q1,q0); invalid 11 fixed",
        "observable_mapping": clifford_mapping(),
        "dense_statevector": False,
    }

    def __init__(self):
        self.metadata = dict(type(self).metadata)

    def make_grid(self, digits):
        grid = EncodedTableauGrid(digits)
        self.metadata["qubits_per_player_grid"] = grid.tableau.n
        self.metadata["tableau_array_bytes_per_player"] = (
            grid.tableau.X.nbytes + grid.tableau.Z.nbytes + grid.tableau.tau_exp.nbytes
        )
        return grid

    def prepare_movement(self, x):
        return prepare_movement_tableau(x)

    def resolve(self, x_a, x_b, seed):
        return tableau_sample(x_a, x_b, seed)

    def run(self, n, inputs, seed):
        return run_game(self, n, inputs, seed)
