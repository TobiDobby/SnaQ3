"""Two independent native QuickQudits qutrit tableau grids."""

import quickqudits as qq

from ..model import run_game
from ..movement import prepare_movement_tableau
from ..resolver import clifford_mapping, tableau_sample


class TableauGrid:
    def __init__(self, digits):
        self.digits = [0] * len(digits)
        self.tableau = qq.Tableau(len(digits), 3)
        self.set_digits(digits)

    def set_digits(self, digits):
        circuit = qq.QuantumCircuit(len(digits), 3)
        for i, desired in enumerate(digits):
            for _ in range((desired - self.digits[i]) % 3):
                circuit.X(i)
        if circuit.ops:
            self.tableau.apply_circuit(circuit, engine="python")
        self.digits = list(digits)


class QutritTableau:
    name = "qutrit_tableau_clifford"
    metadata = {"grid_backend": "QuickQudits Tableau d=3; X only",
                "movement_backend": "QuickQudits Tableau d=2; H and S only",
                "bell_backend": "QuickQudits Tableau d=2; H CX and Clifford observables",
                "observable_mapping": clifford_mapping()}

    def make_grid(self, digits):
        return TableauGrid(digits)

    def prepare_movement(self, x):
        return prepare_movement_tableau(x)

    def resolve(self, x_a, x_b, seed):
        return tableau_sample(x_a, x_b, seed)

    def run(self, n, inputs, seed):
        return run_game(self, n, inputs, seed)
