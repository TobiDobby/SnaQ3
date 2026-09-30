"""Four interchangeable QuickQudits simulation modes."""

from .qutrit_circuit import QutritCircuit
from .qutrit_tableau import QutritTableau
from .qubit_transpiled import QubitTranspiled
from .qubit_tableau_transpiled import QubitTableauTranspiled

BACKENDS = {
    QutritCircuit.name: QutritCircuit,
    QutritTableau.name: QutritTableau,
    QubitTranspiled.name: QubitTranspiled,
    QubitTableauTranspiled.name: QubitTableauTranspiled,
}
