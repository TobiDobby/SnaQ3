"""Classical direction labels used by Snake grid movement."""

DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
DIRECTION_NAMES = ("UP", "RIGHT", "DOWN", "LEFT")


def prepare_movement_tableau(x: int):
    """Legacy tableau API; active gameplay and runtime replay do not call it."""
    import quickqudits as qq

    if x not in range(4):
        raise ValueError("direction must be 0..3")
    circuit = qq.QuantumCircuit(1, 2)
    circuit.H(0)
    for _ in range(x):
        circuit.S(0)
    tableau = qq.Tableau(1, 2)
    tableau.apply_circuit(circuit, engine="python")
    return tableau
