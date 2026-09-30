"""Binary qutrit embedding using only native qubit Clifford gates."""


def append_encoded_x(circuit, cell: int) -> None:
    """Transpile qutrit X on cell: 00→01→10→00, while 11 stays fixed.

    QuickQudits d=2 gate convention gives the sequence X(q0), CX(q0,q1),
    CX(q1,q0). It is an affine Clifford permutation on the full four-state
    space, so it preserves the encoded qutrit subspace exactly.
    """
    q0, q1 = 2 * cell, 2 * cell + 1
    circuit.X(q0)
    circuit.CX(q0, q1)
    circuit.CX(q1, q0)
