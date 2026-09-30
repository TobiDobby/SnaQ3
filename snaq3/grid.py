"""Classical head/tail ordering and basis occupation of independent grids."""

from dataclasses import dataclass, field
from random import Random

import numpy as np
import quickqudits as qq
from quickqudits.gate_class import Gate

from .movement import DIRECTIONS

Position = tuple[int, int]


# QuickQudits 1.0.1 has qutrit circuits and measurement sampling but no
# U_H302 gate. This one-qutrit involution acts only on levels 0 and 2.
_H302_MATRIX = np.array(((1 / np.sqrt(2), 0, 1 / np.sqrt(2)),
                         (0, 1, 0),
                         (1 / np.sqrt(2), 0, -1 / np.sqrt(2))), dtype=complex)


def _h302_matrix(dimension: int) -> np.ndarray:
    if dimension != 3:
        raise ValueError("U_H302 requires a qutrit")
    return _H302_MATRIX


U_H302 = Gate("U_H302", 1, _h302_matrix, _h302_matrix)


@dataclass
class Grid:
    n: int
    body: list[Position]
    apples: set[Position] = field(default_factory=set)
    score: float = 0.0
    alive: bool = True

    def __post_init__(self):
        if self.n < 3 or not self.body or len(set(self.body)) != len(self.body):
            raise ValueError("invalid grid or body")
        if set(self.body) & self.apples:
            raise ValueError("apple overlaps snake")

    @classmethod
    def initial(cls, n: int, rng: Random) -> "Grid":
        grid = cls(n, [divmod(rng.randrange(n * n), n)])
        grid.prepare_apples(rng)
        return grid

    def digits(self) -> tuple[int, ...]:
        cells = [0] * (self.n * self.n)
        for r, c in self.apples:
            cells[r * self.n + c] = 2
        for r, c in self.body:
            cells[r * self.n + c] = 1
        return tuple(cells)

    def prepare_apples(self, rng: Random) -> None:
        """Independently prepare and measure every currently empty cell."""
        empty = [divmod(i, self.n) for i, value in enumerate(self.digits()) if value == 0]
        circuit = qq.QuantumCircuit(1, 3)
        circuit.append(U_H302, 0)
        for position in empty:
            _, measured, _ = circuit.sample(1, seed=rng.getrandbits(64))
            outcome = int(measured[0, 0])
            if outcome == 2:
                self.apples.add(position)
            elif outcome != 0:
                raise RuntimeError(f"U_H302 produced invalid apple outcome {outcome}")

    def move(self, direction: int | None, rng: Random) -> None:
        if direction is None or not self.alive:
            return
        dr, dc = DIRECTIONS[direction]
        r, c = self.body[0]
        next_cell = ((r + dr) % self.n, (c + dc) % self.n)
        eating = next_cell in self.apples
        occupied = self.body if eating else self.body[:-1]
        if next_cell in occupied:
            self.alive = False
            return
        self.body.insert(0, next_cell)
        if eating:
            self.apples.remove(next_cell)
            self.score += 1
        else:
            self.body.pop()
        if not self.apples:
            self.prepare_apples(rng)
