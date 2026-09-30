"""Common backend interface and round orchestration."""

from dataclasses import dataclass, field
from random import Random
from typing import Protocol

from .grid import Grid
from .movement import PHASES, phase_to_input, prepare_movement
from .payoff import movement_policy, utility, wins


@dataclass(frozen=True)
class SimulationResult:
    F_A: float
    F_B: float
    F_avg: float
    p_win: float
    metadata: dict = field(default_factory=dict)


class SimulationBackend(Protocol):
    name: str

    def run(self, n: int, inputs: list[tuple[int, int]], seed: int) -> SimulationResult: ...


def run_game(backend, n: int, inputs: list[tuple[int, int]], seed: int) -> SimulationResult:
    if not inputs:
        raise ValueError("at least one round required")
    rng = Random(seed)
    resolver_rng = Random(seed ^ 0xA1B2C3D4E5F60718)
    grids = [Grid.initial(n, rng), Grid.initial(n, rng)]
    registers = [backend.make_grid(grid.digits()) for grid in grids]
    totals = [0.0, 0.0]
    successes = 0
    rounds = 0
    for x_a, x_b in inputs:
        movement_preparer = getattr(backend, "prepare_movement", prepare_movement)
        movement_preparer(x_a)
        movement_preparer(x_b)
        assert phase_to_input(PHASES[x_a]) == x_a
        assert phase_to_input(PHASES[x_b]) == x_b
        a, b = backend.resolve(x_a, x_b, resolver_rng.getrandbits(64))
        payoff = utility(x_a, x_b, a, b)
        successes += wins(x_a, x_b, a, b)
        moves = movement_policy(x_a, x_b, payoff)
        for i in range(2):
            totals[i] += payoff[i]
            grids[i].score += payoff[i]
            grids[i].move(moves[i], rng)
            registers[i].set_digits(grids[i].digits())
        rounds += 1
        if not all(grid.alive for grid in grids):
            break
    return SimulationResult(totals[0] / rounds, totals[1] / rounds,
                            sum(totals) / (2 * rounds), successes / rounds,
                            {"grid_cells_per_player": n * n, "grid_count": 2,
                             "scores": [grid.score for grid in grids],
                             "alive": [grid.alive for grid in grids],
                             "rounds_executed": rounds,
                             **backend.metadata})
