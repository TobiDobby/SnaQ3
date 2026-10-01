import unittest
from math import pi
from random import Random

import quickqudits as qq

from snaq3.grid import Grid
from collections import Counter
from unittest.mock import patch

from snaq3.benchmark import (EXACT_PAIR, TABLEAU_PAIR, _check_pair_agreement,
                             payoff_samples, replay_runtime_workload,
                             runtime_workload, run_payoff_experiment,
                             run_runtime_experiment)
from snaq3.movement import DIRECTIONS, DIRECTION_NAMES
from snaq3.payoff import PAYOFF, movement_policy, utility, wins
from snaq3.resolver import (P_QUANTUM, clifford_mapping, exact_probabilities,
                            tableau_circuit)
from snaq3.simulation import BACKENDS
from snaq3.simulation.qubit_transpiled import EncodedGrid
from snaq3.simulation.qubit_tableau_transpiled import EncodedTableauGrid
from snaq3.simulation.encoding import append_encoded_x
from snaq3.simulation.qutrit_circuit import CircuitGrid
from snaq3.simulation.qutrit_tableau import TableauGrid


class FinalModelTests(unittest.TestCase):
    def test_payoff_table_and_policy(self):
        for mask in range(4):
            self.assertEqual([utility(mask, 3, a, b) for a, b in
                              ((0, 0), (0, 1), (1, 0), (1, 1))],
                             [row[mask] for row in PAYOFF])
        for x in range(4):
            for y in range(4):
                for a in range(2):
                    for b in range(2):
                        self.assertEqual(sum(utility(x, y, a, b)),
                                         1.5 if wins(x, y, a, b) else 0)
        self.assertEqual(movement_policy(1, 2, (1, .5)), (1, 1))
        self.assertEqual(movement_policy(1, 2, (.5, 1)), (2, 2))
        self.assertEqual(movement_policy(1, 2, (.75, .75)), (1, 2))
        self.assertEqual(movement_policy(1, 2, (0, 0)), (None, None))

    def test_classical_direction_labels(self):
        self.assertEqual(DIRECTION_NAMES, ("UP", "RIGHT", "DOWN", "LEFT"))
        self.assertEqual(DIRECTIONS, ((-1, 0), (0, 1), (1, 0), (0, -1)))

    def test_analytical_exact_and_classical(self):
        p = f_a = f_b = 0
        for x in range(4):
            for y in range(4):
                probs = exact_probabilities(x, y)
                self.assertAlmostEqual(float(probs.sum()), 1)
                for a in range(2):
                    for b in range(2):
                        weight = probs[a, b] / 16
                        p += weight * wins(x, y, a, b)
                        ua, ub = utility(x, y, a, b)
                        f_a += weight * ua
                        f_b += weight * ub
        self.assertAlmostEqual(p, P_QUANTUM, places=12)
        self.assertAlmostEqual((f_a + f_b) / 2, .75 * P_QUANTUM, places=12)
        # A fixed deterministic local strategy wins 12 of 16 uniform inputs.
        classical = sum(wins(x, y, 0, 0) for x in range(4) for y in range(4)) / 16
        self.assertEqual(classical, .75)
        self.assertEqual(.75 * classical, 9 / 16)

    def test_tableau_mapping_and_clifford_only(self):
        mapping = clifford_mapping()
        self.assertEqual(mapping["Bob_0"]["implemented_theta"], 0)
        self.assertAlmostEqual(mapping["Bob_1"]["angular_error"], pi / 8)
        for x in range(4):
            for y in range(4):
                self.assertTrue(all(gate.name in {"H", "CNOT", "X"}
                                    for gate, _, _ in tableau_circuit(x, y).ops))
        # For Z/X versus Z/Z, exact tableau win probability is 3/4.
        from snaq3.resolver import tableau_sample
        rng = Random(12)
        wins_count = sum(wins(x, y, *tableau_sample(x, y, rng.getrandbits(64)))
                         for x in range(4) for y in range(4) for _ in range(100))
        self.assertLess(abs(wins_count / 1600 - .75), .05)

    def test_encoding_preserves_subspace(self):
        for value in range(4):
            circuit = qq.QuantumCircuit(2, 2)
            if value & 2:
                circuit.X(0)
            if value & 1:
                circuit.X(1)
            append_encoded_x(circuit, 0)
            self.assertTrue(all(gate.name in {"X", "CNOT"}
                                for gate, _, _ in circuit.ops))
            tableau = qq.Tableau(2, 2)
            tableau.apply_circuit(circuit, engine="python")
            bits = tableau.sample_measurements(1, seed=7)[0]
            output = int(bits[0]) * 2 + int(bits[1])
            self.assertEqual(output, (value + 1) % 3 if value < 3 else 3)
        grid = EncodedGrid((0, 1, 2))
        self.assertEqual(grid.circuit.n, 6)
        self.assertEqual(len(grid.state), 2**6)
        grid.set_digits((2, 0, 1))
        self.assertEqual(grid.digits, [2, 0, 1])
        encoded_index = int("100001", 2)
        self.assertAlmostEqual(abs(grid.state[encoded_index]), 1)

    def test_encoded_tableau_grid_never_uses_invalid_cell(self):
        digits = tuple((i % 3) for i in range(25))
        grid = EncodedTableauGrid(digits)
        self.assertEqual(grid.tableau.n, 50)
        for desired in (digits, tuple((x + 1) % 3 for x in digits)):
            grid.set_digits(desired)
            bits = tuple(map(int, grid.tableau.sample_measurements(1, seed=3)[0]))
            decoded = tuple(2 * bits[2*i] + bits[2*i+1] for i in range(25))
            self.assertEqual(decoded, desired)
            self.assertNotIn(3, decoded)

    def test_full_qutrit_circuit_grid(self):
        digits = (0, 1, 2, 0, 2, 1, 0, 0, 1)
        grid = CircuitGrid(digits)
        self.assertEqual(grid.circuit.n, 9)
        self.assertEqual(len(grid.state), 3**9)
        expected = sum(value * 3**(8-i) for i, value in enumerate(digits))
        self.assertAlmostEqual(abs(grid.state[expected]), 1)
        changed = (2, 0, 1, 0, 1, 2, 0, 0, 0)
        grid.set_digits(changed)
        expected = sum(value * 3**(8-i) for i, value in enumerate(changed))
        self.assertAlmostEqual(abs(grid.state[expected]), 1)

    def test_dense_memory_failures_are_explicit(self):
        with self.assertRaisesRegex(MemoryError, "two dense grids need"):
            CircuitGrid((0,) * 25)
        with self.assertRaisesRegex(MemoryError, "two dense grids need"):
            EncodedGrid((0,) * 16)

    def test_full_tableau_grid_has_basis_occupation(self):
        digits = (0, 1, 2, 0, 2, 1, 0, 0, 1)
        grid = TableauGrid(digits)
        self.assertEqual(tuple(grid.tableau.sample_measurements(1, seed=7)[0]), digits)
        changed = (2, 0, 1, 0, 1, 2, 0, 0, 0)
        grid.set_digits(changed)
        self.assertEqual(tuple(grid.tableau.sample_measurements(1, seed=8)[0]), changed)

    def test_grid_gates_do_not_entangle_distinct_cells(self):
        digits = (1, 2, 0, 1, 2, 0, 1, 2, 0)
        native = CircuitGrid(digits)
        encoded = EncodedGrid(digits)
        self.assertTrue(all(gate.name == "X" and len(targets) == 1
                            for gate, targets, _ in native.circuit.ops))
        self.assertTrue(all(len(targets) == 1 or targets[0] // 2 == targets[1] // 2
                            for _, targets, _ in encoded.circuit.ops))
        # Tableau grids use the same native X and encoded-X circuit builders.

    def test_grid_and_backend_agreement(self):
        grid = Grid(3, [(0, 0)], {(0, 1)})
        grid.move(1, Random(2))
        self.assertEqual(grid.body, [(0, 1), (0, 0)])
        self.assertEqual(grid.score, 1)
        self.assertEqual(grid.digits()[1], 1)
        inputs = [(x, y) for x in range(4) for y in range(4)] * 2
        exact = BACKENDS["qutrit_circuit_exact"]().run(3, inputs, 22)
        encoded = BACKENDS["qubit_transpiled"]().run(3, inputs, 22)
        self.assertEqual((exact.F_A, exact.F_B, exact.p_win, exact.metadata["scores"]),
                         (encoded.F_A, encoded.F_B, encoded.p_win, encoded.metadata["scores"]))
        self.assertEqual(exact.metadata["grid_count"], 2)

    def test_clifford_tableaus_agree_on_matched_inputs(self):
        inputs = [(x, y) for x in range(4) for y in range(4)]
        for n in (3, 4, 5):
            qutrit = BACKENDS["qutrit_tableau_clifford"]().run(n, inputs, 91)
            encoded = BACKENDS["qubit_tableau_transpiled"]().run(n, inputs, 91)
            self.assertEqual((qutrit.F_A, qutrit.F_B, qutrit.F_avg, qutrit.p_win),
                             (encoded.F_A, encoded.F_B, encoded.F_avg, encoded.p_win))
            self.assertEqual(qutrit.metadata["scores"], encoded.metadata["scores"])

    def test_benchmark_rejects_paired_disagreement(self):
        with self.assertRaisesRegex(AssertionError, "paired resolver disagreement"):
            _check_pair_agreement({TABLEAU_PAIR[0]: (0, 0), TABLEAU_PAIR[1]: (1, 0)},
                                  TABLEAU_PAIR)

    def test_legacy_tableau_movement_is_not_used_by_gameplay(self):
        from snaq3.simulation.qutrit_tableau import QutritTableau
        from snaq3.simulation.qubit_tableau_transpiled import QubitTableauTranspiled
        for cls in (QutritTableau, QubitTableauTranspiled):
            backend = cls()
            for x in range(4):
                self.assertIsInstance(backend.prepare_movement(x), qq.Tableau)
            with patch.object(backend, "prepare_movement", side_effect=AssertionError("unused legacy path")):
                backend.run(3, [(0, 0)], 71)

    def test_gameplay_passes_classical_directions_and_updates_qutrit_grid(self):
        from snaq3.simulation.qutrit_circuit import QutritCircuit
        first = Grid(3, [(1, 1)], {(0, 0)})
        second = Grid(3, [(1, 1)], {(0, 0)})
        backend = QutritCircuit()
        backend.prepare_movement = unittest.mock.Mock(side_effect=AssertionError("unused movement state"))
        original_move = Grid.move
        moves = []
        registers = []

        def record_move(grid, direction, rng):
            moves.append(direction)
            return original_move(grid, direction, rng)

        def make_grid(digits):
            register = CircuitGrid(digits)
            registers.append(register)
            return register

        with patch.object(Grid, "initial", side_effect=[first, second]), \
             patch.object(Grid, "move", record_move), \
             patch.object(backend, "resolve", return_value=(1, 0)) as resolve, \
             patch.object(backend, "make_grid", side_effect=make_grid):
            result = backend.run(3, [(1, 3)], 71)
        self.assertEqual(resolve.call_args.args[:2], (1, 3))
        self.assertEqual(moves, [1, 3])
        self.assertEqual(result.metadata["rounds_executed"], 1)
        for register, grid in zip(registers, (first, second)):
            digits = grid.digits()
            self.assertEqual(register.digits, list(digits))
            basis = sum(value * 3**(8-i) for i, value in enumerate(digits))
            self.assertAlmostEqual(abs(register.state[basis]), 1)
        self.assertEqual(first.body[0], (1, 2))
        self.assertEqual(second.body[0], (1, 0))

    def test_runtime_replay_uses_classical_resolver_inputs_only(self):
        from snaq3.simulation.qutrit_circuit import QutritCircuit
        workload = runtime_workload(9, 2, 52, 0)
        backend = QutritCircuit()
        backend.prepare_movement = unittest.mock.Mock(side_effect=AssertionError("unused movement state"))
        with patch.object(backend, "resolve", return_value=(0, 0)) as resolve:
            grids = replay_runtime_workload(backend, workload)
        self.assertEqual([call.args for call in resolve.call_args_list],
                         [(x_a, x_b, seed) for (x_a, x_b), seed in
                          zip(workload.inputs, workload.resolver_seeds)])
        self.assertEqual(tuple(tuple(grid.digits) for grid in grids), workload.targets[-1])

    def test_paper_payoff_plot_accepts_exact_statevectors_only(self):
        import csv
        import tempfile
        from pathlib import Path
        from snaq3.plotting import plot_payoff

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with (output / "payoff_summary.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=("backend", "mean_F_avg",
                                                           "ci95_low_F_avg", "ci95_high_F_avg"))
                writer.writeheader()
                for backend in EXACT_PAIR:
                    writer.writerow({"backend": backend, "mean_F_avg": .64,
                                     "ci95_low_F_avg": .63, "ci95_high_F_avg": .65})
            plot_payoff(output)
            self.assertTrue((output / "payoff_by_backend.pdf").exists())
            self.assertTrue((output / "payoff_by_backend.png").exists())

    def test_tableau_runtime_has_no_dense_execution_or_sampling(self):
        workload = runtime_workload(9, 2, 105, 0)
        with patch.object(qq.QuantumCircuit, "execute", side_effect=AssertionError("dense execute")), \
             patch.object(qq.QuantumCircuit, "sample", side_effect=AssertionError("dense sample")):
            for name in TABLEAU_PAIR:
                replay_runtime_workload(BACKENDS[name](), workload)

    def test_balanced_inputs_and_independent_seeds(self):
        inputs, seeds, derivation = payoff_samples(3, 42, 0)
        self.assertEqual(Counter(inputs), Counter({(x, y): 3 for x in range(4) for y in range(4)}))
        self.assertEqual(len(seeds), 48)
        self.assertNotEqual(derivation["input_seed"], derivation["resolver_seed"])
        self.assertEqual((inputs, seeds), payoff_samples(3, 42, 0)[:2])

    def test_all_payoff_backends_receive_identical_pairs_and_seeds(self):
        import tempfile
        from snaq3 import benchmark
        calls = {}

        def backend_class(name):
            class Spy:
                def __init__(self):
                    calls[name] = []

                def resolve(self, x, y, seed):
                    calls[name].append((x, y, seed))
                    return (0, 0)

                def make_grid(self, digits):
                    raise AssertionError("payoff experiment allocated a grid")
            return Spy

        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(benchmark.BACKENDS, {name: backend_class(name)
                                                for name in BACKENDS}, clear=True):
                run_payoff_experiment(2, 1, 83, directory)
        self.assertTrue(all(calls[name] == calls[EXACT_PAIR[0]] for name in calls))
        self.assertEqual(len(calls[EXACT_PAIR[0]]), 32)

    def test_random_head_and_death_terminate_game(self):
        from snaq3.model import run_game
        self.assertEqual(Grid.initial(3, Random(11)).body, Grid.initial(3, Random(11)).body)
        self.assertNotEqual(Grid.initial(3, Random(1)).body, Grid.initial(3, Random(2)).body)

        class FatalGrid:
            def __init__(self, digits):
                self.digits = digits
            def set_digits(self, digits):
                self.digits = digits

        class Stub:
            metadata = {}
            def make_grid(self, digits):
                return FatalGrid(digits)
            def resolve(self, x, y, seed):
                return (0, 0)

        # Controlled collision through a Grid.move patch verifies early termination.
        def die(self, direction, rng):
            self.alive = False
        with patch.object(Grid, "move", die):
            result = run_game(Stub(), 3, [(0, 0)] * 4, 11)
        self.assertEqual(result.metadata["rounds_executed"], 1)

    def test_runtime_trace_is_shared_and_full_size(self):
        workload = runtime_workload(9, 3, 52, 0)
        self.assertEqual(workload.digest(), runtime_workload(9, 3, 52, 0).digest())
        self.assertEqual(len(workload.inputs), 3)
        self.assertEqual(len(workload.resolver_seeds), 3)
        for name, cls in BACKENDS.items():
            grids = replay_runtime_workload(cls(), workload)
            self.assertEqual(len(grids), 2)
            self.assertEqual(tuple(tuple(grid.digits) for grid in grids), workload.targets[-1])
            systems = grids[0].tableau.n if hasattr(grids[0], "tableau") else grids[0].circuit.n
            self.assertEqual(systems, 18 if name.startswith("qubit") else 9)

    def test_benchmark_pairing_and_oom_status(self):
        import tempfile
        from snaq3.simulation import _memory
        previous_limit = _memory.MAX_WORKING_SET_BYTES
        self.addCleanup(setattr, _memory, "MAX_WORKING_SET_BYTES", previous_limit)
        with tempfile.TemporaryDirectory() as directory:
            rows, _ = run_payoff_experiment(2, 2, 77, directory)
            for repetition in range(2):
                selected = {r["backend"]: r for r in rows if r["repetition"] == repetition}
                for pair in (EXACT_PAIR, TABLEAU_PAIR):
                    self.assertEqual(tuple(selected[pair[0]][metric] for metric in
                                           ("F_A", "F_B", "F_avg", "p_win")),
                                     tuple(selected[pair[1]][metric] for metric in
                                           ("F_A", "F_B", "F_avg", "p_win")))
            rows, _ = run_runtime_experiment((9,), 1, 1, 1, 77, directory, .001)
            self.assertEqual({r["workload_hash"] for r in rows}, {runtime_workload(9, 1, 77, 0).digest()})
            self.assertTrue(any(r["status"] == "OOM" for r in rows))
            self.assertTrue(all(r["runtime_seconds"] == "" for r in rows if r["status"] == "OOM"))


if __name__ == "__main__":
    unittest.main()
