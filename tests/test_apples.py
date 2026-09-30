"""Gameplay apple preparation and benchmark exclusion checks."""

import unittest
from random import Random
from unittest.mock import patch

import numpy as np
import quickqudits as qq

from snaq3.benchmark import runtime_workload
from snaq3.grid import Grid, U_H302


class AppleRulesTests(unittest.TestCase):
    def test_h302_is_a_qutrit_unitary_with_requested_action(self):
        matrix = U_H302.matrix(3)
        np.testing.assert_allclose(matrix.conj().T @ matrix, np.eye(3), atol=1e-15)
        np.testing.assert_allclose(matrix[:, 0], [2**-.5, 0, 2**-.5])
        np.testing.assert_allclose(matrix[:, 1], [0, 1, 0])

    def test_only_empty_cells_are_prepared_and_snake_is_untouched(self):
        grid = Grid(3, [(0, 0), (0, 1)], {(0, 2)})
        original_body = list(grid.body)
        calls = []

        def sample(circuit, shots, **kwargs):
            calls.append((circuit.n, circuit.d, tuple(gate.name for gate, _, _ in circuit.ops),
                          shots, kwargs.get("seed")))
            return None, np.array([[2]]), None

        with patch.object(qq.QuantumCircuit, "sample", sample):
            grid.prepare_apples(Random(13))
        self.assertEqual(len(calls), 6)
        self.assertTrue(all(call[:4] == (1, 3, ("U_H302",), 1) for call in calls))
        self.assertEqual(grid.body, original_body)
        self.assertEqual(grid.digits()[:3], (1, 1, 2))
        self.assertEqual(len(grid.apples), 7)  # existing apple plus six empty cells
        self.assertTrue(all(value == 1 for value in grid.digits()[:2]))

    def test_native_measurements_are_binary_zero_or_two_and_near_half(self):
        # Eight classical snake cells leave one independently prepared qutrit.
        body = [divmod(i, 3) for i in range(8)]
        rng = Random(151)
        outcomes = []
        for _ in range(1000):
            grid = Grid(3, list(body))
            grid.prepare_apples(rng)
            outcomes.append(grid.digits()[-1])
            self.assertEqual(grid.digits()[:8], (1,) * 8)
        self.assertEqual(set(outcomes), {0, 2})
        self.assertLess(abs(outcomes.count(2) / len(outcomes) - .5), .05)

    def test_empty_cells_are_sampled_independently(self):
        body = [divmod(i, 3) for i in range(6)]
        rng = Random(1901)
        patterns = []
        for _ in range(1000):
            grid = Grid(3, list(body))
            grid.prepare_apples(rng)
            patterns.append(tuple(value == 2 for value in grid.digits()[-3:]))
        for cell in range(3):
            self.assertLess(abs(sum(row[cell] for row in patterns) / len(patterns) - .5), .05)
        self.assertLess(abs(sum(row[0] and row[1] for row in patterns) / len(patterns) - .25), .05)

    def test_no_respawn_until_last_apple_is_consumed(self):
        grid = Grid(3, [(0, 0)], {(0, 1), (0, 2)})
        original = Grid.prepare_apples
        calls = []

        def record(self, rng):
            calls.append((tuple(self.body), tuple(self.apples)))
            return original(self, rng)

        with patch.object(Grid, "prepare_apples", record):
            grid.move(1, Random(7))
            self.assertEqual(calls, [])
            self.assertEqual(grid.apples, {(0, 2)})
            self.assertEqual(grid.score, 1)
            self.assertEqual(len(grid.body), 2)
            grid.move(1, Random(7))
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], ())
        self.assertEqual(grid.score, 2)
        self.assertEqual(len(grid.body), 3)
        self.assertTrue(all(grid.digits()[r * 3 + c] == 1 for r, c in grid.body))

    def test_zero_apple_initial_batch_is_retried_after_next_move(self):
        grid = Grid(3, [(0, 0)])
        outcomes = iter([0] * 8 + [2] * 8)

        def sample(circuit, shots, **kwargs):
            return None, np.array([[next(outcomes)]]), None

        with patch.object(qq.QuantumCircuit, "sample", sample):
            grid.prepare_apples(Random(1))
            self.assertFalse(grid.apples)
            grid.move(1, Random(2))
        self.assertEqual(len(grid.apples), 8)
        self.assertEqual(len(grid.body), 1)
        self.assertEqual(grid.score, 0)

    def test_initial_apple_distribution_is_seed_reproducible(self):
        first = Grid.initial(4, Random(2026))
        again = Grid.initial(4, Random(2026))
        self.assertEqual((first.body, first.apples), (again.body, again.apples))
        self.assertTrue(set(first.body).isdisjoint(first.apples))

    def test_fixed_runtime_trace_excludes_quantum_apple_preparation(self):
        with patch.object(qq.QuantumCircuit, "sample", side_effect=AssertionError("quantum apple timing")):
            workload = runtime_workload(9, 3, 72, 0)
        self.assertEqual(len(workload.initial_grids), 2)
        self.assertEqual(len(workload.targets), 3)
        self.assertEqual(sum(v == 2 for v in workload.initial_grids[0]), 1)
        self.assertEqual(sum(v == 2 for v in workload.initial_grids[1]), 1)
