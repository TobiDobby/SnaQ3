import csv
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import quickqudits as qq

from snaq3.benchmark import replay_runtime_workload
from snaq3.grid import Grid
from snaq3.scaling_benchmark import (
    BACKEND_ORDER, RAW_FIELDS, ScalingConfig, _worker, dimensions,
    memory_estimate, run_scaling, scaling_workload, summarize,
)
from snaq3.simulation import BACKENDS


class ScalingBenchmarkTests(unittest.TestCase):
    def test_n_to_m_and_full_backend_registers(self):
        workload = scaling_workload(3, 1, 33, 0)
        for name in BACKEND_ORDER:
            m, systems = dimensions(3, name)
            self.assertEqual(m, 9)
            self.assertEqual(systems, 18 if name.startswith("qubit_") else 9)
            grids = replay_runtime_workload(BACKENDS[name](), workload)
            self.assertEqual(len(grids), 2)
            for player, grid in enumerate(grids):
                actual = grid.tableau.n if hasattr(grid, "tableau") else grid.circuit.n
                self.assertEqual(actual, systems)
                self.assertEqual(tuple(grid.digits), workload.targets[-1][player])
        self.assertEqual(dimensions(60, "qutrit_tableau_clifford"), (3600, 3600))
        self.assertEqual(dimensions(60, "qubit_tableau_transpiled"), (3600, 7200))

    def test_common_n3_workload_is_deterministic_and_paired(self):
        a = scaling_workload(3, 3, 77, 0)
        b = scaling_workload(3, 3, 77, 0)
        c = scaling_workload(3, 3, 77, 1)
        self.assertEqual(a, b)
        self.assertNotEqual(a.resolver_seeds, c.resolver_seeds)
        self.assertEqual(len(a.initial_grids), 2)
        self.assertEqual(len(a.targets), 3)
        self.assertTrue(all(value in (0, 1, 2) for grid in a.initial_grids for value in grid))

    def test_runner_reuses_one_trace_for_common_n(self):
        with tempfile.TemporaryDirectory() as directory:
            config = ScalingConfig(qutrit_sv_n=(3,), qutrit_tableau_n=(3,),
                                   qubit_sv_n=(3,), qubit_tableau_n=(3,), rounds=1,
                                   dense_repetitions=1, tableau_repetitions=1,
                                   output_directory=Path(directory))
            seen = []
            def measure(backend, workload, config, systems):
                seen.append((backend, id(workload), workload.digest()))
                return {"status": "COMPLETED", "runtime_seconds": .01}
            with patch("snaq3.scaling_benchmark._measure", side_effect=measure):
                rows, _ = run_scaling(config)
            self.assertEqual(len(rows), 4)
            self.assertEqual({item[0] for item in seen}, set(BACKEND_ORDER))
            self.assertEqual(len({item[1] for item in seen}), 1)
            self.assertEqual(len({item[2] for item in seen}), 1)

    def test_synthetic_n2_is_local_to_scaling_workload(self):
        config = ScalingConfig(qubit_sv_n=(2, 3), rounds=1)
        config.validate()
        workload = scaling_workload(2, 4, 80, 0)
        self.assertEqual(len(workload.initial_grids[0]), 4)
        self.assertTrue(workload.seeds["synthetic_scaling_point"])
        grids = replay_runtime_workload(BACKENDS["qubit_transpiled"](), workload)
        self.assertEqual([grid.circuit.n for grid in grids], [8, 8])
        with self.assertRaises(ValueError):
            Grid.initial(2, __import__("random").Random(1))
        with self.assertRaises(ValueError):
            dimensions(2, "qutrit_circuit_exact")

    def test_dense_oom_no_allocation_and_memory_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            config = ScalingConfig(qutrit_sv_n=(5, 6), qutrit_tableau_n=(),
                                   qubit_sv_n=(), qubit_tableau_n=(), rounds=1,
                                   dense_repetitions=2, max_memory_gib=1,
                                   output_directory=Path(directory))
            with patch("snaq3.scaling_benchmark._measure", side_effect=AssertionError("allocated")):
                rows, _ = run_scaling(config)
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(row["status"] == "OOM" for row in rows))
            self.assertTrue(all(row["runtime_seconds"] == "" for row in rows))
            self.assertEqual({row["error"] for row in rows if row["n"] == 6},
                             {"monotonic memory bound after previous OOM"})
            self.assertEqual(int(rows[0]["vector_bytes_per_player"]), 16 * 3**25)
            self.assertEqual(memory_estimate("qubit_transpiled", 16)[0], 16 * 2**32)

    def test_tableau_memory_model_matches_quickqudits(self):
        for name, systems in (("qutrit_tableau_clifford", 9),
                              ("qubit_tableau_transpiled", 18)):
            tableau = qq.Tableau(systems, 3 if name.startswith("qutrit") else 2)
            arrays = tableau.X.nbytes + tableau.Z.nbytes + tableau.tau_exp.nbytes
            self.assertEqual(memory_estimate(name, 9), ("", 8 * arrays))

    def test_resume_retry_and_raw_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            config = ScalingConfig(qutrit_sv_n=(), qutrit_tableau_n=(3,),
                                   qubit_sv_n=(), qubit_tableau_n=(), rounds=1,
                                   tableau_repetitions=1, output_directory=Path(directory))
            with patch("snaq3.scaling_benchmark._measure", return_value={"status": "TIMEOUT", "error": "limit"}) as measure:
                rows, _ = run_scaling(config)
                self.assertEqual(measure.call_count, 1)
            self.assertEqual(rows[0]["status"], "TIMEOUT")
            with patch("snaq3.scaling_benchmark._measure", side_effect=AssertionError("reran")):
                rows, _ = run_scaling(replace(config, resume=True))
            self.assertEqual(len(rows), 1)
            with patch("snaq3.scaling_benchmark._measure", return_value={"status": "COMPLETED", "runtime_seconds": .2}) as measure:
                rows, summary = run_scaling(replace(config, resume=True, retry_failed=True,
                                                    timeout_seconds=42))
                self.assertEqual(measure.call_count, 1)
            self.assertEqual(summary[0]["mean_runtime_seconds"], .2)
            with (Path(directory) / "runtime_scaling_raw.csv").open(newline="") as file:
                reader = csv.DictReader(file)
                self.assertEqual(tuple(reader.fieldnames), RAW_FIELDS)
                self.assertEqual(len(list(reader)), 1)

    def test_timeout_alarm_is_distinct_from_oom(self):
        class Sender:
            message = None
            def send(self, value):
                self.message = value
            def close(self):
                pass
        sender = Sender()
        calls = 0
        def replay(*args):
            nonlocal calls
            calls += 1
            if calls > 1:
                time.sleep(.08)
            return []
        with patch("snaq3.scaling_benchmark.replay_runtime_workload", side_effect=replay):
            _worker(sender, "qutrit_tableau_clifford", scaling_workload(3, 1, 3, 0),
                    32, .01, 9)
        self.assertEqual(sender.message["status"], "TIMEOUT")

    def test_interruption_leaves_checkpoint_and_partial_plot(self):
        with tempfile.TemporaryDirectory() as directory:
            config = ScalingConfig(qutrit_sv_n=(3, 4), qutrit_tableau_n=(),
                                   qubit_sv_n=(), qubit_tableau_n=(), rounds=1,
                                   dense_repetitions=2, output_directory=Path(directory))
            with patch("snaq3.scaling_benchmark._measure", side_effect=[
                {"status": "COMPLETED", "runtime_seconds": .1}, KeyboardInterrupt]):
                rows, summary = run_scaling(config)
            self.assertEqual(len(rows), 1)
            self.assertGreater(float(rows[0]["runtime_seconds"]), 0)
            with (Path(directory) / "runtime_scaling_raw.csv").open(newline="") as file:
                self.assertEqual(len(list(csv.DictReader(file))), 1)
            self.assertEqual(summary[0]["completed_repetitions"], 1)
            self.assertTrue((Path(directory) / "runtime_vs_grid_size.pdf").exists())
            self.assertTrue((Path(directory) / "runtime_vs_grid_size.png").exists())

    def test_summary_ignores_failures_and_missing_backend_sizes_plot(self):
        common = {"backend": "qutrit_circuit_exact", "n": 3,
                  "estimated_working_set_bytes": 100}
        rows = [{**common, "status": "COMPLETED", "runtime_seconds": 1.0},
                {**common, "status": "COMPLETED", "runtime_seconds": 3.0},
                {**common, "status": "OOM", "runtime_seconds": ""},
                {**common, "status": "TIMEOUT", "runtime_seconds": ""}]
        item = summarize(rows, 42)[0]
        self.assertEqual(item["mean_runtime_seconds"], 2.0)
        self.assertEqual(item["median_runtime_seconds"], 2.0)
        self.assertEqual(item["completed_repetitions"], 2)
        self.assertEqual(item["oom_repetitions"], 1)
        self.assertEqual(item["timeout_repetitions"], 1)
        self.assertLessEqual(item["CI95_low"], item["mean_runtime_seconds"])
        self.assertGreaterEqual(item["CI95_high"], item["mean_runtime_seconds"])
        from snaq3.scaling_plotting import plot_scaling
        with tempfile.TemporaryDirectory() as directory:
            plot_scaling(Path(directory), [item, {"backend": "qubit_transpiled", "n": 4,
                "grid_cells_per_player": 16, "mean_runtime_seconds": "", "CI95_low": "",
                "CI95_high": "", "completed_repetitions": 0, "oom_repetitions": 1,
                "timeout_repetitions": 0, "synthetic_scaling_point": False}])
            self.assertTrue((Path(directory) / "runtime_vs_grid_size.pdf").exists())

    def test_plot_does_not_connect_across_oom_gap(self):
        from matplotlib.axes import Axes
        from snaq3.scaling_plotting import plot_scaling
        source = [
            {"backend": "qutrit_circuit_exact", "n": n,
             "grid_cells_per_player": n*n,
             "mean_runtime_seconds": value, "CI95_low": "", "CI95_high": "",
             "completed_repetitions": 1 if value != "" else 0,
             "oom_repetitions": 1 if value == "" else 0,
             "timeout_repetitions": 0, "synthetic_scaling_point": False}
            for n, value in ((3, .01), (4, ""), (5, .1))]
        original = Axes.plot
        styles = []
        def spy(axis, *args, **kwargs):
            styles.append(kwargs.get("linestyle"))
            return original(axis, *args, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(Axes, "plot", spy):
                plot_scaling(Path(directory), source)
        self.assertTrue(styles)
        self.assertTrue(all(style == "none" for style in styles))


if __name__ == "__main__":
    unittest.main()
