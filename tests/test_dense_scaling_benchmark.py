import csv
import json
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from snaq3.benchmark import replay_runtime_workload, runtime_workload
from snaq3.dense_scaling_benchmark import (
    DenseScalingConfig, RAW_FIELDS, run_dense_scaling, summarize, synthetic_workload,
)
from snaq3.scaling_plotting import COLORS, merged_rows, plot_scaling
import snaq3.scaling_plotting as scaling_plotting
from snaq3.simulation import BACKENDS


class DenseScalingTests(unittest.TestCase):
    def test_nonsquare_workload_is_synthetic_only_and_paired(self):
        a = synthetic_workload(6, 3, 33, 0)
        self.assertEqual(a, synthetic_workload(6, 3, 33, 0))
        self.assertNotEqual(a.digest(), synthetic_workload(6, 3, 33, 1).digest())
        with self.assertRaises(ValueError):
            runtime_workload(6, 1, 33, 0)
        self.assertEqual(len(a.initial_grids), 2)
        self.assertEqual(len(a.targets), 3)
        for states in (a.initial_grids, *a.targets):
            for state in states:
                self.assertEqual(len(state), 6)
                self.assertEqual(sorted(set(state)), [0, 1, 2])
                self.assertEqual(state.count(1), 1)
                self.assertEqual(state.count(2), 1)

    def test_full_dense_register_dimensions_and_same_logical_trace(self):
        workload = synthetic_workload(6, 1, 15, 0)
        for backend_name, expected in (("qutrit_circuit_exact", 6), ("qubit_transpiled", 12)):
            grids = replay_runtime_workload(BACKENDS[backend_name](), workload)
            self.assertEqual(len(grids), 2)
            self.assertEqual([grid.circuit.n for grid in grids], [expected, expected])
            self.assertEqual([tuple(grid.digits) for grid in grids], list(workload.targets[-1]))
            self.assertEqual([grid.state.size for grid in grids],
                             [3**6 if expected == 6 else 2**12] * 2)

    def test_runner_shares_one_trace_and_marks_synthetic(self):
        with tempfile.TemporaryDirectory() as directory:
            config = DenseScalingConfig(qutrit_m=(6,), qubit_m=(6,), rounds=1,
                                        repetitions=1, output_directory=Path(directory))
            seen = []
            def measure(backend, workload, config, systems):
                seen.append((backend, id(workload), workload.digest(), systems))
                return {"status": "COMPLETED", "runtime_seconds": .01}
            with patch("snaq3.dense_scaling_benchmark._measure", side_effect=measure):
                rows, _ = run_dense_scaling(config)
            self.assertEqual(len(rows), 2)
            self.assertEqual({item[1] for item in seen}, {seen[0][1]})
            self.assertEqual({item[2] for item in seen}, {seen[0][2]})
            self.assertEqual({item[3] for item in seen}, {6, 12})
            self.assertTrue(all(row["synthetic_scaling_point"] for row in rows))
            self.assertTrue(all(json.loads(row["backend_metadata"])["workload_sha256"] == seen[0][2]
                                for row in rows))
            with (Path(directory) / "dense_synthetic_raw.csv").open(newline="") as file:
                reader = csv.DictReader(file)
                self.assertEqual(tuple(reader.fieldnames), RAW_FIELDS)
                self.assertEqual(len(list(reader)), 2)

    def test_existing_full_grid_point_is_not_duplicated(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with (output / "runtime_scaling_raw.csv").open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=("backend", "grid_cells_per_player"))
                writer.writeheader()
                writer.writerow({"backend": "qutrit_circuit_exact", "grid_cells_per_player": 9})
            config = DenseScalingConfig(qutrit_m=(9,), qubit_m=(), rounds=1,
                                        repetitions=1, output_directory=output)
            with patch("snaq3.dense_scaling_benchmark._measure", side_effect=AssertionError("duplicated")):
                rows, _ = run_dense_scaling(config)
            self.assertEqual(rows, [])

    def test_oom_propagation_never_allocates(self):
        with tempfile.TemporaryDirectory() as directory:
            config = DenseScalingConfig(qutrit_m=(17, 18), qubit_m=(), rounds=1,
                                        repetitions=2, max_memory_gib=.01,
                                        output_directory=Path(directory))
            with patch("snaq3.dense_scaling_benchmark._measure", side_effect=AssertionError("allocated")):
                rows, summary = run_dense_scaling(config)
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(row["status"] == "OOM" and row["runtime_seconds"] == "" for row in rows))
            self.assertEqual({row["error"] for row in rows if row["m"] == 18},
                             {"monotonic memory bound after previous OOM"})
            self.assertEqual(rows[0]["vector_bytes_per_player"], 16 * 3**17)
            self.assertTrue(all(item["mean_runtime_seconds"] == "" for item in summary))

    def test_resume_retry_and_interrupted_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            config = DenseScalingConfig(qutrit_m=(4, 6), qubit_m=(), rounds=1,
                                        repetitions=1, output_directory=Path(directory))
            with patch("snaq3.dense_scaling_benchmark._measure", side_effect=[
                {"status": "TIMEOUT", "error": "limit"}, KeyboardInterrupt]):
                rows, _ = run_dense_scaling(config)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "TIMEOUT")
            with patch("snaq3.dense_scaling_benchmark._measure", return_value={"status": "COMPLETED", "runtime_seconds": .2}) as measure:
                rows, _ = run_dense_scaling(replace(config, resume=True))
                self.assertEqual(measure.call_count, 1)
            self.assertEqual([row["status"] for row in rows], ["TIMEOUT", "COMPLETED"])
            with patch("snaq3.dense_scaling_benchmark._measure", return_value={"status": "COMPLETED", "runtime_seconds": .3}) as measure:
                rows, summary = run_dense_scaling(replace(config, resume=True, retry_failed=True,
                                                          timeout_seconds=42))
                self.assertEqual(measure.call_count, 1)
            self.assertTrue(all(row["status"] == "COMPLETED" for row in rows))
            self.assertEqual(len(summary), 2)
            with (Path(directory) / "dense_synthetic_raw.csv").open(newline="") as file:
                self.assertEqual(len(list(csv.DictReader(file))), 2)
            with patch("snaq3.dense_scaling_benchmark._measure", side_effect=AssertionError("reran completed")):
                run_dense_scaling(replace(config, resume=True, timeout_seconds=42))

    def test_summary_excludes_failed_and_plot_merges_fill_styles(self):
        rows = [{"backend": "qutrit_circuit_exact", "m": 6, "status": "COMPLETED",
                 "runtime_seconds": .1, "estimated_working_set_bytes": 100},
                {"backend": "qutrit_circuit_exact", "m": 6, "status": "OOM",
                 "runtime_seconds": "", "estimated_working_set_bytes": 100}]
        synthetic = summarize(rows, 7)
        self.assertEqual(synthetic[0]["mean_runtime_seconds"], .1)
        self.assertEqual(synthetic[0]["oom_repetitions"], 1)
        real = [{**synthetic[0], "m": 9, "grid_cells_per_player": 9,
                 "synthetic_scaling_point": False}]
        self.assertEqual(len(merged_rows(real, synthetic)), 2)
        overlap = [{**synthetic[0], "synthetic_scaling_point": False}]
        self.assertIs(merged_rows(overlap, synthetic)[0], overlap[0])
        from matplotlib.axes import Axes
        faces = []
        original = Axes.plot
        def spy(axis, *args, **kwargs):
            if kwargs.get("marker") == "o" and kwargs.get("color") == COLORS["qutrit_circuit_exact"]:
                faces.append(kwargs.get("markerfacecolor"))
            return original(axis, *args, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(Axes, "plot", spy):
                plot_scaling(Path(directory), real, synthetic)
            self.assertTrue((Path(directory) / "runtime_vs_grid_size.png").exists())
        self.assertIn("none", faces)
        self.assertIn(COLORS["qutrit_circuit_exact"], faces)

    def test_final_runtime_figure_uses_merged_scaling_rows(self):
        source = Path(__file__).resolve().parents[1] / "results"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "runtime").mkdir()
            for name in ("dense_synthetic_summary.csv",
                         "runtime/runtime_scaling_summary.csv"):
                shutil.copyfile(source / name, output / name)
            original = scaling_plotting.plot_scaling_axis
            seen = []

            def capture(ax, real, synthetic):
                rows = original(ax, real, synthetic)
                seen.append(rows)
                return rows

            with patch.object(scaling_plotting, "plot_scaling_axis", side_effect=capture):
                plot_scaling(output)
            self.assertEqual(len(seen), 1)
            self.assertTrue(seen[0])
            self.assertEqual({row["backend"] for row in seen[0]},
                             set(scaling_plotting.STATEVECTOR_BACKENDS))
            self.assertTrue(any(row["backend"] == "qutrit_circuit_exact"
                                and row["grid_cells_per_player"] == "6"
                                and row["synthetic_scaling_point"] == "True"
                                for row in seen[0]))
            self.assertTrue(any(row["backend"] == "qutrit_circuit_exact"
                                and row["grid_cells_per_player"] == "25"
                                and int(row["oom_repetitions"]) > 0
                                for row in seen[0]))
