"""Resumable dense-register scaling at arbitrary logical qutrit cell counts.

These synthetic register sizes measure representation cost, not playable boards.
The two full QuickQudits statevectors and exact Bell resolver are unchanged.
"""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from random import Random
from statistics import mean, median, stdev

import quickqudits

from .benchmark import RuntimeWorkload, derive_seed
from .scaling_benchmark import (
    BOOTSTRAP_DRAWS, _atomic_csv, _effective_memory_limit, _measure,
    _percentile, memory_estimate,
)
from .simulation import BACKENDS

DENSE_BACKENDS = ("qutrit_circuit_exact", "qubit_transpiled")
DEFAULT_M = {
    "qutrit_circuit_exact": (4, 6, 8, 9, 10, 12, 14, 16, 17, 18, 19),
    "qubit_transpiled": (4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
}
RAW_FIELDS = (
    "backend", "m", "backend_systems_per_player", "repetition", "seed", "rounds",
    "runtime_seconds", "status", "error", "synthetic_scaling_point",
    "vector_bytes_per_player", "estimated_working_set_bytes", "memory_limit_bytes",
    "timeout_seconds", "quickqudits_version", "backend_metadata",
)
SUMMARY_FIELDS = (
    "backend", "m", "grid_cells_per_player", "mean_runtime_seconds",
    "median_runtime_seconds", "std_runtime_seconds", "CI95_low", "CI95_high",
    "completed_repetitions", "oom_repetitions", "timeout_repetitions",
    "synthetic_scaling_point", "representative_memory_requirement",
)


@dataclass(frozen=True)
class DenseScalingConfig:
    qutrit_m: tuple[int, ...] = DEFAULT_M["qutrit_circuit_exact"]
    qubit_m: tuple[int, ...] = DEFAULT_M["qubit_transpiled"]
    rounds: int = 4
    repetitions: int = 5
    seed: int = 20260930
    max_memory_gib: float = 32.0
    timeout_seconds: float = 21600.0
    output_directory: Path = Path("results")
    resume: bool = False
    retry_failed: bool = False

    @property
    def backend_m(self):
        return dict(zip(DENSE_BACKENDS, (self.qutrit_m, self.qubit_m)))

    def validate(self):
        for backend, sizes in self.backend_m.items():
            if len(set(sizes)) != len(sizes) or any(m < 4 for m in sizes):
                raise ValueError(f"{backend}: m must contain unique integers >= 4")
        if self.rounds < 1 or self.repetitions < 1:
            raise ValueError("rounds and repetitions must be positive")
        if self.max_memory_gib <= 0 or self.timeout_seconds <= 0:
            raise ValueError("memory ceiling and timeout must be positive")
        if self.retry_failed and not self.resume:
            raise ValueError("--retry-failed requires --resume")


def synthetic_workload(m: int, rounds: int, seed: int, repetition: int) -> RuntimeWorkload:
    """Two sparse length-m arrays with fixed, geometry-free X updates."""
    if m < 4 or rounds < 1:
        raise ValueError("synthetic workload requires m >= 4 and positive rounds")
    seeds = {stream: derive_seed(seed, repetition, f"dense_synthetic_{stream}", m)
             for stream in ("grid", "workload", "resolver")}
    grid_rng, update_rng = Random(seeds["grid"]), Random(seeds["workload"])
    resolver_rng = Random(seeds["resolver"])
    states, occupied = [], []
    for _ in range(2):
        head, apple = grid_rng.sample(range(m), 2)
        digits = [0] * m
        digits[head], digits[apple] = 1, 2
        states.append(digits)
        occupied.append((head, apple))
    initial = tuple(tuple(state) for state in states)
    inputs, resolver_seeds, targets = [], [], []
    for _ in range(rounds):
        inputs.append((update_rng.randrange(4), update_rng.randrange(4)))
        resolver_seeds.append(resolver_rng.getrandbits(64))
        for player in range(2):
            old_head, old_apple = occupied[player]
            candidates = [i for i in range(m) if i not in (old_head, old_apple)]
            new_head, new_apple = update_rng.sample(candidates, 2)
            state = states[player]
            state[old_head] = state[old_apple] = 0
            state[new_head], state[new_apple] = 1, 2
            occupied[player] = (new_head, new_apple)
        targets.append(tuple(tuple(state) for state in states))
    return RuntimeWorkload(initial, tuple(inputs), tuple(resolver_seeds), tuple(targets), seeds)


def _key(row):
    return row["backend"], int(row["m"]), int(row["repetition"])


def _load_raw(path):
    if not path.exists():
        return {}
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != RAW_FIELDS:
            raise ValueError("existing synthetic raw CSV has a different schema")
        result = {}
        for row in reader:
            key = _key(row)
            if key in result:
                raise ValueError(f"duplicate checkpoint row: {key}")
            result[key] = row
    return result


def _existing_grid_points(output):
    path = output / "runtime_scaling_raw.csv"
    if not path.exists():
        path = output / "runtime" / "runtime_scaling_raw.csv"
    if not path.exists():
        return set()
    with path.open(newline="") as file:
        return {(row["backend"], int(row["grid_cells_per_player"]))
                for row in csv.DictReader(file) if row["backend"] in DENSE_BACKENDS}


def _metadata(config):
    return {
        "backend_m": {backend: list(sizes) for backend, sizes in config.backend_m.items()},
        "rounds": config.rounds, "repetitions": config.repetitions, "seed": config.seed,
        "max_memory_gib": config.max_memory_gib, "timeout_seconds": config.timeout_seconds,
        "quickqudits_version": quickqudits.__version__,
        "workload": "two length-m sparse logical arrays; identical deterministic inputs, resolver seeds, and four-cell-per-player updates for shared m; no board geometry",
        "quantum_apple_preparation": "excluded from timing; synthetic traces use fixed apple-valued digits, not playable apple preparation",
        "runtime_boundary": "full two-grid preparation, exact Bell resolver with classical direction labels, grid updates; worker startup and warm-up excluded",
        "memory": "four complex128 dense vectors; configured and available host/cgroup guards; worker RLIMIT_AS",
        "confidence_interval": f"deterministic percentile bootstrap across completed repetitions ({BOOTSTRAP_DRAWS} draws)",
        "interpretation": "synthetic points measure register representation scaling only; they are not playable SnaQ3 grids",
    }


def _base_row(config, backend, m, repetition, vector_bytes, working_bytes):
    return {
        "backend": backend, "m": m,
        "backend_systems_per_player": m if backend == "qutrit_circuit_exact" else 2 * m,
        "repetition": repetition, "seed": config.seed, "rounds": config.rounds,
        "runtime_seconds": "", "status": "ERROR", "error": "",
        "synthetic_scaling_point": True, "vector_bytes_per_player": vector_bytes,
        "estimated_working_set_bytes": working_bytes,
        "memory_limit_bytes": int(config.max_memory_gib * 1024**3),
        "timeout_seconds": config.timeout_seconds,
        "quickqudits_version": quickqudits.__version__, "backend_metadata": "",
    }


def summarize(rows, seed):
    groups = {}
    for row in rows:
        groups.setdefault((row["backend"], int(row["m"])), []).append(row)
    result = []
    for (backend, m), group in sorted(groups.items(), key=lambda item: (item[0][1], item[0][0])):
        values = [float(row["runtime_seconds"]) for row in group if row["status"] == "COMPLETED"]
        item = {
            "backend": backend, "m": m, "grid_cells_per_player": m,
            "mean_runtime_seconds": mean(values) if values else "",
            "median_runtime_seconds": median(values) if values else "",
            "std_runtime_seconds": stdev(values) if len(values) > 1 else (0.0 if values else ""),
            "CI95_low": "", "CI95_high": "",
            "completed_repetitions": len(values),
            "oom_repetitions": sum(row["status"] == "OOM" for row in group),
            "timeout_repetitions": sum(row["status"] == "TIMEOUT" for row in group),
            "synthetic_scaling_point": True,
            "representative_memory_requirement": group[0]["estimated_working_set_bytes"],
        }
        if len(values) > 1:
            rng = Random(derive_seed(seed, 0, f"dense_synthetic_bootstrap:{backend}:{m}"))
            boot = sorted(mean(values[rng.randrange(len(values))] for _ in values)
                          for _ in range(BOOTSTRAP_DRAWS))
            item["CI95_low"], item["CI95_high"] = _percentile(boot, .025), _percentile(boot, .975)
        result.append(item)
    return result


def _finalize(output, rows, seed):
    summary = summarize(rows, seed)
    _atomic_csv(output / "dense_synthetic_summary.csv", SUMMARY_FIELDS, summary)
    from .scaling_plotting import plot_scaling
    plot_scaling(output)
    return summary


def run_dense_scaling(config: DenseScalingConfig):
    config.validate()
    output = Path(config.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "dense_synthetic_raw.csv"
    metadata_path = output / "dense_synthetic_metadata.json"
    if raw_path.exists() and not config.resume:
        raise FileExistsError(f"{raw_path} exists; use --resume or another output directory")
    expected_metadata = _metadata(config)
    if config.resume and metadata_path.exists():
        found = json.loads(metadata_path.read_text())
        keys = ["rounds", "seed", "quickqudits_version"]
        if not config.retry_failed:
            keys += ["max_memory_gib", "timeout_seconds"]
        for key in keys:
            if found.get(key) != expected_metadata[key]:
                raise ValueError(f"resume configuration differs for {key}")
    rows_by_key = _load_raw(raw_path) if config.resume else {}
    existing_grid = _existing_grid_points(output)
    metadata_path.write_text(json.dumps(expected_metadata, indent=2) + "\n")
    if not raw_path.exists():
        _atomic_csv(raw_path, RAW_FIELDS, [])
    try:
        for m in sorted(set(config.qutrit_m) | set(config.qubit_m)):
            workload_by_repetition = {}
            for backend in DENSE_BACKENDS:
                if m not in config.backend_m[backend] or (backend, m) in existing_grid:
                    continue
                vector_bytes, working_bytes = memory_estimate(backend, m)
                limit = int(config.max_memory_gib * 1024**3)
                previous_bound = any(
                    row["backend"] == backend and int(row["m"]) < m
                    and row["status"] == "OOM" and int(row["memory_limit_bytes"]) == limit
                    and (row["error"].startswith("configured memory guard")
                         or row["error"] == "monotonic memory bound after previous OOM")
                    for row in rows_by_key.values())
                for repetition in range(config.repetitions):
                    key = (backend, m, repetition)
                    existing = rows_by_key.get(key)
                    if existing and (existing["status"] == "COMPLETED" or not config.retry_failed):
                        continue
                    row = _base_row(config, backend, m, repetition, vector_bytes, working_bytes)
                    workload = workload_by_repetition.get(repetition)
                    if previous_bound:
                        row.update(status="OOM", error="monotonic memory bound after previous OOM")
                    elif working_bytes > limit:
                        row.update(status="OOM", error="configured memory guard: estimated working set exceeds limit")
                    elif working_bytes > _effective_memory_limit(limit):
                        row.update(status="OOM", error="available-memory guard: estimated working set exceeds current safe limit")
                    else:
                        try:
                            if workload is None:
                                workload = synthetic_workload(m, config.rounds, config.seed, repetition)
                                workload_by_repetition[repetition] = workload
                            row.update(_measure(backend, workload, config, row["backend_systems_per_player"]))
                            if row["status"] == "COMPLETED" and not row["runtime_seconds"] > 0:
                                row.update(status="ERROR", error="nonpositive measured runtime", runtime_seconds="")
                        except MemoryError as exc:
                            row.update(status="OOM", error=f"{type(exc).__name__}: {exc}")
                        except Exception as exc:
                            row.update(status="ERROR", error=f"{type(exc).__name__}: {exc}")
                    metadata = json.loads(row["backend_metadata"]) if row["backend_metadata"] else dict(BACKENDS[backend].metadata)
                    metadata.update({
                        "systems_per_player": row["backend_systems_per_player"],
                        "player_grid_count": 2, "synthetic_scaling_point": True,
                        "workload_seed_derivation": "SnaQ3-v2 master/m/repetition independent streams",
                    })
                    if workload is not None and row["status"] == "COMPLETED":
                        metadata["workload_sha256"] = workload.digest()
                    row["backend_metadata"] = json.dumps(metadata, sort_keys=True)
                    rows_by_key[key] = row
                    _atomic_csv(raw_path, RAW_FIELDS, list(rows_by_key.values()))
                    print(f"{backend} m={m} rep={repetition}: {row['status']}", flush=True)
    except KeyboardInterrupt:
        print("Interrupted; earlier rows are checkpointed. Resume with --resume.", flush=True)
    finally:
        summary = _finalize(output, list(rows_by_key.values()), config.seed)
    return list(rows_by_key.values()), summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qutrit-m", nargs="*", type=int, default=list(DEFAULT_M["qutrit_circuit_exact"]))
    parser.add_argument("--qubit-m", nargs="*", type=int, default=list(DEFAULT_M["qubit_transpiled"]))
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--max-memory-gib", type=float, default=32.0)
    parser.add_argument("--timeout-seconds", type=float, default=21600.0)
    parser.add_argument("--output-directory", type=Path, default=Path("results"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        output = args.output_directory
        raw = output / "dense_synthetic_raw.csv"
        if not raw.exists():
            raise FileNotFoundError(raw)
        _finalize(output, list(_load_raw(raw).values()), args.seed)
        return
    config = DenseScalingConfig(tuple(args.qutrit_m), tuple(args.qubit_m), args.rounds,
                                args.repetitions, args.seed, args.max_memory_gib,
                                args.timeout_seconds, args.output_directory,
                                args.resume, args.retry_failed)
    run_dense_scaling(config)


if __name__ == "__main__":
    main()
