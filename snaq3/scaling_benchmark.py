"""Resumable runtime-only QuickQudits scaling by actual grid side length n."""

import argparse
import csv
import json
import multiprocessing as mp
import os
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from random import Random
from statistics import mean, median, stdev
from time import perf_counter

import quickqudits

from .benchmark import RuntimeWorkload, derive_seed, replay_runtime_workload, runtime_workload
from .movement import DIRECTIONS
from .simulation import BACKENDS
from .simulation import _memory

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)

BACKEND_ORDER = tuple(BACKENDS)
DEFAULT_N = {
    "qutrit_circuit_exact": (3, 4, 5),
    "qutrit_tableau_clifford": (3, 4, 5, 10, 20, 30, 60),
    "qubit_transpiled": (2, 3, 4),
    "qubit_tableau_transpiled": (3, 4, 5, 10, 20, 30, 60),
}
RAW_FIELDS = (
    "backend", "n", "grid_cells_per_player", "backend_systems_per_player",
    "repetition", "seed", "rounds", "runtime_seconds", "status", "error",
    "synthetic_scaling_point", "vector_bytes_per_player",
    "estimated_working_set_bytes", "memory_limit_bytes", "timeout_seconds",
    "quickqudits_version", "backend_metadata",
)
SUMMARY_FIELDS = (
    "backend", "n", "grid_cells_per_player", "mean_runtime_seconds",
    "median_runtime_seconds", "std_runtime_seconds", "CI95_low", "CI95_high",
    "completed_repetitions", "oom_repetitions", "timeout_repetitions",
    "synthetic_scaling_point", "representative_memory_requirement",
)
MEMORY_FIELDS = ("n", "m", "backend", "vector_bytes_per_player",
                 "estimated_working_set_bytes", "status")
BOOTSTRAP_DRAWS = 3000
STARTUP_GRACE_SECONDS = 30.0


@dataclass(frozen=True)
class ScalingConfig:
    qutrit_sv_n: tuple[int, ...] = DEFAULT_N["qutrit_circuit_exact"]
    qutrit_tableau_n: tuple[int, ...] = DEFAULT_N["qutrit_tableau_clifford"]
    qubit_sv_n: tuple[int, ...] = DEFAULT_N["qubit_transpiled"]
    qubit_tableau_n: tuple[int, ...] = DEFAULT_N["qubit_tableau_transpiled"]
    rounds: int = 4
    dense_repetitions: int = 5
    tableau_repetitions: int = 20
    seed: int = 20260930
    max_memory_gib: float = 32.0
    timeout_seconds: float = 21600.0
    output_directory: Path = Path("results")
    resume: bool = False
    retry_failed: bool = False

    @property
    def backend_n(self):
        return dict(zip(BACKEND_ORDER, (self.qutrit_sv_n, self.qutrit_tableau_n,
                                         self.qubit_sv_n, self.qubit_tableau_n)))

    def repetitions_for(self, backend):
        return self.tableau_repetitions if "tableau" in backend else self.dense_repetitions

    def validate(self):
        for backend, sizes in self.backend_n.items():
            if len(set(sizes)) != len(sizes):
                raise ValueError(f"duplicate side length for {backend}")
            minimum = 2 if backend == "qubit_transpiled" else 3
            if any(n < minimum for n in sizes):
                raise ValueError(f"{backend} requires n >= {minimum}")
        if self.rounds < 1 or self.dense_repetitions < 1 or self.tableau_repetitions < 1:
            raise ValueError("rounds and repetitions must be positive")
        if self.max_memory_gib <= 0 or self.timeout_seconds <= 0:
            raise ValueError("memory ceiling and timeout must be positive")
        if self.retry_failed and not self.resume:
            raise ValueError("--retry-failed requires --resume")


def dimensions(n: int, backend: str) -> tuple[int, int]:
    if backend not in BACKENDS or n < (2 if backend == "qubit_transpiled" else 3):
        raise ValueError("invalid backend or side length")
    m = n * n
    return m, (2 * m if backend.startswith("qubit_") else m)


def _synthetic_two_by_two(rounds: int, seed: int, repetition: int) -> RuntimeWorkload:
    """A basis-state scaling trace; it does not relax Grid's playable n>=3 rule."""
    rng = Random(derive_seed(seed, repetition, "synthetic_n2_grid", 4))
    resolver_rng = Random(derive_seed(seed, repetition, "synthetic_n2_resolver", 4))
    states, heads = [], []
    for _ in range(2):
        head = rng.randrange(4)
        apple = rng.choice([i for i in range(4) if i != head])
        digits = [0] * 4
        digits[head], digits[apple] = 1, 2
        heads.append(head)
        states.append(digits)
    initial = tuple(tuple(state) for state in states)
    inputs, resolver_seeds, targets = [], [], []
    for _ in range(rounds):
        moves = []
        for player in range(2):
            choices = list(range(4))
            rng.shuffle(choices)
            r, c = divmod(heads[player], 2)
            direction = next(x for x in choices if states[player][
                ((r + DIRECTIONS[x][0]) % 2) * 2 + (c + DIRECTIONS[x][1]) % 2] == 0)
            dr, dc = DIRECTIONS[direction]
            destination = ((r + dr) % 2) * 2 + (c + dc) % 2
            states[player][heads[player]] = 0  # old head/tail removed
            states[player][destination] = 1
            heads[player] = destination
            moves.append(direction)
        inputs.append(tuple(moves))
        resolver_seeds.append(resolver_rng.getrandbits(64))
        targets.append(tuple(tuple(state) for state in states))
    return RuntimeWorkload(initial, tuple(inputs), tuple(resolver_seeds), tuple(targets),
                           {"synthetic_scaling_point": True, "seed": seed, "repetition": repetition})


def scaling_workload(n: int, rounds: int, seed: int, repetition: int) -> RuntimeWorkload:
    if n == 2:
        return _synthetic_two_by_two(rounds, seed, repetition)
    if n < 3:
        raise ValueError("n must be >= 2")
    return runtime_workload(n * n, rounds, seed, repetition)


def memory_estimate(backend: str, m: int) -> tuple[int | str, int]:
    if backend == "qutrit_circuit_exact":
        return _memory.dense_memory_estimate(m, 3)
    if backend == "qubit_transpiled":
        return _memory.dense_memory_estimate(m, 2, 2)
    systems = 2 * m if backend == "qubit_tableau_transpiled" else m
    # QuickQudits 1.0.1: int8 X,Z arrays (2N,N), tau_exp (2N).
    per_tableau = 4 * systems * systems + 2 * systems
    return "", 8 * per_tableau  # two grids plus conservative workspace


def _effective_memory_limit(configured_bytes: int) -> int:
    available = _memory._available_bytes()
    cgroup_root = Path("/sys/fs/cgroup")
    membership = Path("/proc/self/cgroup")
    if membership.exists():
        for line in membership.read_text().splitlines():
            if line.startswith("0::"):
                directory = cgroup_root / line.split("::", 1)[1].lstrip("/")
                for parent in (directory, *directory.parents):
                    if cgroup_root not in (parent, *parent.parents):
                        break
                    limit_file, current_file = parent / "memory.max", parent / "memory.current"
                    if limit_file.exists() and current_file.exists():
                        limit = limit_file.read_text().strip()
                        if limit != "max":
                            available = min(available, max(0, int(limit) - int(current_file.read_text())))
                    if parent == cgroup_root:
                        break
                break
    return min(configured_bytes, available // 2)


def _key(row):
    return row["backend"], int(row["n"]), int(row["repetition"])


def _atomic_csv(path: Path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _load_raw(path: Path):
    if not path.exists():
        return {}
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != RAW_FIELDS:
            raise ValueError("existing raw CSV has a different schema")
        rows = list(reader)
    result = {}
    for row in rows:
        key = _key(row)
        if key in result:
            raise ValueError(f"duplicate checkpoint row: {key}")
        result[key] = row
    return result


def _metadata(config: ScalingConfig):
    return {"backend_n": {name: list(sizes) for name, sizes in config.backend_n.items()},
            "rounds": config.rounds, "dense_repetitions": config.dense_repetitions,
            "tableau_repetitions": config.tableau_repetitions, "seed": config.seed,
            "max_memory_gib": config.max_memory_gib, "timeout_seconds": config.timeout_seconds,
            "quickqudits_version": quickqudits.__version__,
            "workload": "one deterministic trace per (n,repetition), shared across all assigned backends; Bell output ignored for grid updates",
            "quantum_apple_preparation": "excluded from timing; benchmark-only classical apple trace generated before the timed backend replay",
            "synthetic_n2": "qubit statevector only; basis-state head/tail updates; normal Grid keeps n>=3",
            "memory": "dense: four complex128 vectors; tableau: eight times one QuickQudits tableau array set; guard uses configured and available host/cgroup memory; worker RLIMIT_AS",
            "runtime_boundary": "full two-grid preparation, Bell resolver with classical direction labels, grid updates; worker startup and warm-up excluded",
            "timeout": "worker SIGALRM plus parent hard stop after startup grace",
            "confidence_interval": f"deterministic percentile bootstrap across completed repetitions ({BOOTSTRAP_DRAWS} draws)"}


def _check_resume_metadata(path: Path, config: ScalingConfig):
    if not path.exists():
        return
    found, expected = json.loads(path.read_text()), _metadata(config)
    keys = ["rounds", "seed", "quickqudits_version"]
    if not config.retry_failed:
        keys += ["max_memory_gib", "timeout_seconds"]
    for key in keys:
        if found.get(key) != expected[key]:
            raise ValueError(f"resume configuration differs for {key}")


def _base_row(config, backend, n, repetition, vector_bytes, working_bytes):
    m, systems = dimensions(n, backend)
    return {"backend": backend, "n": n, "grid_cells_per_player": m,
            "backend_systems_per_player": systems, "repetition": repetition,
            "seed": config.seed, "rounds": config.rounds, "runtime_seconds": "",
            "status": "ERROR", "error": "", "synthetic_scaling_point": backend == "qubit_transpiled" and n == 2,
            "vector_bytes_per_player": vector_bytes,
            "estimated_working_set_bytes": working_bytes,
            "memory_limit_bytes": int(config.max_memory_gib * 1024**3),
            "timeout_seconds": config.timeout_seconds,
            "quickqudits_version": quickqudits.__version__, "backend_metadata": ""}


def _timeout_handler(signum, frame):
    raise TimeoutError("timed backend workload exceeded timeout")


def _worker(connection, backend_name, workload, memory_gib, timeout_seconds, expected_systems):
    try:
        import resource
        baseline_kib = next(int(line.split()[1]) for line in Path("/proc/self/status").read_text().splitlines()
                            if line.startswith("VmSize:"))
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        safe_budget = _effective_memory_limit(int(memory_gib * 1024**3))
        ceiling = baseline_kib * 1024 + safe_budget
        soft = ceiling if hard == resource.RLIM_INFINITY else min(ceiling, hard)
        resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
        _memory.set_max_working_set_gib(memory_gib)
        backend = BACKENDS[backend_name]()
        # A fresh worker needs its own small warm-up outside the timed region.
        replay_runtime_workload(backend, runtime_workload(9, 1, 1, -1))
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
        try:
            start = perf_counter()
            grids = replay_runtime_workload(backend, workload)
            elapsed = perf_counter() - start
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        assert len(grids) == 2
        actual = tuple(grid.tableau.n if hasattr(grid, "tableau") else grid.circuit.n for grid in grids)
        assert actual == (expected_systems, expected_systems), (actual, expected_systems)
        metadata = {**backend.metadata, "systems_per_player": expected_systems,
                    "player_grid_count": 2,
                    "tableau_array_bytes_per_player": (
                        grids[0].tableau.X.nbytes + grids[0].tableau.Z.nbytes
                        + grids[0].tableau.tau_exp.nbytes) if hasattr(grids[0], "tableau") else None}
        connection.send({"status": "COMPLETED", "runtime_seconds": elapsed,
                         "backend_metadata": json.dumps(metadata, sort_keys=True)})
    except TimeoutError as exc:
        connection.send({"status": "TIMEOUT", "error": str(exc)})
    except MemoryError as exc:
        connection.send({"status": "OOM", "error": f"{type(exc).__name__}: {exc}"})
    except BaseException as exc:
        connection.send({"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"})
    finally:
        connection.close()


def _measure(backend_name, workload, config, expected_systems):
    context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(sender, backend_name, workload,
                                                     config.max_memory_gib, config.timeout_seconds,
                                                     expected_systems))
    process.start()
    sender.close()
    try:
        process.join(config.timeout_seconds + STARTUP_GRACE_SECONDS)
        if process.is_alive():
            process.terminate()
            process.join(5)
            if process.is_alive():
                process.kill()
                process.join()
            return {"status": "TIMEOUT", "error": "worker exceeded timeout plus startup grace"}
        if receiver.poll():
            return receiver.recv()
        if process.exitcode == -signal.SIGKILL:
            return {"status": "OOM", "error": "worker killed by SIGKILL; likely host memory exhaustion"}
        return {"status": "ERROR", "error": f"worker exited without result (exit code {process.exitcode})"}
    except KeyboardInterrupt:
        if process.is_alive():
            process.terminate()
            process.join(5)
            if process.is_alive():
                process.kill()
                process.join()
        raise
    finally:
        receiver.close()


def _percentile(values, fraction):
    position = fraction * (len(values) - 1)
    lower = int(position)
    mix = position - lower
    return values[lower] * (1 - mix) + values[min(lower + 1, len(values) - 1)] * mix


def summarize(rows, seed: int):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["backend"], int(row["n"])), []).append(row)
    summary = []
    for (backend, n), group in grouped.items():
        completed = [float(row["runtime_seconds"]) for row in group if row["status"] == "COMPLETED"]
        item = {"backend": backend, "n": n, "grid_cells_per_player": n * n,
                "mean_runtime_seconds": mean(completed) if completed else "",
                "median_runtime_seconds": median(completed) if completed else "",
                "std_runtime_seconds": stdev(completed) if len(completed) > 1 else (0.0 if completed else ""),
                "CI95_low": "", "CI95_high": "",
                "completed_repetitions": len(completed),
                "oom_repetitions": sum(row["status"] == "OOM" for row in group),
                "timeout_repetitions": sum(row["status"] == "TIMEOUT" for row in group),
                "synthetic_scaling_point": backend == "qubit_transpiled" and n == 2,
                "representative_memory_requirement": group[0]["estimated_working_set_bytes"]}
        if len(completed) > 1:
            rng = Random(derive_seed(seed, 0, f"grid_size_bootstrap:{backend}:{n}"))
            boot = sorted(mean(completed[rng.randrange(len(completed))] for _ in completed)
                          for _ in range(BOOTSTRAP_DRAWS))
            item["CI95_low"], item["CI95_high"] = _percentile(boot, .025), _percentile(boot, .975)
        summary.append(item)
    return summary


def _write_memory_table(output: Path, rows):
    groups = {}
    for row in rows:
        if row["backend"] in ("qutrit_circuit_exact", "qubit_transpiled"):
            groups.setdefault((row["backend"], int(row["n"])), []).append(row)
    table = []
    for (backend, n), group in groups.items():
        status = "COMPLETED" if any(r["status"] == "COMPLETED" for r in group) else (
            "OOM" if any(r["status"] == "OOM" for r in group) else group[0]["status"])
        table.append({"n": n, "m": n * n, "backend": backend,
                      "vector_bytes_per_player": group[0]["vector_bytes_per_player"],
                      "estimated_working_set_bytes": group[0]["estimated_working_set_bytes"],
                      "status": status})
    _atomic_csv(output / "runtime_memory_limits.csv", MEMORY_FIELDS, table)


def _finalize(output: Path, rows, seed):
    summary = summarize(rows, seed)
    _atomic_csv(output / "runtime_scaling_summary.csv", SUMMARY_FIELDS, summary)
    _write_memory_table(output, rows)
    from .scaling_plotting import plot_scaling
    plot_scaling(output, summary)
    return summary


def run_scaling(config: ScalingConfig):
    config.validate()
    output = Path(config.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "runtime_scaling_raw.csv"
    metadata_path = output / "runtime_scaling_metadata.json"
    if raw_path.exists() and not config.resume:
        raise FileExistsError(f"{raw_path} exists; use --resume or another output directory")
    if config.resume:
        _check_resume_metadata(metadata_path, config)
    rows_by_key = _load_raw(raw_path) if config.resume else {}
    run_metadata = _metadata(config)
    metadata_path.write_text(json.dumps(run_metadata, indent=2) + "\n")
    if not raw_path.exists():
        _atomic_csv(raw_path, RAW_FIELDS, [])
    interrupted = False
    try:
        # Increasing n gives common reference points early, then extends tableau range.
        for n in sorted({n for sizes in config.backend_n.values() for n in sizes}):
            workloads = {}  # one trace per (n, repetition), shared across backends
            for backend_name in BACKEND_ORDER:
                if n not in config.backend_n[backend_name]:
                    continue
                m, systems = dimensions(n, backend_name)
                vector_bytes, working_bytes = memory_estimate(backend_name, m)
                memory_limit = int(config.max_memory_gib * 1024**3)
                previous_bound = any(
                    row["backend"] == backend_name and int(row["n"]) < n
                    and row["status"] == "OOM" and int(row["memory_limit_bytes"]) == memory_limit
                    and (row["error"].startswith("configured memory guard")
                         or row["error"] == "monotonic memory bound after previous OOM")
                    for row in rows_by_key.values()) if backend_name in ("qutrit_circuit_exact", "qubit_transpiled") else False
                for repetition in range(config.repetitions_for(backend_name)):
                    key = (backend_name, n, repetition)
                    existing = rows_by_key.get(key)
                    if existing and (existing["status"] == "COMPLETED" or not config.retry_failed):
                        continue
                    row = _base_row(config, backend_name, n, repetition, vector_bytes, working_bytes)
                    workload = workloads.get(repetition)
                    if previous_bound:
                        row.update(status="OOM", error="monotonic memory bound after previous OOM")
                    elif working_bytes > memory_limit:
                        row.update(status="OOM", error="configured memory guard: estimated working set exceeds limit")
                    elif working_bytes > _effective_memory_limit(memory_limit):
                        row.update(status="OOM", error="available-memory guard: estimated working set exceeds current safe limit")
                    else:
                        try:
                            if workload is None:
                                workload = scaling_workload(n, config.rounds, config.seed, repetition)
                                workloads[repetition] = workload
                            result = _measure(backend_name, workload, config, systems)
                            row.update(result)
                            if row["status"] == "COMPLETED" and not row["runtime_seconds"] > 0:
                                row.update(status="ERROR", error="nonpositive measured runtime", runtime_seconds="")
                        except MemoryError as exc:
                            row.update(status="OOM", error=f"{type(exc).__name__}: {exc}")
                        except Exception as exc:
                            row.update(status="ERROR", error=f"{type(exc).__name__}: {exc}")
                    metadata = json.loads(row["backend_metadata"]) if row["backend_metadata"] else dict(BACKENDS[backend_name].metadata)
                    metadata.update({"systems_per_player": systems, "player_grid_count": 2,
                                     "synthetic_scaling_point": row["synthetic_scaling_point"],
                                     "workload_seed_derivation": "SnaQ3-v2 master/n/repetition streams"})
                    if workload is not None and row["status"] == "COMPLETED":
                        metadata["workload_sha256"] = workload.digest()
                    row["backend_metadata"] = json.dumps(metadata, sort_keys=True)
                    rows_by_key[key] = row
                    _atomic_csv(raw_path, RAW_FIELDS, list(rows_by_key.values()))
                    print(f"{backend_name} n={n} rep={repetition}: {row['status']}", flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print("Interrupted; earlier rows are checkpointed. Resume with --resume.", flush=True)
    finally:
        summary = _finalize(output, list(rows_by_key.values()), config.seed)
    return list(rows_by_key.values()), summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qutrit-sv-n", nargs="*", type=int, default=list(DEFAULT_N["qutrit_circuit_exact"]))
    parser.add_argument("--qutrit-tableau-n", nargs="*", type=int, default=list(DEFAULT_N["qutrit_tableau_clifford"]))
    parser.add_argument("--qubit-sv-n", nargs="*", type=int, default=list(DEFAULT_N["qubit_transpiled"]))
    parser.add_argument("--qubit-tableau-n", nargs="*", type=int, default=list(DEFAULT_N["qubit_tableau_transpiled"]))
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--dense-repetitions", type=int, default=5)
    parser.add_argument("--tableau-repetitions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--max-memory-gib", type=float, default=32.0)
    parser.add_argument("--timeout-seconds", type=float, default=21600.0)
    parser.add_argument("--output-directory", default="results")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        output = Path(args.output_directory)
        raw = output / "runtime_scaling_raw.csv"
        if not raw.exists():
            raise FileNotFoundError(raw)
        _finalize(output, list(_load_raw(raw).values()), args.seed)
        return
    config = ScalingConfig(tuple(args.qutrit_sv_n), tuple(args.qutrit_tableau_n),
                           tuple(args.qubit_sv_n), tuple(args.qubit_tableau_n),
                           args.rounds, args.dense_repetitions, args.tableau_repetitions,
                           args.seed, args.max_memory_gib, args.timeout_seconds,
                           Path(args.output_directory), args.resume, args.retry_failed)
    run_scaling(config)


if __name__ == "__main__":
    main()
