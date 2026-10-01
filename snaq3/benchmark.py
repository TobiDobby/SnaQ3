"""Paired resolver accuracy and fixed-trace full-grid runtime experiments."""

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from math import cos, isqrt, pi
from pathlib import Path
from random import Random
from statistics import mean, stdev
from time import perf_counter

import quickqudits

from .grid import Grid
from .payoff import utility, wins
from .resolver import clifford_mapping
from .simulation import BACKENDS
from .simulation import _memory

EXACT_PAIR = ("qutrit_circuit_exact", "qubit_transpiled")
TABLEAU_PAIR = ("qutrit_tableau_clifford", "qubit_tableau_transpiled")
PAYOFF_METRICS = ("F_A", "F_B", "F_avg", "p_win")
RUNTIME_METRICS = ("runtime_seconds",)
BOOTSTRAP_DRAWS = 5000


def derive_seed(master: int, repetition: int, stream: str, grid_size: int = 0) -> int:
    payload = f"SnaQ3-v2|{master}|{repetition}|{stream}|{grid_size}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def payoff_samples(shots_per_input: int, master_seed: int, repetition: int):
    if shots_per_input < 1:
        raise ValueError("shots_per_input must be positive")
    input_seed = derive_seed(master_seed, repetition, "payoff_inputs")
    resolver_seed = derive_seed(master_seed, repetition, "payoff_resolver")
    inputs = [(x, y) for x in range(4) for y in range(4) for _ in range(shots_per_input)]
    Random(input_seed).shuffle(inputs)
    rng = Random(resolver_seed)
    seeds = [rng.getrandbits(64) for _ in inputs]
    return inputs, seeds, {"input_seed": input_seed, "resolver_seed": resolver_seed}


@dataclass(frozen=True)
class RuntimeWorkload:
    initial_grids: tuple[tuple[int, ...], tuple[int, ...]]
    inputs: tuple[tuple[int, int], ...]
    resolver_seeds: tuple[int, ...]
    targets: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    seeds: dict

    def digest(self):
        payload = json.dumps({"initial_grids": self.initial_grids, "inputs": self.inputs,
                              "resolver_seeds": self.resolver_seeds, "targets": self.targets},
                             separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()


class _RuntimeTraceGrid(Grid):
    """Preserve the fixed classical apple trace outside the timed workload.

    Runtime measurements compare register representations. Quantum apple
    preparation is a gameplay operation and is deliberately not timed here.
    """

    def prepare_apples(self, rng: Random) -> None:
        empty = [divmod(i, self.n) for i, value in enumerate(self.digits()) if value == 0]
        if empty:
            self.apples.add(empty[rng.randrange(len(empty))])


def runtime_workload(grid_size: int, rounds: int, master_seed: int, repetition: int) -> RuntimeWorkload:
    n = isqrt(grid_size)
    if n * n != grid_size or n < 3 or rounds < 1:
        raise ValueError("grid size must be square >=9 and rounds positive")
    seeds = {key: derive_seed(master_seed, repetition, key, grid_size)
             for key in ("grid", "workload", "resolver")}
    grid_rng = Random(seeds["grid"])
    workload_rng = Random(seeds["workload"])
    resolver_rng = Random(seeds["resolver"])
    grids = [_RuntimeTraceGrid.initial(n, grid_rng), _RuntimeTraceGrid.initial(n, grid_rng)]
    initial = (grids[0].digits(), grids[1].digits())
    inputs, resolver_seeds, targets = [], [], []
    for _ in range(rounds):
        for i in range(2):
            if not grids[i].alive:
                grids[i] = _RuntimeTraceGrid.initial(n, grid_rng)
        pair = (workload_rng.randrange(4), workload_rng.randrange(4))
        inputs.append(pair)
        resolver_seeds.append(resolver_rng.getrandbits(64))
        for grid, direction in zip(grids, pair):
            grid.move(direction, grid_rng)
        targets.append((grids[0].digits(), grids[1].digits()))
    return RuntimeWorkload(initial, tuple(inputs), tuple(resolver_seeds), tuple(targets), seeds)


def replay_runtime_workload(backend, workload: RuntimeWorkload):
    """Execute grid registers, classical-label Bell resolution, and grid updates.

    Bell outcomes cannot change the predetermined classical grid trace.
    """
    grids = [backend.make_grid(digits) for digits in workload.initial_grids]
    for (x_a, x_b), resolver_seed, targets in zip(workload.inputs, workload.resolver_seeds,
                                                  workload.targets):
        backend.resolve(x_a, x_b, resolver_seed)
        for grid, digits in zip(grids, targets):
            grid.set_digits(digits)
    return grids


def _write_csv(path, fields, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _percentile(values, fraction):
    position = fraction * (len(values) - 1)
    lower = int(position)
    mix = position - lower
    return values[lower] * (1 - mix) + values[min(lower + 1, len(values) - 1)] * mix


def _summarize(rows, metrics, group_fields, master_seed):
    groups = {}
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        groups.setdefault(key, []).append(row)
    summary = []
    for key, group in groups.items():
        completed = [row for row in group if row["status"] == "completed"]
        item = dict(zip(group_fields, key))
        item["successful_repetitions"] = len(completed)
        item["failed_repetitions"] = len(group) - len(completed)
        for metric in metrics:
            values = [float(row[metric]) for row in completed]
            item[f"mean_{metric}"] = mean(values) if values else ""
            item[f"std_{metric}"] = stdev(values) if len(values) > 1 else (0.0 if values else "")
            if len(values) > 1:
                shared_key = tuple(value for field, value in zip(group_fields, key) if field != "backend")
                rng = Random(derive_seed(master_seed, 0, "bootstrap:" + metric + ":" + ":".join(map(str, shared_key))))
                boot = sorted(mean(values[rng.randrange(len(values))] for _ in values)
                              for _ in range(BOOTSTRAP_DRAWS))
                item[f"ci95_low_{metric}"] = _percentile(boot, .025)
                item[f"ci95_high_{metric}"] = _percentile(boot, .975)
            else:
                item[f"ci95_low_{metric}"] = ""
                item[f"ci95_high_{metric}"] = ""
        summary.append(item)
    return summary


def _summary_fields(group_fields, metrics):
    return (*group_fields, *(f"{stat}_{metric}" for metric in metrics
                              for stat in ("mean", "std", "ci95_low", "ci95_high")),
            "successful_repetitions", "failed_repetitions")


def _check_pair_agreement(outcomes, pair):
    if outcomes[pair[0]] != outcomes[pair[1]]:
        raise AssertionError(f"paired resolver disagreement: {pair}: {outcomes}")


def run_payoff_experiment(shots_per_input=1000, payoff_repetitions=10,
                          seed=20260929, output_directory="results"):
    if payoff_repetitions < 1:
        raise ValueError("payoff_repetitions must be positive")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    backends = {name: cls() for name, cls in BACKENDS.items()}
    rows, derivations = [], []
    fields = ("backend", "repetition", "seed", "input_seed", "resolver_seed",
              "shots_per_input", "total_samples", *PAYOFF_METRICS, "status", "error")
    for repetition in range(payoff_repetitions):
        inputs, resolver_seeds, seeds = payoff_samples(shots_per_input, seed, repetition)
        derivations.append({"repetition": repetition, **seeds})
        totals = {name: [0.0, 0.0, 0] for name in backends}
        for pair, resolver_seed in zip(inputs, resolver_seeds):
            outcomes = {name: backend.resolve(*pair, resolver_seed)
                        for name, backend in backends.items()}
            _check_pair_agreement(outcomes, EXACT_PAIR)
            _check_pair_agreement(outcomes, TABLEAU_PAIR)
            for name, (a, b) in outcomes.items():
                u_a, u_b = utility(*pair, a, b)
                totals[name][0] += u_a
                totals[name][1] += u_b
                totals[name][2] += wins(*pair, a, b)
        count = len(inputs)
        for name, (u_a, u_b, won) in totals.items():
            rows.append({"backend": name, "repetition": repetition, "seed": seed,
                         **seeds, "shots_per_input": shots_per_input,
                         "total_samples": count, "F_A": u_a / count,
                         "F_B": u_b / count, "F_avg": (u_a + u_b) / (2 * count),
                         "p_win": won / count, "status": "completed", "error": ""})
        print(f"payoff repetition {repetition + 1}/{payoff_repetitions} completed", flush=True)
    _write_csv(output / "payoff_raw.csv", fields, rows)
    summary = _summarize(rows, PAYOFF_METRICS, ("backend",), seed)
    _write_csv(output / "payoff_summary.csv", _summary_fields(("backend",), PAYOFF_METRICS), summary)
    (output / "payoff_metadata.json").write_text(json.dumps({
        "master_seed": seed, "seed_derivation": "first 64 bits of SHA256('SnaQ3-v2|master|repetition|stream|grid_size')",
        "repetition_seeds": derivations, "input_design": "each of 16 Z4 x Z4 pairs exactly shots_per_input times; shuffled",
        "resolver_seeds": "Random(resolver_seed).getrandbits(64) in shuffled input order; shared by four backends",
        "shots_per_input": shots_per_input, "payoff_repetitions": payoff_repetitions,
        "confidence_interval": f"percentile bootstrap of repetition means, {BOOTSTRAP_DRAWS} draws, deterministic derived seed shared across paired backends",
        "analytical": {"p_win_quantum": cos(pi / 8)**2,
                       "F_avg_quantum": .75 * cos(pi / 8)**2,
                       "p_win_classical": .75, "F_avg_classical": 9 / 16},
        "quickqudits_version": quickqudits.__version__}, indent=2) + "\n")
    return rows, summary


def _memory_fields(name, m):
    if name == "qutrit_circuit_exact":
        return _memory.dense_memory_estimate(m, 3)
    if name == "qubit_transpiled":
        return _memory.dense_memory_estimate(m, 2, 2)
    return "", ""


def run_runtime_experiment(grid_sizes=(9, 16, 25), rounds=4,
                           dense_runtime_repetitions=10, tableau_runtime_repetitions=100,
                           seed=20260929, output_directory="results", max_statevector_gib=8.0):
    if rounds < 1 or dense_runtime_repetitions < 1 or tableau_runtime_repetitions < 1:
        raise ValueError("rounds and repetitions must be positive")
    sizes = tuple(grid_sizes)
    if any(isqrt(m)**2 != m or m < 9 for m in sizes):
        raise ValueError("grid sizes must be squares >= 9")
    _memory.set_max_working_set_gib(max_statevector_gib)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    (output / "clifford_observable_mapping.json").write_text(json.dumps(clifford_mapping(), indent=2) + "\n")
    backends = {name: cls() for name, cls in BACKENDS.items()}
    # Warm up outside the timed region with a small full-grid trace.
    warmup = runtime_workload(9, 1, seed, -1)
    for backend in backends.values():
        try:
            replay_runtime_workload(backend, warmup)
        except MemoryError:
            # A deliberately low configured limit can make even warm-up infeasible.
            pass
    fields = ("backend", "grid_size", "n", "repetition", "seed", "rounds", "workload_hash",
              "runtime_seconds", "status", "error", "vector_bytes", "estimated_working_set_bytes",
              "configured_limit_bytes", "backend_metadata")
    rows, trace_seeds = [], []
    try:
        for m in sizes:
            repetitions = max(dense_runtime_repetitions, tableau_runtime_repetitions)
            for repetition in range(repetitions):
                workload = runtime_workload(m, rounds, seed, repetition)
                trace_seeds.append({"grid_size": m, "repetition": repetition,
                                    "workload_hash": workload.digest(), **workload.seeds})
                for name, backend in backends.items():
                    limit = tableau_runtime_repetitions if "tableau" in name else dense_runtime_repetitions
                    if repetition >= limit:
                        continue
                    vector_bytes, working_bytes = _memory_fields(name, m)
                    row = {"backend": name, "grid_size": m, "n": isqrt(m), "repetition": repetition,
                           "seed": seed, "rounds": rounds, "workload_hash": workload.digest(),
                           "runtime_seconds": "", "status": "failed", "error": "",
                           "vector_bytes": vector_bytes, "estimated_working_set_bytes": working_bytes,
                           "configured_limit_bytes": _memory.MAX_WORKING_SET_BYTES,
                           "backend_metadata": ""}
                    start = perf_counter()
                    try:
                        grids = replay_runtime_workload(backend, workload)
                        row["runtime_seconds"] = perf_counter() - start
                        row["status"] = "completed"
                        tableau_bytes = (grids[0].tableau.X.nbytes + grids[0].tableau.Z.nbytes
                                         + grids[0].tableau.tau_exp.nbytes) if hasattr(grids[0], "tableau") else None
                        row["backend_metadata"] = json.dumps({**backend.metadata,
                            "systems_per_player": grids[0].tableau.n if hasattr(grids[0], "tableau")
                            else grids[0].circuit.n,
                            "tableau_array_bytes_per_player": tableau_bytes,
                            "grid_count": len(grids), "workload_hash": workload.digest()}, sort_keys=True)
                    except MemoryError as exc:
                        row["status"] = "OOM"
                        row["error"] = f"{type(exc).__name__}: {exc}"
                    except Exception as exc:
                        row["error"] = f"{type(exc).__name__}: {exc}"
                    if not row["backend_metadata"]:
                        row["backend_metadata"] = json.dumps(backend.metadata, sort_keys=True)
                    rows.append(row)
                    print(f"runtime {name} m={m} rep={repetition}: {row['status']}", flush=True)
    finally:
        _write_csv(output / "runtime_raw.csv", fields, rows)
    summary = _summarize(rows, RUNTIME_METRICS, ("backend", "grid_size"), seed)
    _write_csv(output / "runtime_summary.csv", _summary_fields(("backend", "grid_size"), RUNTIME_METRICS), summary)
    (output / "runtime_metadata.json").write_text(json.dumps({
        "grid_sizes": sizes, "rounds": rounds, "dense_runtime_repetitions": dense_runtime_repetitions,
        "tableau_runtime_repetitions": tableau_runtime_repetitions, "master_seed": seed,
        "seed_derivation": "first 64 bits of SHA256('SnaQ3-v2|master|repetition|stream|grid_size')",
        "trace_seeds_and_hashes": trace_seeds,
        "trace_definition": "two random initial grids; independent uniform movement inputs; Grid.move and benchmark-only classical single-apple placement; Bell outcomes ignored for trace; dead snakes reinitialized before next round",
        "quantum_apple_preparation": "excluded: fixed traces are generated before timing with benchmark-only classical apple metadata; no U_H302 or apple measurement is timed",
        "resolver_seeds": "Random(derived resolver seed).getrandbits(64) in round order; shared across backends",
        "confidence_interval": f"percentile bootstrap of repetition means, {BOOTSTRAP_DRAWS} draws, deterministic derived seed shared at each grid size",
        "runtime_boundary": "replay_runtime_workload: full two-grid preparation, Bell preparation/sampling with classical direction labels, grid updates",
        "memory_guard": "two player vectors plus two workspace vectors; configured ceiling and half available memory",
        "max_statevector_gib": max_statevector_gib, "quickqudits_version": quickqudits.__version__}, indent=2) + "\n")
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", nargs="?", choices=("payoff", "runtime", "all"), default="all")
    parser.add_argument("--grid-sizes", nargs="+", type=int, default=[9, 16, 25])
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--shots-per-input", type=int, default=1000)
    parser.add_argument("--payoff-repetitions", type=int, default=10)
    parser.add_argument("--dense-runtime-repetitions", type=int, default=10)
    parser.add_argument("--tableau-runtime-repetitions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--output-directory", default="results")
    parser.add_argument("--max-statevector-gib", type=float, default=8.0)
    args = parser.parse_args()
    if args.experiment in ("payoff", "all"):
        run_payoff_experiment(args.shots_per_input, args.payoff_repetitions,
                              args.seed, args.output_directory)
    if args.experiment in ("runtime", "all"):
        run_runtime_experiment(args.grid_sizes, args.rounds, args.dense_runtime_repetitions,
                               args.tableau_runtime_repetitions, args.seed, args.output_directory,
                               args.max_statevector_gib)
    if args.experiment == "all":
        from .plotting import plot_results
        plot_results(Path(args.output_directory))


if __name__ == "__main__":
    main()
