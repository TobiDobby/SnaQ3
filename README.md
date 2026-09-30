# SnaQ3

SnaQ3 simulates two independent Snake grids of qutrit occupation states (`|0>` empty, `|1>` snake, `|2>` apple). Each player has a separate qubit movement register. A separate two-qubit Bell pair resolves conflicting directions. The four directions lift two CHSH measurement settings to inputs in `Z4`; the winning rule is `a XOR b = (x_A & x_B) & 1`. Only the Bell pair is entangled.

## Architecture

```text
player A qutrit grid ──┐
                       ├─ classical directions ─ Bell resolver ─ payoff ─ grid updates
player B qutrit grid ──┘          ↑
                         separate movement registers
```

Gameplay uses QuickQudits qutrit circuits to prepare `(|0>+|2>)/sqrt(2)` and measure **each empty cell independently** when apples are initialized or exhausted. QuickQudits 1.0.1 lacks `U_H302`; `grid.py` supplies that one-qutrit gate matrix. Snake cells are excluded. Consuming an apple adds one score point and grows the snake; a new batch is prepared only when no apples remain. The runtime benchmarks deliberately exclude this quantum apple preparation and replay a pre-generated, matched grid-update trace.

## Backends

| Backend | Player grid | Conflict resolver |
|---|---|---|
| `qutrit_circuit_exact` | Full `m`-qutrit QuickQudits statevector per player | Exact two-qubit circuit |
| `qutrit_tableau_clifford` | Full `m`-qutrit QuickQudits tableau per player | Clifford-approximated two-qubit tableau |
| `qubit_transpiled` | Full `2m`-qubit QuickQudits statevector per player | Same exact circuit |
| `qubit_tableau_transpiled` | Full `2m`-qubit QuickQudits tableau per player | Same Clifford tableau |

The transpilation encodes `|0>→|00>`, `|1>→|01>`, `|2>→|10>`; `|11>` is unused. Grid cells remain separable. The tableau Bell approximation maps Bob's two target settings to the same Clifford observable, so its payoff accuracy is evaluated separately from runtime. Dense memory failures and timeouts remain recorded results, never extrapolated runtimes.

## Installation and tests

The pinned dependencies in `requirements.txt` include **QuickQudits 1.0.1**, NumPy 2.5.3, and Matplotlib 3.11.2.

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
```

## Experiments

### Resolver payoff

`payoff_raw.csv` and `payoff_summary.csv` contain `F_A`, `F_B`, `F_avg`, and `p_win` for all four resolvers. Every repetition includes each of the 16 input pairs exactly 1,000 times. This experiment allocates no full player grids; payoff is independent of grid size.

```sh
.venv/bin/python -m snaq3.benchmark payoff \
  --shots-per-input 1000 --payoff-repetitions 10 --seed 20260929 \
  --output-directory results/reproduction
```

### Full-grid runtime scaling

This resumable experiment times two full player grids, movement preparation, Bell resolution, and fixed grid updates. One pre-generated trace is shared across backends at the same playable `n`; Bell outcomes do not change that trace. `n=2` for the qubit statevector is a marked synthetic scaling point. Pre-generating classical apple-valued trace data is outside the timer. OOM and timeout rows have no runtime value.

```sh
.venv/bin/python -m snaq3.scaling_benchmark \
  --qutrit-sv-n 3 4 5 --qubit-sv-n 2 3 4 \
  --qutrit-tableau-n 3 4 5 10 20 30 60 \
  --qubit-tableau-n 3 4 5 10 20 30 60 \
  --rounds 4 --dense-repetitions 5 --tableau-repetitions 20 \
  --seed 20260930 --max-memory-gib 32 --timeout-seconds 21600 \
  --output-directory results/reproduction/runtime --resume
```

### Supplementary dense scaling

This uses arbitrary logical cell counts `m`, including non-square values, to measure intermediate dense-statevector sizes. It times two full registers per backend. The workload is a fixed, matched logical update trace without playable board geometry. Run it **after** full-grid scaling in the same `results/reproduction/` tree. The synthetic benchmark reads the full-grid raw CSV from `results/reproduction/runtime/` to avoid duplicating measured square-grid points.

```sh
.venv/bin/python -m snaq3.dense_scaling_benchmark \
  --qutrit-m 4 6 8 9 10 12 14 16 17 18 19 \
  --qubit-m 4 5 6 7 8 9 10 11 12 13 14 15 \
  --rounds 4 --repetitions 5 --seed 20260930 \
  --max-memory-gib 32 --timeout-seconds 21600 \
  --output-directory results/reproduction --resume
```

Both long runtime commands checkpoint every observation. `--resume` skips recorded rows; `--retry-failed` explicitly retries failures. The reproduction directory is ignored by Git so the committed measurements remain intact.

## Results and plot regeneration

The final datasets are `results/payoff_raw.csv`, `results/payoff_summary.csv`, `results/runtime/runtime_scaling_raw.csv`, `results/runtime/runtime_scaling_summary.csv`, `results/dense_synthetic_raw.csv`, and `results/dense_synthetic_summary.csv`. Their companion metadata JSON files record seeds, rounds or shots, memory limits, QuickQudits version, and workload definitions. `results/clifford_observable_mapping.json` records the tableau angle mapping.

The resolver figure is `results/payoff_by_backend.{pdf,png}`. The runtime figure is `results/runtime_vs_grid_size.{pdf,png}`: filled markers are playable square grids, hollow markers are synthetic register sizes, and failure markers are not runtimes. Regenerate both directly from the retained summary CSVs, without executing a benchmark:

```sh
.venv/bin/python -m snaq3.plotting --payoff-only --output-directory results
.venv/bin/python -m snaq3.scaling_plotting --output-directory results
```
