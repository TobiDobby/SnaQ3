"""Runtime versus logical cells, merging playable and synthetic scaling data."""

import argparse
import csv
import json
from pathlib import Path

from .simulation import BACKENDS

BACKEND_ORDER = tuple(BACKENDS)
LABELS = {"qutrit_circuit_exact": "Qutrit SV",
          "qutrit_tableau_clifford": "Qutrit Tableau",
          "qubit_transpiled": "Qubit SV",
          "qubit_tableau_transpiled": "Qubit Tableau"}
COLORS = dict(zip(BACKEND_ORDER, ("#176b9a", "#d17c23", "#27815a", "#a04e84")))
MARKERS = dict(zip(BACKEND_ORDER, ("o", "s", "D", "^")))


def _read_optional(path):
    if not path.exists():
        return []
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def merged_rows(real_summary, synthetic_summary):
    """Prefer an existing full-grid point at the same backend and m."""
    combined = {}
    for row in synthetic_summary:
        combined[row["backend"], int(row["grid_cells_per_player"])] = row
    for row in real_summary:
        combined[row["backend"], int(row["grid_cells_per_player"])] = row
    return list(combined.values())


def plot_scaling(output: Path, summary: list[dict] | None = None,
                 synthetic_summary: list[dict] | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    output = Path(output)
    runtime_source = "runtime/runtime_scaling_summary.csv"
    if summary is None:
        summary = _read_optional(output / runtime_source)
        if not summary:
            runtime_source = "runtime_scaling_summary.csv"
            summary = _read_optional(output / runtime_source)
    if synthetic_summary is None:
        synthetic_summary = _read_optional(output / "dense_synthetic_summary.csv")
    rows = merged_rows(summary, synthetic_summary)
    if not rows:
        return
    cells = [int(row["grid_cells_per_player"]) for row in rows]
    measured = [float(row["mean_runtime_seconds"]) for row in rows
                if row["mean_runtime_seconds"] != ""]
    xlim = (min(cells) / 1.5, max(cells) * 1.5)
    ylim = (max(min(measured) / 30, 1e-9), max(measured) * 2) if measured else (1e-4, 1)
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    ax.add_patch(Rectangle((xlim[0], 0), xlim[1] - xlim[0], .11,
                           transform=ax.get_xaxis_transform(), facecolor="0.965",
                           edgecolor="none", zorder=0))
    for index, backend in enumerate(BACKEND_ORDER):
        backend_rows = sorted((row for row in rows if row["backend"] == backend),
                              key=lambda row: int(row["grid_cells_per_player"]))
        for row in backend_rows:
            m = int(row["grid_cells_per_player"])
            if row["mean_runtime_seconds"] != "":
                y = float(row["mean_runtime_seconds"])
                synthetic = str(row.get("synthetic_scaling_point", "False")).lower() == "true"
                face = "none" if synthetic else COLORS[backend]
                if (int(row["completed_repetitions"]) > 1 and row["CI95_low"] != ""
                        and row["CI95_high"] != ""):
                    ax.errorbar(m, y, yerr=[[y - float(row["CI95_low"])],
                                            [float(row["CI95_high"]) - y]],
                                fmt=MARKERS[backend], color=COLORS[backend],
                                markerfacecolor=face, markersize=5,
                                linestyle="none", capsize=2, elinewidth=1)
                else:
                    ax.plot(m, y, marker=MARKERS[backend], color=COLORS[backend],
                            markersize=5, linestyle="none", markerfacecolor=face)
            if int(row["oom_repetitions"]) > 0:
                ax.plot(m, .025 + .02 * index, marker="x", color=COLORS[backend],
                        markersize=5, linestyle="none", transform=ax.get_xaxis_transform(),
                        clip_on=False)
            if int(row["timeout_repetitions"]) > 0:
                ax.plot(m, .033 + .02 * index, marker="^", color=COLORS[backend],
                        markerfacecolor="none", markersize=5, linestyle="none",
                        transform=ax.get_xaxis_transform(), clip_on=False)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    # A continuous log axis keeps dense synthetic m values at their true positions.
    ax.set_xlabel("Logical qutrit cells per player $m$ (log scale)")
    ax.set_ylabel("Runtime (s, log scale)")
    ax.grid(axis="y", which="major", alpha=.2)
    handles = [Line2D([], [], marker=MARKERS[b], color=COLORS[b], ls="None",
                      markerfacecolor=COLORS[b], label=LABELS[b]) for b in BACKEND_ORDER]
    handles += [Line2D([], [], marker="o", color="0.35", markerfacecolor="0.35",
                       ls="None", label="Filled = playable grid"),
                Line2D([], [], marker="o", color="0.35", markerfacecolor="none",
                       ls="None", label="Hollow = synthetic"),
                Line2D([], [], marker="x", color="0.35", ls="None", label="× OOM"),
                Line2D([], [], marker="^", markerfacecolor="none", color="0.35",
                       ls="None", label="△ Timeout")]
    ax.legend(handles=handles, loc="upper right", ncol=1, fontsize=7,
              facecolor="white", framealpha=0.88)
    for extension in ("pdf", "png"):
        fig.savefig(output / f"runtime_vs_grid_size.{extension}", dpi=300)
    plt.close(fig)
    (output / "runtime_vs_grid_size_metadata.json").write_text(json.dumps({
        "x_axis": "logical qutrit cells per player m, logarithmic; synthetic sizes need not be square",
        "y_axis": "measured runtime in seconds, logarithmic",
        "sources": [runtime_source, "dense_synthetic_summary.csv"],
        "overlap_rule": "full-grid observation takes precedence over synthetic observation for the same backend and m",
        "successful_measurements": "backend-colored points; 95% CI only when at least two repetitions completed; no lines",
        "filled": "playable square SnaQ3 grid",
        "hollow": "synthetic register-size benchmark; representation scaling only, not a playable SnaQ3 game instance",
        "OOM": "backend-colored x in shaded status band at real m, not a runtime value",
        "TO": "backend-colored unfilled triangle in shaded status band at real m, not a runtime value",
        "interpretation": "computational representation cost only; payoff accuracy is a separate experiment",
    }, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=Path("results"))
    args = parser.parse_args()
    plot_scaling(args.output_directory)


if __name__ == "__main__":
    main()
