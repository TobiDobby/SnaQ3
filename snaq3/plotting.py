"""Separate resolver-payoff and full-grid runtime figures."""

import argparse
import csv
import json
from math import cos, pi
from pathlib import Path

MODES = ("qutrit_circuit_exact", "qutrit_tableau_clifford",
         "qubit_transpiled", "qubit_tableau_transpiled")
LABELS = ("Qutrit SV + Exact", "Qutrit Tableau + Clifford",
          "Qubit SV + Exact", "Qubit Tableau + Clifford")
COLORS = ("#176b9a", "#d17c23", "#27815a", "#a04e84")
MARKERS = ("o", "s", "D", "^")


def _load(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _payoff_axis(ax, summary):
    lookup = {row["backend"]: row for row in summary}
    for i, (mode, label, color, marker) in enumerate(zip(MODES, LABELS, COLORS, MARKERS)):
        row = lookup.get(mode)
        if not row or not row["mean_F_avg"]:
            continue
        value = float(row["mean_F_avg"])
        low = float(row["ci95_low_F_avg"]) if row["ci95_low_F_avg"] else value
        high = float(row["ci95_high_F_avg"]) if row["ci95_high_F_avg"] else value
        ax.errorbar(i, value, yerr=[[value-low], [high-value]], fmt=marker,
                    color=color, capsize=3, markersize=6, label=label)
    ax.axhline(.75 * cos(pi / 8)**2, color="0.25", ls="--", lw=1,
               label=r"Analytical $W_Q$")
    ax.axhline(9/16, color="0.5", ls=":", lw=1,
               label=r"Analytical $W_C$")
    ax.set_xticks(range(4), LABELS, rotation=20, ha="right")
    ax.set_xlim(-.45, 3.45)
    ax.set_ylabel(r"Average payoff $F_{\mathrm{avg}}$")
    ax.set_ylim(.48, .72)


def _runtime_axis(ax, summary, raw, sizes):
    lookup = {(row["backend"], int(row["grid_size"])): row for row in summary}
    oom = {(row["backend"], int(row["grid_size"])): row for row in raw
           if row["status"] == "OOM"}
    offsets = (-.36, -.12, .12, .36)
    for mode, label, color, marker, offset in zip(MODES, LABELS, COLORS, MARKERS, offsets):
        points = []
        for m in sizes:
            row = lookup.get((mode, m))
            if row and row["mean_runtime_seconds"]:
                points.append((m + offset, row))
        if points:
            x = [p[0] for p in points]
            y = [float(p[1]["mean_runtime_seconds"]) for p in points]
            lower = [float(p[1]["ci95_low_runtime_seconds"] or p[1]["mean_runtime_seconds"])
                     for p in points]
            upper = [float(p[1]["ci95_high_runtime_seconds"] or p[1]["mean_runtime_seconds"])
                     for p in points]
            ax.errorbar(x, y, yerr=([a-b for a,b in zip(y,lower)],
                                     [a-b for a,b in zip(upper,y)]),
                        marker=marker, color=color, capsize=2.5, lw=1.2,
                        label=label)
        for m in sizes:
            if (mode, m) in oom:
                ax.annotate("OOM", xy=(m + offset, .025), xycoords=("data", "axes fraction"),
                            ha="center", va="bottom", fontsize=7, color=color,
                            rotation=90)
    ax.set_yscale("log")
    ax.set_xticks(sizes)
    ax.set_xlim(min(sizes)-1, max(sizes)+1)
    ax.set_xlabel(r"Grid size $m$ (cells per player)")
    ax.set_ylabel("Runtime (s, log scale)")
    ax.grid(axis="y", alpha=.2, which="major")


def _save(fig, output, stem):
    for extension in ("pdf", "png"):
        fig.savefig(output / f"{stem}.{extension}", dpi=300)


def plot_payoff(output: Path) -> None:
    """Regenerate the final resolver plot from payoff_summary.csv alone."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    payoff = _load(output / "payoff_summary.csv")
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(7.0, 3.8), constrained_layout=True)
    _payoff_axis(ax, payoff)
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    _save(fig, output, "payoff_by_backend")
    plt.close(fig)


def plot_results(output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    payoff = _load(output / "payoff_summary.csv")
    runtime = _load(output / "runtime_summary.csv")
    runtime_raw = _load(output / "runtime_raw.csv")
    sizes = sorted({int(row["grid_size"]) for row in runtime})
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})

    fig, ax = plt.subplots(figsize=(7.0, 3.8), constrained_layout=True)
    _payoff_axis(ax, payoff)
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    _save(fig, output, "payoff_by_backend")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 3.8), constrained_layout=True)
    _runtime_axis(ax, runtime, runtime_raw, sizes)
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    _save(fig, output, "runtime_scaling")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(7.1, 6.8), constrained_layout=True)
    _payoff_axis(axes[0], payoff)
    _runtime_axis(axes[1], runtime, runtime_raw, sizes)
    axes[0].set_title("(a) Resolver payoff", fontsize=10)
    axes[1].set_title("(b) Full-grid runtime", fontsize=10)
    # Analytical reference lines stay in the payoff panel; the method legend is shared.
    handles, labels = axes[1].get_legend_handles_labels()
    payoff_handles, payoff_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles + payoff_handles[:2], labels + payoff_labels[:2],
               loc="outside lower center", ncol=3, fontsize=7)
    _save(fig, output, "evaluation_panels")
    plt.close(fig)
    (output / "plotting_metadata.json").write_text(json.dumps({
        "payoff_plot": "Resolver-only, balanced Z4 x Z4 input design; no grid-size dependence",
        "runtime_plot": "Two full player grids; log-scale measured runtime means with 95% confidence intervals",
        "confidence_interval": "Deterministic percentile bootstrap across independent repetitions; see benchmark metadata",
        "OOM": "Annotations indicate failed allocation guards; no runtime value is plotted or extrapolated",
        "analytical_references": {"W_Q": .75 * cos(pi / 8)**2, "W_C": 9 / 16}
    }, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", default="results")
    parser.add_argument("--payoff-only", action="store_true",
                        help="regenerate payoff_by_backend from payoff_summary.csv only")
    args = parser.parse_args()
    if args.payoff_only:
        plot_payoff(Path(args.output_directory))
    else:
        plot_results(Path(args.output_directory))


if __name__ == "__main__":
    main()
