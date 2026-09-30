"""Paper-ready resolver payoff figure from the final summary CSV."""

import argparse
import csv
from math import cos, pi
from pathlib import Path

from .scaling_plotting import BACKEND_ORDER, COLORS, MARKERS

SHORT_LABELS = ("Qutrit SV", "Qutrit Tab", "Qubit SV", "Qubit Tab")


def plot_payoff(output: Path) -> None:
    """Generate the standalone payoff figure without running an experiment."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    with (output / "payoff_summary.csv").open(newline="") as file:
        summary = {row["backend"]: row for row in csv.DictReader(file)}

    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.35, 2.0), constrained_layout=True)
    for index, backend in enumerate(BACKEND_ORDER):
        row = summary[backend]
        mean = float(row["mean_F_avg"])
        low = float(row["ci95_low_F_avg"] or mean)
        high = float(row["ci95_high_F_avg"] or mean)
        ax.errorbar(index, mean, yerr=[[mean - low], [high - mean]],
                    fmt=MARKERS[backend], color=COLORS[backend],
                    markersize=5, capsize=2.5, linestyle="none")

    ax.axhline(.75 * cos(pi / 8) ** 2, color="0.3", ls="--", lw=1,
               label=r"Analytical $W_Q$")
    ax.axhline(9 / 16, color="0.5", ls=":", lw=1,
               label=r"Analytical $W_C$")
    ax.set_xticks(range(4), SHORT_LABELS, fontsize=7)
    ax.set_xlim(-.4, 3.4)
    ax.set_ylim(.5, .7)
    ax.set_ylabel(r"Average payoff $F_{\mathrm{avg}}$")
    ax.legend(loc="center right", fontsize=6.5, framealpha=.9)
    for extension in ("pdf", "png"):
        fig.savefig(output / f"payoff_by_backend.{extension}", dpi=300)
    plt.close(fig)


def plot_results(output: Path) -> None:
    """Retain the older benchmark CLI hook; final runtime has its own plotter."""
    plot_payoff(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=Path("results"))
    parser.add_argument("--payoff-only", action="store_true",
                        help="retained for existing reproduction commands")
    args = parser.parse_args()
    plot_payoff(args.output_directory)


if __name__ == "__main__":
    main()
