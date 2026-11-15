"""Forecast failure time with a Gaussian process and report interval coverage.

Pass --plot to also draw the forecast for a few test engines (needs matplotlib).
"""

import argparse
from pathlib import Path

from sensor_edge.data import load_subset
from sensor_edge.experiment import ExperimentConfig
from sensor_edge.forecast import (
    N_SIGMA,
    forecast_test_engines,
    summarise_by_true_rul,
    summarise_forecasts,
)

PLOT_UNITS = [24, 34, 76, 82]  # fixed before any results were seen


def plot_forecasts(table, indicator, forecasts, frames_test, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
    for axis, unit in zip(axes.ravel(), PLOT_UNITS):
        cycles, health = indicator.trajectory(frames_test[frames_test["unit"] == unit])
        forecast = forecasts[unit]
        true_failure = cycles[-1] + table.set_index("unit").loc[unit, "true_rul"]
        axis.plot(cycles, health, ".", color="tab:blue", markersize=4, label="observed")
        axis.plot(forecast.future_cycles, forecast.future_mean, color="tab:orange", label="GP mean")
        axis.fill_between(
            forecast.future_cycles,
            forecast.future_mean - N_SIGMA * forecast.future_std,
            forecast.future_mean + N_SIGMA * forecast.future_std,
            color="tab:orange",
            alpha=0.25,
            label=f"±{N_SIGMA:.0f} std",
        )
        axis.axhline(indicator.failure_level_, color="tab:red", linestyle="--", label="failure level")
        axis.axvline(true_failure, color="black", linestyle=":", label="true failure")
        axis.set_title(f"Test engine {unit}")
        axis.set_xlabel("cycle")
        axis.set_xlim(0, cycles[-1] + 130)
    axes[0, 0].set_ylabel("health indicator")
    axes[1, 0].set_ylabel("health indicator")
    axes[0, 0].legend(loc="upper left", fontsize=8)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=110)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/cmapss"))
    parser.add_argument("--subset", default="FD001")
    parser.add_argument("--output", type=Path, default=Path("artifacts/uncertainty.md"))
    parser.add_argument("--plot", type=Path, default=None)
    args = parser.parse_args()

    config = ExperimentConfig()
    subset = load_subset(args.data_dir, args.subset)
    table, indicator, forecasts = forecast_test_engines(subset, config)
    summary = summarise_forecasts(table)

    report = (
        f"Gaussian process failure forecast on {args.subset}, interval of ±{N_SIGMA:.0f} std.\n\n"
        "| Engines | RMSE | NASA score | Interval coverage | Mean interval width | No crossing in horizon |\n"
        "|---|---|---|---|---|---|\n"
        f"| {summary['n_engines']} | {summary['rmse']:.2f} | {summary['nasa_score']:.0f} | "
        f"{summary['coverage']:.2f} | {summary['mean_width']:.1f} | {summary['censored_share']:.2f} |\n"
    )
    report += (
        "\nBy true RUL of the test engine (uses the official labels, for analysis only):\n\n"
        "| True RUL | Engines | RMSE | Interval coverage | Mean interval width | No crossing in horizon |\n"
        "|---|---|---|---|---|---|\n"
    )
    for row in summarise_by_true_rul(table):
        report += (
            f"| {row['true_rul_range']} | {row['n_engines']} | {row['rmse']:.2f} | "
            f"{row['coverage']:.2f} | {row['mean_width']:.1f} | {row['censored_share']:.2f} |\n"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report)

    if args.plot:
        from sensor_edge.experiment import normalise_subset

        plot_forecasts(table, indicator, forecasts, normalise_subset(subset, config).test, args.plot)
        print(f"Wrote {args.plot}")


if __name__ == "__main__":
    main()
