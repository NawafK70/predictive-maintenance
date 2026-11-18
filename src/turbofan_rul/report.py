"""Generate the results report and figures from the trained models.

CLI:  rul-report      (after rul-train --subset all)

Writes reports/results.md, reports/metrics.json and reports/figures/*.png.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from turbofan_rul import settings  # noqa: E402
from turbofan_rul.data import SENSOR_NAMES, load_test, load_train  # noqa: E402
from turbofan_rul.evaluate import NEAR_FAILURE, last_cycle_positions  # noqa: E402
from turbofan_rul.inference import RULPredictor, engine_trajectory, load_all  # noqa: E402

BLUE = "#2a78d6"     # the model's predictions (categorical slot 1)
GRAY = "#8a8984"     # ground truth / reference lines
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight", "figure.facecolor": "white",
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


def fig_regime_normalisation(model: RULPredictor, path) -> None:
    """Why per-regime normalisation matters: one FD002 engine, raw vs normalised s11."""
    engine = load_train(model.subset).query("unit == 1").sort_values("cycle")
    z = model.featurizer.normalize(engine)["s11"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.2))
    a.scatter(engine["cycle"], engine["s11"], s=10, color=BLUE, alpha=0.8, linewidths=0)
    a.set(title="Raw: six flight regimes bury the signal",
          xlabel="Operating cycle", ylabel=f"{SENSOR_NAMES['s11']} (s11)")
    b.scatter(engine["cycle"], z, s=10, color=BLUE, alpha=0.35, linewidths=0)
    b.plot(engine["cycle"], z.rolling(20, min_periods=1).mean(), color=BLUE, lw=2)
    b.set(title="Normalised within regime: the wear trend appears",
          xlabel="Operating cycle", ylabel="z-score within regime")
    fig.suptitle(f"{model.subset}, training engine 1, sensor s11 (line: 20-cycle rolling mean)",
                 x=0.01, y=1.04, ha="left", fontsize=10, color=INK_2)
    fig.savefig(path)
    plt.close(fig)


def fig_predicted_vs_actual(models: dict[str, RULPredictor], path) -> None:
    fig, axes = plt.subplots(1, len(models), figsize=(3 * len(models), 3.2), sharey=True)
    for ax, (subset, model) in zip(axes, models.items(), strict=True):
        test, true_rul = load_test(subset)
        pred = model.predict(test)[last_cycle_positions(test)]
        top = max(true_rul.max(), model.rul_cap) + 10
        ax.plot([0, top], [0, top], color=GRAY, lw=1, ls="--")
        ax.axhline(model.rul_cap, color=GRAY, lw=1, ls=":")
        ax.scatter(true_rul, pred, s=16, color=BLUE, alpha=0.7, linewidths=0)
        ax.set(title=f"{subset}  ·  RMSE {model.metadata['test_metrics']['rmse']:.1f}",
               xlabel="Actual RUL (cycles)", xlim=(0, top), ylim=(0, top))
        ax.text(top - 4, model.rul_cap + 4, f"cap {model.rul_cap}", ha="right", color=INK_2, fontsize=8)
    axes[0].set_ylabel("Predicted RUL (cycles)")
    fig.suptitle("Test engines at their last observed cycle (dashed: perfect prediction)",
                 x=0.01, y=1.04, ha="left", fontsize=10, color=INK_2)
    fig.savefig(path)
    plt.close(fig)


def fig_engine_trajectory(model: RULPredictor, path) -> int:
    test, true_rul = load_test(model.subset)
    unit = int(test.groupby("unit")["cycle"].max().idxmax())  # longest observed history
    final = dict(zip(sorted(test["unit"].unique()), true_rul, strict=True))[unit]
    traj = engine_trajectory(model, test[test["unit"] == unit], final)
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.plot(traj["cycle"], traj["actual"], color=GRAY, lw=2, ls="--", label="Actual")
    ax.plot(traj["cycle"], traj["predicted"], color=BLUE, lw=2, label="Predicted")
    ax.axhline(NEAR_FAILURE, color=GRAY, lw=1, ls=":")
    ax.text(traj["cycle"].min(), NEAR_FAILURE + 4, f"maintenance window ({NEAR_FAILURE} cycles)",
            color=INK_2, fontsize=8)
    end = traj.iloc[-1]
    # Label in the empty area above the curves, pointing down at the final cycle.
    ax.annotate(f"last cycle: predicted {end['predicted']:.0f}, actual {end['actual']:.0f}",
                (end["cycle"], end["predicted"]), xytext=(end["cycle"], 0.55 * traj["actual"].max()),
                ha="right", color=INK, fontsize=8,
                arrowprops={"arrowstyle": "-", "color": INK_2, "lw": 0.8})
    ax.set(title=f"Replaying test engine {unit} ({model.subset}): the prediction at each cycle uses only "
                 "the cycles before it", xlabel="Operating cycle", ylabel="Remaining useful life (cycles)",
           ylim=(0, None))
    ax.legend(loc="upper center", ncols=2)
    fig.savefig(path)
    plt.close(fig)
    return unit


def results_markdown(models: dict[str, RULPredictor]) -> str:
    rows = []
    for subset, m in models.items():
        t, meta = m.metadata["test_metrics"], m.metadata
        rows.append({
            "Subset": subset, "Conditions / faults": settings.SUBSET_DESCRIPTIONS[subset],
            "Model": meta["model_name"], "CV RMSE": meta["model_comparison"][meta["model_name"]]["cv_rmse"],
            "Test RMSE": t["rmse"], "RMSE vs capped truth": t["rmse_vs_capped_truth"],
            "RMSE last 50 cycles": t["rmse_last_50_cycles"], "NASA score": round(t["nasa_score"]),
            "Within ±15 cycles": f"{t['within_15_cycles_pct']:.0f}%",
        })
    cv = pd.DataFrame({s: {k: v["cv_rmse"] for k, v in m.metadata["model_comparison"].items()}
                       for s, m in models.items()}).rename_axis("CV RMSE by model")
    return ("# Results\n\nGenerated by `rul-report` from `models/*/metadata.json`.\n\n"
            "## Test set (NASA official, last observed cycle per engine)\n\n"
            + pd.DataFrame(rows).to_markdown(index=False, floatfmt=".2f")
            + "\n\n## Model comparison (engine-grouped 5-fold CV RMSE)\n\n"
            + cv.to_markdown(floatfmt=".2f") + "\n")


def main() -> None:
    models = load_all()
    if not models:
        raise SystemExit("No trained models; run rul-train --subset all first.")
    figures = settings.REPORTS_DIR / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    six_regime = next((m for s, m in models.items() if settings.SUBSETS[s] > 1), None)
    if six_regime:
        fig_regime_normalisation(six_regime, figures / "regime_normalisation.png")
    fig_predicted_vs_actual(models, figures / "predicted_vs_actual.png")
    first = models[min(models)]
    unit = fig_engine_trajectory(first, figures / "engine_trajectory.png")

    (settings.REPORTS_DIR / "results.md").write_text(results_markdown(models), encoding="utf-8")
    metrics = {s: {"model": m.metadata["model_name"], "model_version": m.version,
                   "cv": m.metadata["model_comparison"], "test": m.metadata["test_metrics"]}
               for s, m in models.items()}
    (settings.REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Wrote {settings.REPORTS_DIR / 'results.md'}, metrics.json and figures "
          f"(trajectory: {first.subset} engine {unit}) to {figures}")


if __name__ == "__main__":
    main()
