"""Interactive demo: replay a NASA test engine and watch the RUL prediction evolve.

Run:  streamlit run demo/app.py      (after `rul-train --subset all`)
"""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from turbofan_rul import settings
from turbofan_rul.data import SENSOR_NAMES, load_test
from turbofan_rul.inference import RULPredictor, available_models, engine_trajectory

PREDICTED_COLOR = "#2a78d6"  # categorical slot 1 (blue)
ACTUAL_COLOR = "#8a8984"     # neutral: ground truth is a reference, not a competing series
MAINTENANCE_WINDOW = 50      # plan maintenance when predicted RUL drops below this
CRITICAL = 20
# Sensors with the clearest degradation signal.
KEY_SENSORS = {s: SENSOR_NAMES[s] for s in ("s11", "s4", "s12", "s7")}

st.set_page_config(page_title="Turbofan RUL demo", page_icon="✈️", layout="wide")


@st.cache_resource
def load_model(subset: str) -> RULPredictor:
    return RULPredictor.load(settings.MODEL_DIR / subset)


@st.cache_data
def load_engines(subset: str) -> tuple[pd.DataFrame, dict[int, int]]:
    df, rul = load_test(subset)
    return df, dict(zip(sorted(df["unit"].unique()), rul, strict=True))


@st.cache_data
def trajectory(subset: str, unit: int) -> pd.DataFrame:
    """Prediction at every observed cycle. Features are causal, so the prediction at
    cycle t is exactly what the model would have said with only cycles 1..t."""
    model = load_model(subset)
    df, final_rul = load_engines(subset)
    engine = df[df["unit"] == unit].sort_values("cycle")
    out = engine_trajectory(model, engine, final_rul[unit]).rename(
        columns={"predicted": "Predicted", "actual": "Actual"})
    feats = model.featurizer.transform(engine)
    for s in KEY_SENSORS:
        col = f"{s}_mean{model.featurizer.short_window}"
        out[s] = feats[col].to_numpy() if col in feats else np.nan
    return out


def show_status(pred: float, cap: int) -> None:
    """Status always carries an icon and a label, never colour alone."""
    if pred <= CRITICAL:
        st.error(f"**Critical** · predicted failure within {CRITICAL} cycles. Remove from service.", icon="🛑")
    elif pred <= MAINTENANCE_WINDOW:
        st.warning(f"**Plan maintenance** · inside the {MAINTENANCE_WINDOW}-cycle maintenance window.", icon="🔧")
    elif pred >= cap - 0.5:
        st.success(f"**Healthy** · no measurable degradation yet. The model caps predictions at {cap} "
                   f"cycles, because a healthy engine can't show whether 150 or 300 cycles remain.", icon="✅")
    else:
        st.success("**Healthy** · degrading, but outside the maintenance window.", icon="✅")


def rul_chart(traj: pd.DataFrame, now: int, cap: int) -> alt.LayerChart:
    wide = traj.assign(Predicted=traj["Predicted"].where(traj["cycle"] <= now))
    long = wide.melt("cycle", ["Predicted", "Actual"], var_name="series", value_name="rul").dropna()
    domain, colors = ["Predicted", "Actual"], [PREDICTED_COLOR, ACTUAL_COLOR]

    x = alt.X("cycle:Q", title="Operating cycle", scale=alt.Scale(nice=False))
    lines = alt.Chart(long).mark_line(strokeWidth=2).encode(
        x=x,
        y=alt.Y("rul:Q", title="Remaining useful life (cycles)", scale=alt.Scale(domainMin=0)),
        color=alt.Color("series:N", scale=alt.Scale(domain=domain, range=colors),
                        legend=alt.Legend(orient="top", title=None)),
        strokeDash=alt.StrokeDash("series:N", scale=alt.Scale(domain=domain, range=[[1, 0], [6, 4]]),
                                  legend=None),
    )
    hover = alt.selection_point(fields=["cycle"], nearest=True, on="pointerover",
                                clear="pointerout", empty=False)
    points = lines.mark_point(filled=True, size=70).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0)))
    crosshair = alt.Chart(wide).transform_calculate(
        predicted_label="isValid(datum.Predicted) ? format(datum.Predicted, '.0f') : '— (future)'"
    ).mark_rule(color=ACTUAL_COLOR, strokeWidth=1).encode(
        x=x,
        opacity=alt.condition(hover, alt.value(0.6), alt.value(0)),
        tooltip=[alt.Tooltip("cycle:Q", title="Cycle"),
                 alt.Tooltip("predicted_label:N", title="Predicted RUL"),
                 alt.Tooltip("Actual:Q", title="Actual RUL")],
    ).add_params(hover)

    window = pd.DataFrame({"y": [MAINTENANCE_WINDOW], "label": [f"Maintenance window ({MAINTENANCE_WINDOW})"]})
    window_rule = alt.Chart(window).mark_rule(color=ACTUAL_COLOR, strokeDash=[2, 3], opacity=0.7).encode(y="y:Q")
    window_text = alt.Chart(window).mark_text(align="left", dx=4, dy=-7, color=ACTUAL_COLOR).encode(
        x=alt.value(0), y="y:Q", text="label:N")
    now_df = pd.DataFrame({"cycle": [now], "label": ["now"]})
    now_rule = alt.Chart(now_df).mark_rule(color=PREDICTED_COLOR, strokeWidth=1, opacity=0.5).encode(x=x)
    now_text = alt.Chart(now_df).mark_text(align="left", dx=4, color=PREDICTED_COLOR).encode(
        x=x, y=alt.value(10), text="label:N")

    return alt.layer(window_rule, window_text, now_rule, now_text, lines, points, crosshair).properties(height=360)


def sensor_chart(traj: pd.DataFrame, sensor: str, now: int) -> alt.Chart:
    data = traj[traj["cycle"] <= now]
    # Fixed axes over the engine's whole history, so replaying shows the drift building up.
    lo, hi = float(traj[sensor].min()), float(traj[sensor].max())
    return alt.Chart(data).mark_line(strokeWidth=2, color=PREDICTED_COLOR).encode(
        x=alt.X("cycle:Q", title=None, scale=alt.Scale(domain=[1, int(traj["cycle"].max())], nice=False)),
        y=alt.Y(f"{sensor}:Q", title=None, scale=alt.Scale(domain=[lo, hi], nice=True)),
        tooltip=[alt.Tooltip("cycle:Q", title="Cycle"), alt.Tooltip(f"{sensor}:Q", title="z-score", format=".2f")],
    ).properties(height=150)


# --- sidebar ------------------------------------------------------------------
subsets = available_models()
if not subsets:
    st.error("No trained models found. Run `rul-train --subset all` first.")
    st.stop()

with st.sidebar:
    st.header("Engine")
    subset = st.selectbox("Dataset", subsets,
                          format_func=lambda s: f"{s} · {settings.SUBSET_DESCRIPTIONS.get(s, '')}")
    engines, final_rul = load_engines(subset)
    lengths = engines.groupby("unit")["cycle"].max()
    # Default to the engine with the longest history: the clearest degradation story.
    units = list(lengths.sort_values(ascending=False).index)
    unit = st.selectbox("Test engine", units,
                        format_func=lambda u: f"Engine {u} · {lengths[u]} cycles observed")
    traj = trajectory(subset, unit)
    last = int(traj["cycle"].max())
    now = st.slider("Replay up to cycle", 1, last, last,
                    help="The model only sees cycles up to here, as it would in service.")
    st.caption("These engines are from NASA's held-out test set: the model never saw them in training. "
               "The actual RUL is known because NASA published each engine's remaining life.")

model = load_model(subset)
meta, cap = model.metadata, model.rul_cap
row = traj[traj["cycle"] == now].iloc[0]
pred, actual = float(row["Predicted"]), int(row["Actual"])

# --- main ---------------------------------------------------------------------
st.title("Turbofan remaining useful life")
st.caption(f"{subset} · engine {unit} · cycle {now} of {last} observed · model: "
           f"{meta['model_name']} ({meta['model_version']})")

c1, c2, c3 = st.columns(3)
c1.metric("Predicted RUL (cycles)", f"{pred:.0f}" + ("+" if model.is_at_cap(pred) else ""))
c2.metric("Actual RUL (cycles)", f"{actual}")
if actual > cap:
    c3.metric("Error (cycles)", "—", help=f"Actual RUL is above the {cap}-cycle cap, so there is "
              "nothing to score yet: the model can only say 'healthy'.")
else:
    c3.metric("Error (cycles)", f"{pred - actual:+.0f}", help="Predicted minus actual. Positive = "
              "predicted too late, the costly direction.")
show_status(pred, cap)

st.altair_chart(rul_chart(traj, now, cap), use_container_width=True, theme="streamlit")

st.subheader("Key sensors")
st.caption(f"Normalised for the operating condition and smoothed over {model.featurizer.short_window} "
           "cycles: the inputs the model actually sees. Degradation shows up as a drift away from zero.")
cols = st.columns(len(KEY_SENSORS))
for col, sensor in zip(cols, KEY_SENSORS, strict=True):
    col.markdown(f"**{KEY_SENSORS[sensor]}** `{sensor}`")
    col.altair_chart(sensor_chart(traj, sensor, now), use_container_width=True, theme="streamlit")

with st.expander("Data table"):
    st.dataframe(traj[traj["cycle"] <= now].set_index("cycle"), use_container_width=True)

with st.expander(f"How good is the {subset} model?"):
    t = meta["test_metrics"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test RMSE", f"{t['rmse']:.1f}")
    m2.metric("RMSE, last 50 cycles", f"{t['rmse_last_50_cycles']:.1f}")
    m3.metric("Within ±15 cycles", f"{t['within_15_cycles_pct']:.0f}%")
    m4.metric("Late predictions", f"{t['late_predictions_pct']:.0f}%")
    st.markdown(
        f"Scored on all {t['n_engines']} test engines at their last observed cycle. Close to failure "
        "the error is much smaller than overall, because early-life RUL is capped and inherently "
        "uncertain. Model chosen by engine-grouped cross-validation:"
    )
    st.dataframe(pd.DataFrame(meta["model_comparison"]).T.rename_axis("model"), use_container_width=True)
