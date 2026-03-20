import streamlit as st
import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.strategy_service import StrategyService

st.set_page_config(page_title="F1 Pit Strategy Assistant", page_icon="🏁", layout="wide")

@st.cache_resource
def load_service():
    return StrategyService()

svc = load_service()
available_races = sorted(svc.df_fe["RaceId"].dropna().unique())

def format_race_label(race_id: str) -> str:
    return race_id.replace("SaoPaulo", "Brazil").replace("SãoPaulo", "Brazil")

display_race_map = {format_race_label(r): r for r in available_races}
display_races = sorted(display_race_map.keys(), reverse=True)

import pandas as pd
import streamlit as st

def show_pit_debug(svc, race_id: str, driver: str, pit_lap: int):
    st.markdown("### Debug: pit-window data")

    sub = svc.df_fe[
        (svc.df_fe["RaceId"] == race_id) &
        (svc.df_fe["Driver"] == driver)
    ].copy().sort_values("LapNumber")

    if sub.empty:
        st.error("No rows found for that race/driver.")
        return

    pit_col = None
    if "is_pit_lap" in sub.columns:
        pit_col = "is_pit_lap"
    elif "IsPitLap" in sub.columns:
        pit_col = "IsPitLap"

    pits = []
    if pit_col is not None:
        pits = sub.loc[sub[pit_col].astype(int) == 1, "LapNumber"].astype(int).tolist()

    cols_wanted = [
        "LapNumber", "LapTime_s", "Position", "Compound", "Stint", "StopsSoFar",
        "LapsSinceLastPit", "TyreWearFrac", "lap_avg_3", "TrackStatus",
        "is_pit_lap", "IsPitLap", "IsOutLap", "IsInLap"
    ]
    cols = [c for c in cols_wanted if c in sub.columns]

    window_df = sub[sub["LapNumber"].astype(int).between(int(pit_lap) - 3, int(pit_lap) + 4)][cols].copy()
    st.dataframe(window_df, use_container_width=True)

    # missing / duplicate lap checks
    sub["LapNumber_i"] = sub["LapNumber"].astype(int)
    laps = sorted(sub["LapNumber_i"].tolist())
    if laps:
        full = set(range(min(laps), max(laps) + 1))
        missing = sorted(full - set(laps))
        dupes = sub[sub["LapNumber_i"].duplicated(keep=False)]["LapNumber_i"].tolist()
    else:
        missing, dupes = [], []

    # quick pit-loss style diagnostic
    if pit_lap in sub["LapNumber_i"].values:
        around = sub[sub["LapNumber_i"].between(int(pit_lap)-2, int(pit_lap)+2)].copy()

st.title("F1 Pit Strategy Assistant")
st.caption("Explain real pit stops, run what-if simulations, and get pit recommendations.")

# -------- Sidebar: Context --------
with st.sidebar:
    st.header("Context")
    race_display = st.selectbox("Race", display_races)
    race_id = display_race_map[race_display]
    available_drivers = sorted(
        svc.df_fe.loc[svc.df_fe["RaceId"] == race_id, "Driver"].dropna().unique()
    )

    default_driver_index = 0
    if "VER" in available_drivers:
        default_driver_index = available_drivers.index("VER")

    driver = st.selectbox("Driver", available_drivers, index=default_driver_index)
    horizon = st.slider("Horizon (laps)", 5, 40, 20)
    st.divider()

    st.subheader("Mode inputs")
    pit_lap = st.number_input("Pit lap", min_value=1, max_value=200, value=28)
    window = st.slider("Explain window (± laps)", 1, 6, 2)
    debug_mode = st.checkbox("Show debug pit-window data", value=False)

# -------- Tabs --------
tab1, tab2, tab3 = st.tabs(["Explain real pit", "What-if sim", "Recommend"])

def show_quality_banner(payload: dict):
    q = payload.get("quality") or {}
    if q.get("interpolated_used"):
        st.warning(
            f"Telemetry gaps detected — interpolated laps used. "
            f"Missing laps: {q.get('missing_laps_count')} "
            f"(sample: {q.get('missing_laps_sample')})."
        )

def show_confidence_badge(out: dict):
    quality = out.get("quality", {}) or {}
    stop_ctx = out.get("stop_context", {}) or {}

    warnings = []

    if quality.get("interpolated_used"):
        warnings.append("interpolated lap data")

    if stop_ctx.get("weather_stop"):
        warnings.append("weather-related stop")

    if stop_ctx.get("short_gap_stop"):
        warnings.append("short-gap stop")

    cf = out.get("counterfactuals") or []
    if cf:
        vals = []
        for x in cf:
            try:
                vals.append(abs(float(x.get("delta_vs_actual_s", 0.0))))
            except Exception:
                pass
        if vals and max(vals) > 20:
            warnings.append("unstable nearby-lap comparison")

    if not warnings:
        st.success("Confidence: High")
    elif len(warnings) == 1:
        st.warning(f"Confidence: Medium — caution due to {warnings[0]}")
    else:
        st.warning("Confidence: Low — " + ", ".join(warnings))

def plot_sim(df: pd.DataFrame):
    if df is None or df.empty:
        st.write("No graph data available.")
        return

    st.subheader("Projected time gain/loss")
    st.line_chart(df.set_index("LapNumber")[["DeltaCum_s"]])
    st.caption("Negative means the strategy is gaining time. Positive means it is losing time.")

    st.subheader("Projected race position")
    pos_df = (
        df.set_index("LapNumber")[["WhatIfPos"]]
        .rename(columns={"WhatIfPos": "Projected position"})
    )
    st.line_chart(pos_df)
    st.caption("This shows the projected running position across the comparison window.")

def plot_recommend_sim(df: pd.DataFrame):
    if df is None or df.empty:
        st.write("No graph data available.")
        return

    st.subheader("Projected time gain/loss after pitting")
    st.line_chart(df.set_index("LapNumber")[["DeltaCum_s"]])
    st.caption("Negative means the projected pit strategy is gaining time. Positive means it is losing time.")

    st.subheader("Projected race position after pitting")
    pos_df = (
        df.set_index("LapNumber")[["WhatIfPos"]]
        .rename(columns={"WhatIfPos": "Projected position"})
    )
    st.line_chart(pos_df)
    st.caption("This shows the projected running position if the driver pits on the recommended lap.")

with tab2:
    st.subheader("🔁 What-if simulation")
    run = st.button("Run what-if", type="primary", use_container_width=True)
    if run:
        out = svc.simulate(race_id=race_id, driver=driver, pit_lap=int(pit_lap), horizon_laps=int(horizon))
        show_quality_banner(out)
        show_confidence_badge(out)

        st.markdown("### Result")
        st.success(out.get("summary_short", out.get("summary_viewer", out.get("summary", "Done."))))

        with st.expander("Why did the model say this?"):
            st.markdown(out.get("summary_detail", "No additional detail available."))

        with st.expander("See ranked nearby alternatives"):
            cf = out.get("counterfactuals")
            if cf:
                import pandas as pd
                st.dataframe(pd.DataFrame(cf), use_container_width=True)
            else:
                st.write("No nearby alternatives available.")

        with st.expander("See graphs"):
            df = out.get("df")
            if df is not None and len(df):
                plot_sim(df)
            else:
                st.write("No graph data available for this stop.")
        with st.expander("Show raw lap table"):
            df = out.get("df")
            if df is not None and len(df):
                st.dataframe(df, use_container_width=True)
            else:
                st.write("No lap table available.")

with tab3:
    st.subheader("Pit recommendation")
    current_lap = st.number_input("Current lap", min_value=1, max_value=200, value=18)

    if st.button("Recommend pit lap", type="primary", use_container_width=True):
        out = svc.recommend_pit_lap(
            race_id=race_id,
            driver=driver,
            current_lap=int(current_lap),
            candidate_laps=list(range(int(current_lap), int(current_lap) + 6)),
            horizon_laps=int(horizon),
        )

        show_quality_banner(out)
        show_confidence_badge(out)

        st.success(out.get("summary_short", out.get("summary_viewer", out.get("summary", "Done."))))

        with st.expander("Why this lap is recommended"):
            st.markdown(out.get("summary_detail", "No additional detail available."))

        with st.expander("See ranked nearby options"):
            cf = out.get("counterfactuals")
            if cf:
                cf_df = pd.DataFrame(cf)[["pit_lap", "delta_vs_best_s", "whatif_pos_end", "label"]].copy()
                cf_df["delta_vs_best_s"] = cf_df["delta_vs_best_s"].round(3)
                st.dataframe(cf_df, use_container_width=True)
            else:
                st.write("No nearby alternatives available.")

        with st.expander("See pit-window comparison"):
            cf = out.get("counterfactuals")
            if cf:
                cf_df = pd.DataFrame(cf)
                if "pit_lap" in cf_df.columns and "delta_vs_best_s" in cf_df.columns:
                    st.line_chart(cf_df.set_index("pit_lap")[["delta_vs_best_s"]])
                    st.caption("Zero is the recommended lap. Positive values are worse than the best option.")
                else:
                    st.write("No pit-window chart available.")
            else:
                st.write("No nearby alternatives available.")

        with st.expander("Advanced: projected race trace if the driver pits on the recommended lap"):
            best = out.get("best", {})
            df = best.get("df") if isinstance(best, dict) else None
            if df is not None and len(df):
                plot_recommend_sim(df)
            else:
                st.write("No projection data available.")

with tab1:
    st.subheader("Explain a real pit stop")

    if st.button("Explain pit", type="primary", use_container_width=True):
        out = svc.explain_real_pit(
            race_id=race_id,
            driver=driver,
            pit_lap=int(pit_lap),
            horizon_laps=int(horizon),
            window=int(window),
            prefer="viewer",
        )

        show_quality_banner(out)
        show_confidence_badge(out)

        st.info(out.get("summary_short", out.get("summary_viewer", out.get("summary", "Done."))))

        with st.expander("Why did the model say this?"):
            st.markdown(out.get("summary_detail", "No additional detail available."))

        with st.expander("See ranked nearby alternatives"):
            cf = out.get("counterfactuals")
            if cf:
                cf_df = pd.DataFrame(cf)[["pit_lap", "delta_vs_actual_s", "whatif_pos_end_s", "label"]].copy()
                cf_df["delta_vs_actual_s"] = cf_df["delta_vs_actual_s"].round(3)
                st.dataframe(cf_df, use_container_width=True)
            else:
                st.write("No nearby alternatives available.")

        with st.expander("See nearby timing comparison"):
            cf = out.get("counterfactuals")
            if cf:
                cf_df = pd.DataFrame(cf)
                if "pit_lap" in cf_df.columns and "delta_vs_actual_s" in cf_df.columns:
                    st.line_chart(cf_df.set_index("pit_lap")[["delta_vs_actual_s"]])
                    st.caption("Negative values are better than the actual stop. Positive values are worse.")
                else:
                    st.write("No nearby comparison chart available.")
            else:
                st.write("No nearby alternatives available.")

        with st.expander("See expected position by pit lap"):
            cf = out.get("counterfactuals")
            if cf:
                cf_df = pd.DataFrame(cf)
                if "pit_lap" in cf_df.columns and "whatif_pos_end_s" in cf_df.columns:
                    st.line_chart(cf_df.set_index("pit_lap")[["whatif_pos_end_s"]])
                    st.caption("This shows the expected position outcome for each nearby pit-lap option.")
                else:
                    st.write("No position chart available.")
            else:
                st.write("No nearby alternatives available.")

