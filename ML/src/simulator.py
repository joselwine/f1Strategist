# src/simulator.py
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Iterable, List, Optional, Tuple, Dict

import numpy as np
import pandas as pd


# ----------------------------
# Tyre wear helpers
# ----------------------------
WEAR_CAP_DEFAULT = {"SOFT": 18, "MEDIUM": 25, "HARD": 35}


def wear_frac(tyre_life: float, compound: str, wear_cap: Dict[str, int] = WEAR_CAP_DEFAULT) -> float:
    """Return a 0..1 wear fraction based on tyre life and a simple compound cap."""
    cap = wear_cap.get(str(compound).upper(), 25)
    if tyre_life is None or (isinstance(tyre_life, float) and np.isnan(tyre_life)):
        return np.nan
    return float(np.clip(float(tyre_life) / float(cap), 0.0, 1.0))


def _mean(dq: deque) -> float:
    return float(np.mean(dq)) if len(dq) else np.nan


# ----------------------------
# Data selection / validation
# ----------------------------
def get_driver_race_laps(df: pd.DataFrame, race_id: str, driver_code: str) -> pd.DataFrame:
    """Filter and sort per-lap rows for a single driver in a single race."""
    sub = df[(df["RaceId"] == race_id) & (df["Driver"] == driver_code)].copy()
    if sub.empty:
        raise ValueError(f"No data found for driver={driver_code} race_id={race_id}.")
    sub = sub.sort_values("LapNumber").reset_index(drop=True)
    return sub


def pick_same_compound(sub: pd.DataFrame, pit_lap: int) -> str:
    """
    Choose the compound we assume the driver will be on after the pit.
    For a 'same compound pit', we use the compound at pit_lap if available,
    otherwise fall back to the mode.
    """
    comp = sub["Compound"].mode().iloc[0]
    if (sub["LapNumber"] == pit_lap).any():
        tmp = sub.loc[sub["LapNumber"] == pit_lap, "Compound"].mode()
        if len(tmp):
            comp = tmp.iloc[0]
    return comp


def require_features_exist(df: pd.DataFrame, features: List[str]) -> None:
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(
            "Simulator was asked to use features that are not in the dataframe.\n"
            f"Missing: {missing}\n"
            "Fix: ensure your df_fe has these engineered features, and pass the same list used in training."
        )


def infer_total_laps(sub: pd.DataFrame) -> int:
    return int(sub["LapNumber"].max())


def neutralise_pit_event(what_laptime, drv, pitlap, slow_thresh=8.0, max_window=2):
    # normal pace proxy from previous clean laps
    pre = drv[(drv["LapNumber"] >= pitlap - 4) & (drv["LapNumber"] <= pitlap - 1)].copy()
    pre["LapTime_s"] = pd.to_numeric(pre["LapTime_s"], errors="coerce")
    pre = pre.dropna(subset=["LapTime_s"])

    for col in ["is_pit_lap", "IsOutLap", "IsInLap"]:
        if col in pre.columns:
            pre = pre[pre[col].astype(int) == 0]
    if "TrackStatus" in pre.columns:
        pre = pre[pre["TrackStatus"].astype(int) == 1]

    prev_lap = max(1, int(pitlap) - 1)
    repl = float(pre["LapTime_s"].median()) if len(pre) else float(what_laptime.get(prev_lap, np.nan))
    if not np.isfinite(repl):
        repl = float(what_laptime.dropna().median())

    # build event window
    event_laps = [pitlap]
    for k in range(1, max_window + 1):
        L = pitlap + k
        if L not in what_laptime.index:
            break
        rr = drv.loc[drv["LapNumber"].astype(int) == L]
        if len(rr):
            rr = rr.iloc[0]
            flagged = int(rr.get("IsOutLap", 0)) == 1 or int(rr.get("IsInLap", 0)) == 1
            slow = float(rr["LapTime_s"]) > repl + slow_thresh
            if flagged or slow:
                event_laps.append(L)

    # neutralise
    what_laptime = what_laptime.copy()
    for L in event_laps:
        if L in what_laptime.index:
            what_laptime.loc[L] = repl

    return what_laptime, event_laps, repl


# ----------------------------
# Core simulator
# ----------------------------
def simulate_what_if_pit_stateful(
    df: pd.DataFrame,
    pace_model,
    race_id: str,
    driver_code: str,
    pit_lap: int,
    pit_loss_s: float = 22.0,
    horizon_laps: int = 25,
    pace_features: Optional[List[str]] = None,
    wear_cap: Dict[str, int] = WEAR_CAP_DEFAULT,
) -> pd.DataFrame:
    """
    Stateful 'what-if pit' simulator.

    Baseline:
      - Uses the real race rows as state templates
      - Predicts baseline lap time using pace_model at each lap

    What-if:
      - Forces a pit at pit_lap
      - Resets tyre age (TyreLife / LapsSinceLastPit) from pit_lap onward
      - Keeps the compound the same (configurable logic)
      - Adds a one-off pit loss to the pit lap
      - Updates rolling features (e.g., lap_avg_3) using predicted values,
        so later-lap predictions can change dynamically.

    Returns:
      DataFrame with LapNumber, predicted baseline lap time, predicted what-if lap time,
      per-lap delta, cumulative delta.
    """
    if pace_features is None:
        raise ValueError("Pass pace_features exactly as used to train pace_model (same order).")

    sub = get_driver_race_laps(df, race_id, driver_code)

    # Ensure required columns exist for feature computation
    require_features_exist(sub, [c for c in pace_features if c not in ("lap_avg_3", "TyreWearFrac")])

    max_lap = infer_total_laps(sub)
    end_lap = min(pit_lap + horizon_laps - 1, max_lap)
    if pit_lap > max_lap:
        raise ValueError(f"pit_lap={pit_lap} is beyond last lap={max_lap} for {driver_code} in {race_id}.")
    final_lap = max_lap

    comp_after = pick_same_compound(sub, pit_lap)

    # Seed rolling window with REAL LapTime_s values before pit lap if available
    # This helps lap_avg_3 behave sensibly at the first simulated laps.
    base_q = deque(maxlen=3)
    what_q = deque(maxlen=3)
    if "LapTime_s" in sub.columns:
        pre = sub[sub["LapNumber"].between(pit_lap - 3, pit_lap - 1)]
        seed = pre["LapTime_s"].dropna().tolist()
        base_q.extend(seed)
        what_q.extend(seed)

    rows = []

    for lap in range(pit_lap, end_lap + 1):
        row_real = sub.loc[sub["LapNumber"] == lap].iloc[0].copy()

        # -------------------------
        # Baseline row (real state template)
        # -------------------------
        base_row = row_real.copy()

        if "lap_avg_3" in base_row.index:
            base_row["lap_avg_3"] = _mean(base_q)

        if "TyreWearFrac" in base_row.index:
            base_row["TyreWearFrac"] = wear_frac(
                base_row.get("TyreLife", np.nan),
                base_row.get("Compound", "MEDIUM"),
                wear_cap=wear_cap,
            )

        # -------------------------
        # What-if row (apply pit + reset state)
        # -------------------------
        what_row = row_real.copy()

        # Force compound after pit (same compound scenario)
        if "Compound" in what_row.index:
            what_row["Compound"] = comp_after

        # Reset tyre age after pit
        rel = lap - pit_lap
        if "LapsSinceLastPit" in what_row.index:
            what_row["LapsSinceLastPit"] = 0 if rel == 0 else rel
        if "TyreLife" in what_row.index:
            what_row["TyreLife"] = 1 if rel == 0 else (rel + 1)

        # Bump stint/stops
        if "Stint" in what_row.index:
            what_row["Stint"] = what_row["Stint"] + 1
        if "StopsSoFar" in what_row.index:
            what_row["StopsSoFar"] = what_row["StopsSoFar"] + 1

        # Rolling feature (use what-if queue)
        if "lap_avg_3" in what_row.index:
            what_row["lap_avg_3"] = _mean(what_q)

        # Update TyreWearFrac for what-if state AFTER updating TyreLife/Compound
        if "TyreWearFrac" in what_row.index:
            what_row["TyreWearFrac"] = wear_frac(
                what_row.get("TyreLife", np.nan),
                what_row.get("Compound", "MEDIUM"),
                wear_cap=wear_cap,
            )


        # -------------------------
        # Predict pace
        # -------------------------
        Xw = pd.DataFrame([what_row])[pace_features]

        # What-if: model predicts LapDelta
        what_delta = float(pace_model.predict(Xw)[0])

        # Baseline: use REAL lap time from the dataset
        base_pred_time = float(row_real["LapTime_s"])

        # Convert what-if delta -> absolute lap time using the what-if lap_avg_3 anchor
        what_lap_avg = float(what_row.get("lap_avg_3", np.nan))
        if not np.isfinite(what_lap_avg):
            # fallback anchor if lap_avg_3 missing/NaN
            what_lap_avg = base_pred_time

        what_pred_time = what_lap_avg + what_delta

        # Apply pit loss once on pit lap
        if lap == pit_lap:
            what_pred_time += pit_loss_s



        # Apply pit loss once on pit lap (in lap-time seconds)
        if lap == pit_lap:
            what_pred_time += pit_loss_s

        # Update rolling queues using predicted LAP TIMES (not deltas)
        base_q.append(base_pred_time)
        what_q.append(what_pred_time)

        rows.append({
            "LapNumber": lap,
            "PredLapTime_Base": base_pred_time,        # now REAL seconds (90–110 etc)
            "PredLapTime_WhatIf": what_pred_time,      # now REAL seconds too
            "Delta_WhatIf_minus_Base": what_pred_time - base_pred_time
        })


    out = pd.DataFrame(rows)
    out["CumulativeDelta"] = out["Delta_WhatIf_minus_Base"].cumsum()
    out["RaceId"] = race_id
    out["Driver"] = driver_code
    out["PitLap"] = pit_lap
    out["PitLoss_s"] = pit_loss_s
    out["HorizonLaps"] = horizon_laps

    return out
# src/simulator.py

import numpy as np
import pandas as pd

TRAFFIC_GAP_S = 1.5  # affected by dirty air
SIM_WINDOW = 8
STABILISE_LAPS = 3  # 2–4 is realistic


def _build_cumtime_table(race_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a wide table:
      index = LapNumber
      columns = Driver
      values = cumulative race time at end of that lap
    """
    tmp = race_df.sort_values(["Driver", "LapNumber"]).copy()

    # If LapTime_s has weird outliers (e.g., red flags), you may later filter those laps
    tmp["LapTime_s"] = pd.to_numeric(tmp["LapTime_s"], errors="coerce")
    tmp = tmp.dropna(subset=["LapTime_s"])

    tmp["CumTime_s"] = tmp.groupby("Driver")["LapTime_s"].cumsum()

    wide = tmp.pivot(index="LapNumber", columns="Driver", values="CumTime_s").sort_index()
    return wide


def simulate_pit_rejoin_and_traffic(
    df: pd.DataFrame,
    pace_model,
    race_id: str,
    driver_code: str,
    pit_lap: int,
    horizon_laps: int = 20,
    pit_loss_s: float = 22.0,
    pace_features: list | None = None,
    traffic_gap_s: float = TRAFFIC_GAP_S,
    traffic_loss_s: float = 0.6,
    pass_advantage_s: float = 0.4,
    cancel_real_pit: bool = True,
    post_pit_push_laps: int = 0,
    post_pit_push_boost_s: float = 0.0,   # subtract this many seconds (faster)
    push_decay: str = "linear",   # "linear" or "exp"
    push_exp_k: float = 0.7,      # only used if push_decay="exp"
    use_real_post_pit: bool = True,
    verbose: bool = False,
    real_pit_reference_lap : int | None = None
):
    
    """
    Race-aware what-if:
      - Baseline for all drivers uses REAL LapTime_s.
      - For chosen driver, overwrite lap times in a window with model-based predictions.
      - If overcut (pit later than real pit), we simulate from the REAL pit lap, and
        force the state to look like "stayed out" until the what-if pit lap.
      - Add pit loss only on the what-if pit lap.
      - Recompute position each lap using cumulative time.
      - Traffic penalty: if within traffic_gap_s and not enough pace advantage to pass, add traffic_loss_s.

    Returns a per-lap table for lap range [pit_lap..end_lap] with:
      positions, cum times, delta, penalties, and debug columns.
    """
    # ---- debug stores (loop-time) ----
    gap_used = {}
    ahead_used = {}
    pace_adv_used = {}
    penalty_used = {}
    pos_used = {}
    push_boost_used = {}
    # ---- debug stores (prediction-time) ----
    ref_used = {}
    delta_used = {}
    pred_lt_used = {}
    push_used = {}
    tyrelife_used = {}
    lsp_used = {}
    tyre_adj_used = {}

    
    if pace_features is None:
        raise ValueError("Pass pace_features (exact columns pace_model expects).")

    race = df[df["RaceId"] == race_id].copy()
    if race.empty:
        raise ValueError(f"No data for race {race_id}")
    
    drv = race[race["Driver"] == driver_code].sort_values("LapNumber").copy()
    if drv.empty:
        raise ValueError(f"No data for driver {driver_code} in race {race_id}")
 

    # ---- Consistent int lap key ----
    race["LapNumber_i"] = race["LapNumber"].astype(int)
    drv["LapNumber_i"]  = drv["LapNumber"].astype(int)

    max_lap_race = int(race["LapNumber_i"].max())
    end_lap = min(int(pit_lap) + int(horizon_laps), max_lap_race)
    delta_lock_lap = min(int(pit_lap) + STABILISE_LAPS, end_lap)

    # ---- Baseline cumulative times for all drivers (REAL) ----
    base_cum = _build_cumtime_table(race)
    if driver_code not in base_cum.columns:
        raise ValueError(f"{driver_code} not found in race {race_id}")

    # ---- Build continuous lap-time Series for the driver (fixes KeyError 10 / missing laps) ----
    full_laps = pd.Index(
        range(1, max_lap_race + 1), 
        name="LapNumber_i"
    )

    real_laptime = (
        drv.set_index("LapNumber_i")["LapTime_s"]
        .astype(float)
        .reindex(full_laps)
        .interpolate(limit_direction="both")
        .ffill()
        .bfill()
    )

    # after drv and max_lap_race are known, before interpolate
    raw = drv.set_index("LapNumber_i")["LapTime_s"].astype(float)
    full_laps = pd.Index(range(1, max_lap_race + 1), name="LapNumber_i")

    missing_laps = sorted(set(full_laps) - set(raw.index))
    missing_count = len(missing_laps)

    real_laptime = (
        raw.reindex(full_laps)
        .interpolate(limit_direction="both")
        .ffill()
        .bfill()
    )

    # later when returning out dataframe, attach metadata:
    quality = {
        "missing_laps_count": missing_count,
        "missing_laps_sample": missing_laps[:10],
        "interpolated_used": missing_count > 0,
    }

    what_laptime = real_laptime.copy()  # we overwrite simulated laps later
    drv_laps_set = set(full_laps.tolist())  # now every lap exists

    # --- Detect real pit lap (first pit) if available
    real_pit_lap = None
    real_pits = []
    if cancel_real_pit and ("is_pit_lap" in drv.columns):
        real_pits = drv.loc[
            drv["is_pit_lap"].astype(int) == 1, "LapNumber"
            ].astype(int).tolist()
        if real_pit_reference_lap is not None:
            if int(real_pit_reference_lap) in real_pits:
                real_pit_lap = int(real_pit_reference_lap)
            else:
                raise ValueError(
                    f"real_pit_reference_lap={real_pit_reference_lap} not found in real pit laps {real_pits}"
                )
        else:
            real_pit_lap = min(real_pits) if real_pits else None

    rp = real_pit_lap
    wp = pit_lap

    # --- detect where the "pit loss" actually sits in THIS driver's data (pit lap vs outlap) ---
    rp_loss_lap = None
    pitloss_offset = 0

    if rp is not None:
        lt_rp = float(real_laptime.loc[int(rp)])
        lt_next = float(real_laptime.loc[int(rp) + 1]) if int(rp) + 1 <= max_lap_race else np.nan

        if np.isfinite(lt_next) and (lt_next > lt_rp + 8.0):
            pitloss_offset = 1
            rp_loss_lap = int(rp) + 1
        else:
            pitloss_offset = 0
            rp_loss_lap = int(rp)

    wp_loss_lap = int(wp) + int(pitloss_offset)


    if (rp_loss_lap is not None) and (wp_loss_lap != rp_loss_lap):
        what_laptime, real_event_laps, repl = neutralise_pit_event(what_laptime, drv, rp)
        real_loss = (
            sum(float(real_laptime.loc[int(L)]) for L in real_event_laps)
            - float(repl) * len(real_event_laps)
        )
        real_loss = max(0.0, real_loss)  # safety


    # only predict from the WHAT-IF pit onwards (both undercut + overcut)
    laps_to_predict = list(range(wp, end_lap + 1))


    laps_to_predict = sorted(set(int(x) for x in laps_to_predict))

    first_sim_lap = int(laps_to_predict[0])
    sim_rows = drv[drv["LapNumber_i"].astype(int).isin(laps_to_predict)].copy().sort_values("LapNumber_i")
    sim_laps = sim_rows["LapNumber_i"].astype(int).values

    # Pick the "post-pit" compound from the REAL race (what he actually switched to)
    post_compound = None
    if real_pit_lap is not None:
        # often compound is updated from the lap AFTER the pit
        nxt = drv.loc[drv["LapNumber"] == real_pit_lap + 1]
        if len(nxt) and "Compound" in nxt.columns:
            post_compound = nxt["Compound"].iloc[0]

    # --- Base cumulative time series for driver (REAL)
    base_driver_cum = base_cum[driver_code].copy()

    first_sim_lap = int(min(laps_to_predict))

    anchor_lap = first_sim_lap - 1
    anchor = 0.0 if anchor_lap <= 0 else float(base_driver_cum.loc[anchor_lap])

    # --- Lap-time series for traffic comparisons (start from REAL, overwrite simulated laps)
    #what_laptime = drv.set_index("LapNumber")["LapTime_s"].astype(float).copy()

    if real_pit_lap is not None and verbose:
        print("rp/wp:", real_pit_lap, pit_lap)
        for L in range(real_pit_lap-1, min(pit_lap+2, real_pit_lap+6)):
            real_lt = float(real_laptime.loc[int(L)])
            wl_lt = float(what_laptime.loc[L]) if L in what_laptime.index else None
            is_pit = None
            if "is_pit_lap" in drv.columns:
                tmp = drv.loc[drv["LapNumber_i"] == int(L), "is_pit_lap"]
                is_pit = int(tmp.iloc[0]) if len(tmp) else 0
            print(L, "real", round(real_lt,3), "what", round(wl_lt,3), "is_pit", is_pit)
    
    # ------------------------------------------------------------
    # Predict: model outputs LapDelta, convert to LapTime using lap_avg_3
    # ------------------------------------------------------------
    is_overcut = (rp is not None) and (wp > rp)
    is_undercut = (rp is not None) and (wp < rp)

    if "lap_avg_3" not in sim_rows.columns:
        raise ValueError("lap_avg_3 missing from sim_rows. Ensure add_features() ran and lap_avg_3 is in df_fe.")

    from collections import deque

    first_sim_lap = int(min(laps_to_predict))

    seed_df = drv[
        (drv["LapNumber"].astype(int) >= first_sim_lap - 6) &
        (drv["LapNumber"].astype(int) <= first_sim_lap - 1)
    ].copy()

    if "TrackStatus" in seed_df.columns:
        seed_df = seed_df[seed_df["TrackStatus"].astype(int) == 1]

    # basic filtering to avoid SC/pit/outliers poisoning the seed
    seed_df["LapTime_s"] = pd.to_numeric(seed_df["LapTime_s"], errors="coerce")
    seed_df = seed_df.dropna(subset=["LapTime_s"])
    seed_df = seed_df[(seed_df["LapTime_s"] > 60) & (seed_df["LapTime_s"] < 130)]
    if "is_pit_lap" in seed_df.columns:
        seed_df = seed_df[seed_df["is_pit_lap"].astype(int) == 0]
    # If you have outlap/inlap flags, drop them too (they distort ref pace)
    for col in ["IsOutLap", "IsInLap"]:
        if col in seed_df.columns:
            seed_df = seed_df[seed_df[col].astype(int) == 0]

    # Optional: tighten "normal lap" band for ref pace seeding
    seed_df = seed_df[(seed_df["LapTime_s"] > 75) & (seed_df["LapTime_s"] < 115)]

    seed = seed_df.tail(3)["LapTime_s"].astype(float).tolist()
    if len(seed) < 3:
        fallback = drv.loc[drv["LapNumber"] == first_sim_lap-1, "LapTime_s"]
        if len(fallback):
            seed = [float(fallback.iloc[0])] * 3
    q = deque(seed, maxlen=3)

    # --- estimate typical POST-pit pace from the REAL race (clean laps after the real pit) ---
    post_pit_seed = None
    if real_pit_lap is not None:
        post_df = drv[(drv["LapNumber"] >= real_pit_lap + 2) & (drv["LapNumber"] <= real_pit_lap + 6)].copy()
        post_df["LapTime_s"] = pd.to_numeric(post_df["LapTime_s"], errors="coerce")
        post_df = post_df.dropna(subset=["LapTime_s"])
        post_df = post_df[(post_df["LapTime_s"] > 75) & (post_df["LapTime_s"] < 115)]
        if "is_pit_lap" in post_df.columns:
            post_df = post_df[post_df["is_pit_lap"].astype(int) == 0]
        for col in ["IsOutLap", "IsInLap"]:
            if col in post_df.columns:
                post_df = post_df[post_df[col].astype(int) == 0]
        if len(post_df):
            post_pit_seed = float(post_df["LapTime_s"].median())

    # --- pre-pit state at lap BEFORE what-if pit (used for stint/stops transition) ---
    pre_state = {"Stint": 1.0, "StopsSoFar": 0.0, "Compound": None}

    pre_row = drv.loc[drv["LapNumber"].astype(int) == int(pit_lap - 1)]
    if len(pre_row):
        pre_row = pre_row.iloc[0]
        pre_state["Stint"] = float(pre_row.get("Stint", 1.0))
        pre_state["StopsSoFar"] = float(pre_row.get("StopsSoFar", 0.0))
        if "Compound" in pre_row.index:
            pre_state["Compound"] = pre_row.get("Compound", None)

    real_event_laps = []
    real = np.nan
    real_loss = 0.0
    if is_overcut:
            what_laptime, real_event_laps, repl = neutralise_pit_event(
                what_laptime, drv, rp, slow_thresh=8.0, max_window=2
            )
            real_loss = sum(float(real_laptime.loc[int(L)]) for L in real_event_laps) - float(repl) * len(real_event_laps)
            real_loss = max(0.0, float(real_loss))
            real_loss = min(real_loss, pit_loss_s)

    pred_laptime = []
    push_boost_used = {}
    for i, lap in enumerate(sim_laps):
        rows = sim_rows.loc[sim_rows["LapNumber_i"] == int(lap)]
        if rows.empty:
            pred_laptime.append(float(what_laptime.loc[int(lap)]))
            continue
        row = rows.iloc[0]
        row_feat = row[pace_features].copy()

        # Build SCENARIO features for the model (don't mutate sim_rows)
        row_feat_scn = row_feat.copy()


        ref = float(np.mean(q)) if len(q) else float(row.get("lap_avg_3", row.get("LapTime_s", 0.0)))

        if not np.isfinite(ref):
            # fallback if lap_avg_3 missing/NaN
            ref = float(np.mean(q)) if len(q) else float(row.get("LapTime_s", 0.0))

        # ---- apply WHAT-IF pit state to features for ALL laps >= pit_lap ----
        if lap >= pit_lap:
            k_age = int(lap - pit_lap)  # 0 on pit lap, 1 next lap, ...

            # tyre age resets relative to what-if pit
            for col in ["TyreLife", "LapsSinceLastPit"]:
                if col in row_feat_scn.index:
                    row_feat_scn[col] = 1.0 + k_age

            # compound should switch to what the driver actually went onto in the real race
            # (otherwise you get "fresh Medium" when reality is "fresh Hard" etc.)
            if post_compound is not None and "Compound" in row_feat_scn.index:
                row_feat_scn["Compound"] = post_compound

            # stint/stops increment once at the what-if pit (constant afterwards)
            if "Stint" in row_feat_scn.index:
                row_feat_scn["Stint"] = pre_state["Stint"] + 1.0
            if "StopsSoFar" in row_feat_scn.index:
                row_feat_scn["StopsSoFar"] = pre_state["StopsSoFar"] + 1.0

            # IMPORTANT: don't let the model think this lap is a real pit/outlap if you already add pit_loss_s manually
            for col in ["is_pit_lap","IsPitLap","IsOutLap","IsInLap"]:
                if col in row_feat_scn.index:
                    row_feat_scn[col] = 0

            # If your model uses TyreWearFrac, reset it consistently too
            # (Only do this if TyreWearFrac is actually in pace_features)
            if "TyreWearFrac" in row_feat_scn.index:
                # simplest generic reset (safe): proportional to age, capped
                # if you already have a wear_frac(compound, age) helper, use that instead
                row_feat_scn["TyreWearFrac"] = min(1.0, (1.0 + k_age) / 35.0)

        real_lt = real_laptime

        # ---- Keep your "neutralise REAL pit lap" block, but apply it to row_feat_scn ----
        if is_undercut and lap == real_pit_lap:
            for col in ["is_pit_lap","IsPitLap","IsOutLap","IsInLap"]:
                if col in row_feat_scn.index:
                    row_feat_scn[col] = 0

        if use_real_post_pit:
            # IMPORTANT: use the (possibly neutralised) what_laptime, not raw drv
            lt = float(what_laptime.loc[int(lap)])
            scenario_loss_lap = wp + pitloss_offset  # your offset logic

            if lap == scenario_loss_lap and (rp is None or wp != rp):
                lt += real_loss


            pred_laptime.append(float(lt))
            continue


        # Now predict using scenario-consistent features
        #d = float(pace_model.predict(row_feat_scn.to_frame().T)[0])
        d = float(pace_model.predict(row_feat_scn.to_frame().T)[0])

        # hard safety bound on LapDelta (tune these numbers)
        d = float(np.clip(d, -2.5, 4.0))

        lt = ref + d


        # Apply pit loss only on the WHAT-IF pit lap
        if lap == wp_loss_lap:
            lt += pit_loss_s
            # After the pit lap, reset the rolling reference pace to post-pit pace
            if post_pit_seed is not None:
                q = deque([post_pit_seed, post_pit_seed, post_pit_seed], maxlen=3)

        # Push boost (your existing logic)
        push_applied = 0.0
        k = int(lap - pit_lap)
        if post_pit_push_laps and (1 <= k <= post_pit_push_laps):
            if push_decay == "linear":
                denom = max(1, post_pit_push_laps - 1)
                factor = max(0.0, 1.0 - (k - 1) / denom)
            elif push_decay == "exp":
                factor = float(np.exp(-push_exp_k * (k - 1)))
            else:
                factor = 1.0
            push_applied = post_pit_push_boost_s * factor
            lt -= push_applied

        real_row = drv.loc[drv["LapNumber"].astype(int) == int(lap)]
        real_age = float(real_row["LapsSinceLastPit"].iloc[0]) if len(real_row) else np.nan
        
        if lap < pit_lap:
            scn_age = real_age  # before what-if pit, tyre age same as real
        else:
            scn_age = 1.0 + float(lap - pit_lap)  # resets at what-if pit

        # --- Controlled tyre-age delta adjustment (bounded) ---
        # If we pit earlier than reality, scn_age is LOWER => we should be FASTER.
        # If we pit later than reality (overcut), scn_age can be HIGHER => should be SLOWER.



        if np.isfinite(real_age) and np.isfinite(scn_age):
            age_diff = real_age - scn_age  # + means scenario has fresher tyres

            # sensitivity: seconds per lap per lap-of-age difference
            # start conservative; tune later
            s_per_lap_age = 0.07  # try 0.05–0.10
            tyre_adj = 0.0
            #tyre_adj = np.clip(age_diff * s_per_lap_age, -1.5, +1.5)
            #lt -= tyre_adj
        else:
            tyre_adj = 0.0

        # debug stores
        push_boost_used[int(lap)] = float(push_applied)
        ref_used[int(lap)] = float(ref)
        delta_used[int(lap)] = float(d)
        pred_lt_used[int(lap)] = float(lt)
        tyre_adj_used[int(lap)] = float(tyre_adj)
        pred_laptime.append(float(lt))

        if lap < pit_lap:
            # Update q with the "pace context" for next lap
            real_row = drv.loc[drv["LapNumber"].astype(int) == int(lap)]
            if len(real_row):
                r = real_row.iloc[0]
                is_bad = (
                    int(r.get("is_pit_lap", 0)) == 1 or
                    int(r.get("IsOutLap", 0)) == 1 or
                    int(r.get("IsInLap", 0)) == 1
                )
            else:
                is_bad = True

            if lap < pit_lap:
                # before what-if pit: use real pace context
                if (not is_bad) and len(real_row):
                    q.append(float(r["LapTime_s"]))
            else:
                # after what-if pit: use predicted pace context (bounded)
                q.append(float(np.clip(lt, 70.0, 130.0)))

    pred_laptime = np.array(pred_laptime, dtype=float)
    if len(pred_laptime) != len(sim_laps):
        raise ValueError(f"Length mismatch: sim_laps={len(sim_laps)} pred_laptime={len(pred_laptime)}")

    if verbose:
        print("len(sim_laps):", len(sim_laps))
        print("len(pred_laptime):", len(pred_laptime))
        print("first/last sim_laps:", sim_laps[:3], sim_laps[-3:])

    # overwrite driver lap times for simulated laps (for traffic comparisons)
    what_laptime.loc[sim_laps] = pred_laptime

    cum_start = int(min(sim_laps))

    if is_overcut:
        # start before the real pit-loss lap so neutralisation affects cumtime
        # (use rp_loss_lap if you have it; else rp)
        start_event = int(rp_loss_lap if rp_loss_lap is not None else rp)
        cum_start = max(1, start_event - 1)

    anchor_lap=cum_start - 1
    #first_sim = int(min(sim_laps))
    anchor = 0.0 if anchor_lap <= 1 else float(base_driver_cum.loc[anchor_lap])

    cum = {}
    t = anchor
    for lap in range(int(cum_start), int(end_lap) + 1):
        lt = what_laptime.get(int(lap), np.nan)
        if not np.isfinite(lt):
            continue
        t += float(lt)  # REAL unless we overwrote it
        cum[lap] = t

    what_driver_cum = base_driver_cum.copy()
    what_driver_cum.loc[cum_start:end_lap] = pd.Series(cum)

    # Lock delta once order stabilises
    #delta_at_lock = float(
     #   what_driver_cum.loc[delta_lock_lap] - base_driver_cum.loc[delta_lock_lap]
    #)

    #what_driver_cum.loc[delta_lock_lap + 1 :] = (
     #   base_driver_cum.loc[delta_lock_lap + 1 :] + delta_at_lock
    #)

    what_cum = base_cum.copy()
    what_cum[driver_code] = what_driver_cum

    # sanity
    tmp = what_cum[driver_code].dropna()
    if verbose:
        print("Monotonic cumtime?", tmp.is_monotonic_increasing)
        print("Pred laptime median/min/max:", float(np.median(pred_laptime)), float(np.min(pred_laptime)), float(np.max(pred_laptime)))

    score_lap = int(end_lap)
    #final_lap = int(race["LapNumber"].max())

    base_finish = base_cum.loc[score_lap].dropna().sort_values()
    what_finish = what_cum.loc[score_lap].dropna().sort_values()

    if driver_code not in base_finish.index or driver_code not in what_finish.index:
        raise ValueError(
            f"{driver_code} missing at score lap={score_lap}"
            f"(pit_lap={pit_lap}, end_lap={end_lap})."
            f"Present in base? {driver_code in base_finish.index}, "
            f"Present in what-if? {driver_code in what_finish.index}" 
        )

    base_pos_finish = int(base_finish.index.get_loc(driver_code) + 1)
    what_pos_finish = int(what_finish.index.get_loc(driver_code) + 1)

    net_delta_finish = float(
        what_finish.loc[driver_code] - base_finish.loc[driver_code]
    )
    
    # ------------------------------------------------------------
    # Traffic penalty loop (apply only in reported window)
    # ------------------------------------------------------------
    penalties = {}

    for lap in range(pit_lap, end_lap + 1):
        times = what_cum.loc[lap].dropna().sort_values()
        if driver_code not in times.index:
            continue

        pos = times.index.get_loc(driver_code)  # 0-based
        pos_used[lap] = int(pos + 1)

        ahead = None
        gap = np.nan
        pace_adv = np.nan
        penalty = 0.0

        if pos > 0:
            ahead = times.index[pos - 1]
            gap = float(times.loc[driver_code] - times.loc[ahead])

            my_lt = float(what_laptime.get(lap, np.nan))
            ahead_row = race[(race["Driver"] == ahead) & (race["LapNumber"] == lap)]
            ahead_lt = float(ahead_row["LapTime_s"].iloc[0]) if len(ahead_row) else np.nan

            if np.isfinite(my_lt) and np.isfinite(ahead_lt):
                pace_adv = ahead_lt - my_lt  # + means I'm faster

            if np.isfinite(gap) and gap <= traffic_gap_s and np.isfinite(pace_adv):
                if pace_adv < pass_advantage_s:
                    penalty = traffic_loss_s * min(1.0, gap / traffic_gap_s)

        ahead_used[lap] = ahead
        gap_used[lap] = gap
        pace_adv_used[lap] = pace_adv
        penalty_used[lap] = penalty
        penalties[lap] = penalty

        if penalty > 0:
            what_cum.loc[lap:, driver_code] = what_cum.loc[lap:, driver_code] + penalty
            # keep lap-time series consistent with penalty
            what_laptime.loc[lap] = float(what_laptime.loc[lap]) + float(penalty)



    # ------------------------------------------------------------
    # Build per-lap output
    # ------------------------------------------------------------
    out_rows = []
    for lap in range(pit_lap, end_lap + 1):
        base_times = base_cum.loc[lap].dropna().sort_values()
        what_times = what_cum.loc[lap].dropna().sort_values()

        if driver_code not in base_times.index or driver_code not in what_times.index:
            continue

        base_pos = int(base_times.index.get_loc(driver_code) + 1)
        what_pos = int(what_times.index.get_loc(driver_code) + 1)

        base_t = float(base_times.loc[driver_code])
        what_t = float(what_times.loc[driver_code])

        idx = what_times.index.get_loc(driver_code)

        ahead_driver = what_times.index[idx - 1] if idx > 0 else None
        gap_ahead = float(what_times.loc[driver_code] - what_times.loc[ahead_driver]) if ahead_driver else np.nan

        behind_driver = what_times.index[idx + 1] if idx < (len(what_times) - 1) else None
        gap_behind = float(what_times.loc[behind_driver] - what_times.loc[driver_code]) if behind_driver else np.nan

        
        out_rows.append({
            "LapNumber": lap,
            "BasePos": base_pos,
            "WhatIfPos": what_pos,
            "BaseCumTime_s": base_t,
            "WhatIfCumTime_s": what_t,
            "DeltaCum_s": what_t - base_t,

            "TrafficPenalty_s": float(penalties.get(lap, 0.0)),
            "AheadDriver": ahead_driver,
            "GapAhead_s": gap_ahead,
            "BehindDriver": behind_driver,
            "GapBehind_s": gap_behind,

            "Pos_loop": pos_used.get(lap, np.nan),
            "AheadDriver_loop": ahead_used.get(lap, None),
            "GapAhead_loop_s": gap_used.get(lap, np.nan),
            "PaceAdv_loop_s": pace_adv_used.get(lap, np.nan),
            "TrafficPenalty_loop_s": penalty_used.get(lap, 0.0),

            "PushBoostApplied_s": float(push_boost_used.get(lap, 0.0)),#"RefPace_used": float(ref_used.get(lap, np.nan)),
            "PredDelta_used": float(delta_used.get(lap, np.nan)),
            "PredLapTime_used": float(pred_lt_used.get(lap, np.nan)),
            "TyreLife_used": float(tyrelife_used.get(lap, np.nan)),
            "LapsSinceLastPit_used": float(lsp_used.get(lap, np.nan)),
            "RefPace_used": float(ref_used.get(lap, np.nan)),
            "TyreAdj_used": float(tyre_adj_used.get(lap, 0.0)),


        })

    out = pd.DataFrame(out_rows)
    out["DeltaCum_s"] = out["DeltaCum_s"].astype(float)
    return out, quality



