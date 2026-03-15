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
    post_pit_push_boost_s: float = 0.8,   # subtract this many seconds (faster)
    push_decay: str = "linear",   # "linear" or "exp"
    push_exp_k: float = 0.7,      # only used if push_decay="exp"

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


    if pace_features is None:
        raise ValueError("Pass pace_features (exact columns pace_model expects).")

    race = df[df["RaceId"] == race_id].copy()
    if race.empty:
        raise ValueError(f"No data for race {race_id}")

    max_lap = int(race["LapNumber"].max())
    end_lap = min(pit_lap + horizon_laps, max_lap)

    # --- Baseline cumulative times for all drivers (REAL)
    base_cum = _build_cumtime_table(race)
    if driver_code not in base_cum.columns:
        raise ValueError(f"{driver_code} not found in race {race_id}")

    drv = race[race["Driver"] == driver_code].sort_values("LapNumber").copy()
    if drv.empty:
        raise ValueError(f"No data for driver {driver_code} in race {race_id}")

    # --- Detect real pit lap (first pit) if available
    real_pit_lap = None
    if cancel_real_pit and ("is_pit_lap" in drv.columns):
        real_pits = drv.loc[drv["is_pit_lap"].astype(int) == 1, "LapNumber"].astype(int).tolist()
        real_pit_lap = min(real_pits) if real_pits else None

    # --- Decide simulation start
    sim_start = real_pit_lap if (real_pit_lap is not None and real_pit_lap < pit_lap) else pit_lap

    sim_rows = drv[(drv["LapNumber"] >= sim_start) & (drv["LapNumber"] <= end_lap)].copy()
    if sim_rows.empty:
        raise ValueError(f"No laps found for {driver_code} in {race_id} within [{sim_start},{end_lap}]")

    sim_laps = sim_rows["LapNumber"].astype(int).values

    # Pick the "post-pit" compound from the REAL race (what he actually switched to)
    post_compound = None
    if real_pit_lap is not None:
        # often compound is updated from the lap AFTER the pit
        nxt = drv.loc[drv["LapNumber"] == real_pit_lap + 1]
        if len(nxt) and "Compound" in nxt.columns:
            post_compound = nxt["Compound"].iloc[0]

    # --- pre-pit state at lap BEFORE what-if pit (used for what-if pit feature transition)
    pre_state = {}
    pre_row = drv.loc[drv["LapNumber"] == (pit_lap - 1)]
    if len(pre_row):
        pre_row = pre_row.iloc[0]
        pre_state["Stint"] = float(pre_row.get("Stint", 1.0))
        pre_state["StopsSoFar"] = float(pre_row.get("StopsSoFar", 0.0))
        pre_state["Compound"] = pre_row.get("Compound", None)
    else:
        pre_state["Stint"] = 1.0
        pre_state["StopsSoFar"] = 0.0
        pre_state["Compound"] = None

    # --- Base cumulative time series for driver (REAL)
    base_driver_cum = base_cum[driver_code].copy()

    # Anchor at end of lap sim_start-1
    anchor_lap = sim_start - 1
    anchor = 0.0 if anchor_lap <= 0 else float(base_driver_cum.loc[anchor_lap])

    # --- Lap-time series for traffic comparisons (start from REAL, overwrite simulated laps)
    what_laptime = drv.set_index("LapNumber")["LapTime_s"].astype(float).copy()

    # ------------------------------------------------------------
    # OVERCUT handling: force "stayed out" state on laps real_pit..(pit_lap-1)
    # ------------------------------------------------------------
    if (real_pit_lap is not None) and (real_pit_lap < pit_lap):
        prev = drv.loc[drv["LapNumber"] == (real_pit_lap - 1)]
        if len(prev):
            prev = prev.iloc[0]
            base_tyre_life = float(prev.get("TyreLife", prev.get("LapsSinceLastPit", 1.0)))
            base_lsp = float(prev.get("LapsSinceLastPit", base_tyre_life))
            base_stint = float(prev.get("Stint", 1.0))
            base_stops = float(prev.get("StopsSoFar", 0.0))
            base_comp = prev.get("Compound", None)
        else:
            base_tyre_life, base_lsp, base_stint, base_stops, base_comp = 1.0, 1.0, 1.0, 0.0, None

        mask_stayout = (sim_rows["LapNumber"] >= real_pit_lap) & (sim_rows["LapNumber"] < pit_lap)
        for idx, r in sim_rows.loc[mask_stayout].iterrows():
            lap = int(r["LapNumber"])
            inc = lap - (real_pit_lap - 1)  # +1 on real_pit_lap

            if "TyreLife" in sim_rows.columns:
                sim_rows.at[idx, "TyreLife"] = base_tyre_life + inc
            if "LapsSinceLastPit" in sim_rows.columns:
                sim_rows.at[idx, "LapsSinceLastPit"] = base_lsp + inc
            if "Stint" in sim_rows.columns:
                sim_rows.at[idx, "Stint"] = base_stint
            if "StopsSoFar" in sim_rows.columns:
                sim_rows.at[idx, "StopsSoFar"] = base_stops
            if base_comp is not None and "Compound" in sim_rows.columns:
                sim_rows.at[idx, "Compound"] = base_comp

            # cancel pit flags if present
            if "is_pit_lap" in sim_rows.columns:
                sim_rows.at[idx, "is_pit_lap"] = 0
            if "IsPitLap" in sim_rows.columns:
                sim_rows.at[idx, "IsPitLap"] = 0
            if "IsOutLap" in sim_rows.columns:
                sim_rows.at[idx, "IsOutLap"] = 0

    # ------------------------------------------------------------
    # UNDERCUT handling: cancel the REAL pit lap if we pit earlier
    # ------------------------------------------------------------
    if (real_pit_lap is not None) and (pit_lap < real_pit_lap):
        # only matters if the real pit lap is inside our sim_rows window
        mask_realpit = sim_rows["LapNumber"].astype(int) == int(real_pit_lap)
        if mask_realpit.any():
            idx_rp = sim_rows.index[mask_realpit][0]

            # Cancel pit/outlap/inlap flags so model doesn't treat it as a pit lap
            for col in ["is_pit_lap", "IsPitLap", "IsOutLap", "IsInLap"]:
                if col in sim_rows.columns:
                    sim_rows.at[idx_rp, col] = 0

            # (optional but usually correct) keep stint/compound as "post what-if pit" already,
            # i.e. don't increment again here.


    # ------------------------------------------------------------
    # Predict: model outputs LapDelta, convert to LapTime using lap_avg_3
    # ------------------------------------------------------------
    if "lap_avg_3" not in sim_rows.columns:
        raise ValueError("lap_avg_3 missing from sim_rows. Ensure add_features() ran and lap_avg_3 is in df_fe.")

    from collections import deque

    # seed rolling pace from last 3 NORMAL laps before sim_start
    seed_df = drv[(drv["LapNumber"] >= sim_start-6) & (drv["LapNumber"] <= sim_start-1)].copy()

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
    q = deque(seed, maxlen=3)

    pred_laptime = []
    push_boost_used = {}
    for i, lap in enumerate(sim_laps):
        row = sim_rows.iloc[i]
        row_feat = row[pace_features].copy()
        # ------------------------------------------------------------
        # UNDERCUT FIX (prediction-only)
        # If we pit EARLIER than reality, neutralise REAL pit features
        # on the real pit lap so the model doesn't "double count" a pit
        # ------------------------------------------------------------
        if (
            real_pit_lap is not None
            and pit_lap < real_pit_lap
            and lap == real_pit_lap
        ):
            # 1) Disable pit / outlap flags for prediction
            for col in ["is_pit_lap", "IsPitLap", "IsOutLap", "IsInLap"]:
                if col in row_feat.index:
                    row_feat[col] = 0

            # 2) Carry tyre age forward as if we DID NOT pit here
            if i > 0:
                prev = sim_rows.iloc[i - 1]

                for col in ["TyreLife", "LapsSinceLastPit", "TyreWearFrac"]:
                    if col in row_feat.index and col in prev.index:
                        row_feat[col] = prev[col]

                for col in ["Stint", "StopsSoFar"]:
                    if col in row_feat.index and col in prev.index:
                        row_feat[col] = prev[col]


        # reference pace = rolling mean of recent *normal* laps
        ref = float(np.mean(q)) if len(q) else float(row.get("lap_avg_3", 0.0))
        # ---- Guardrail: keep ref pace anchored to lap_avg_3 to prevent runaway speed ----
        lap_avg = float(row.get("lap_avg_3", ref))

        d = float(pace_model.predict(row_feat.to_frame().T)[0])
        lt = ref + d
        lap_avg = float(row.get("lap_avg_3", lt))

        d = float(pace_model.predict(row_feat.to_frame().T)[0])

        # Base model lap time (no pit loss, no push)
        lt_base = ref + d

        # Scenario lap time (this is what you simulate/output)
        lt = lt_base
        # apply pit loss only on the what-if pit lap
        if lap == pit_lap:
            # reset tyre state for simulation AFTER pit
            if "TyreLife" in sim_rows.columns:
                sim_rows.at[sim_rows.index[i], "TyreLife"] = 1.0
            if "LapsSinceLastPit" in sim_rows.columns:
                sim_rows.at[sim_rows.index[i], "LapsSinceLastPit"] = 1.0
            if "Stint" in sim_rows.columns:
                sim_rows.at[sim_rows.index[i], "Stint"] = sim_rows["Stint"].max() + 1
            lt += pit_loss_s
        
        #decaying push out of pit
        push_applied = 0.0
        k = int(lap - pit_lap)  # outlap is k=0, next lap is k=1

        # apply boost on laps AFTER the pit lap (usually outlap+1 etc.)
        if post_pit_push_laps and (1 <= k <= post_pit_push_laps):
            if push_decay == "linear":
                # 3 laps -> factors: 1.0, 0.5, 0.0 (if you want last lap to still have some, tweak)
                denom = max(1, post_pit_push_laps - 1)
                factor = max(0.0, 1.0 - (k - 1) / denom)
            elif push_decay == "exp":
                factor = float(np.exp(-push_exp_k * (k - 1)))
            else:
                factor = 1.0  # fallback = constant boost

            push_applied = post_pit_push_boost_s * factor
            lt -= push_applied

        push_boost_used[int(lap)] = float(push_applied)
        # store prediction diagnostics
        ref_used[int(lap)] = ref
        delta_used[int(lap)] = d
        pred_lt_used[int(lap)] = lt
        # -----------------------------
        # OUTPUT-ONLY simulated tyre age
        # (does NOT affect predictions)
        # -----------------------------
        if lap < pit_lap:
            # before the WHAT-IF pit, use whatever tyre age your sim_rows currently implies
            # (this includes your overcut "stayed out" edits already done earlier)
            sim_tl = float(row.get("TyreLife", np.nan))
            sim_lsp = float(row.get("LapsSinceLastPit", np.nan))
        else:
            # after the WHAT-IF pit, tyre age resets relative to pit_lap
            k_age = int(lap - pit_lap)  # 0 on pit lap, 1 next lap, ...
            sim_tl = 1.0 + k_age
            sim_lsp = 1.0 + k_age

        tyrelife_used[int(lap)] = sim_tl
        lsp_used[int(lap)] = sim_lsp
        pred_laptime.append(lt)
        # IMPORTANT: don't poison the rolling ref with pit-loss lap (and optionally outlap)
        if lap != pit_lap:
            q.append(lt)

    pred_laptime = np.array(pred_laptime, dtype=float)



    # overwrite driver lap times for simulated laps (for traffic comparisons)
    what_laptime.loc[sim_laps] = pred_laptime

    # anchored cumulative overwrite
    window_cum = anchor + np.cumsum(pred_laptime)
    what_driver_cum = base_driver_cum.copy()
    what_driver_cum.loc[sim_laps] = window_cum

    # For laps after end_lap, keep baseline shape but shifted by delta at end_lap
    if end_lap < what_driver_cum.index.max():
        shift = float(what_driver_cum.loc[end_lap] - base_driver_cum.loc[end_lap])
        what_driver_cum.loc[end_lap + 1 :] = base_driver_cum.loc[end_lap + 1 :] + shift

    what_cum = base_cum.copy()
    what_cum[driver_code] = what_driver_cum

    # sanity
    tmp = what_cum[driver_code].dropna()
    print("Monotonic cumtime?", tmp.is_monotonic_increasing)
    print("Pred laptime median/min/max:", float(np.median(pred_laptime)), float(np.min(pred_laptime)), float(np.max(pred_laptime)))

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

        })

    out = pd.DataFrame(out_rows)
    out["DeltaCum_s"] = out["DeltaCum_s"].astype(float)
    return out