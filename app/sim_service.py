# app/sim_service.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd


@dataclass
class SimContext:
    """
    Holds heavy objects in memory so you don't reload them every request.
    """
    df_fe: pd.DataFrame
    pace_model: Any
    pace_features: list[str]


def _ensure_int(x, name: str) -> int:
    try:
        return int(x)
    except Exception as e:
        raise ValueError(f"{name} must be an int, got {x!r}") from e


def run_strategy_sim(
    ctx: SimContext,
    *,
    race_id: str,
    driver: str,
    pit_lap: int,
    horizon_laps: int = 20,
    pit_loss_s: float = 22.0,
    traffic_gap_s: float = 1.5,
    traffic_loss_s: float = 0.6,
    pass_advantage_s: float = 0.4,
    cancel_real_pit: bool = True,
    post_pit_push_laps: int = 0,
    post_pit_push_boost_s: float = 0.0,
    use_real_post_pit: bool = True,
) -> dict:
    """
    One clean entry-point for the entire app.
    Returns JSON-safe dict with:
      - summary: key metrics
      - laps: list[dict] per lap output
    """
    # ---- validate ----
    if not isinstance(race_id, str) or not race_id:
        raise ValueError("race_id must be a non-empty string")
    if not isinstance(driver, str) or not driver:
        raise ValueError("driver must be a non-empty string")

    pit_lap = _ensure_int(pit_lap, "pit_lap")
    horizon_laps = _ensure_int(horizon_laps, "horizon_laps")

    if horizon_laps <= 0:
        raise ValueError("horizon_laps must be > 0")
    if pit_loss_s < 0:
        raise ValueError("pit_loss_s must be >= 0")

    # ---- import here to avoid circular imports ----
    # adjust this import path to match your project
    from ML.src.simulator import simulate_pit_rejoin_and_traffic

    out: pd.DataFrame = simulate_pit_rejoin_and_traffic(
        df=ctx.df_fe,
        pace_model=ctx.pace_model,
        race_id=race_id,
        driver_code=driver,
        pit_lap=pit_lap,
        horizon_laps=horizon_laps,
        pit_loss_s=pit_loss_s,
        pace_features=ctx.pace_features,
        traffic_gap_s=traffic_gap_s,
        traffic_loss_s=traffic_loss_s,
        pass_advantage_s=pass_advantage_s,
        cancel_real_pit=cancel_real_pit,
        post_pit_push_laps=post_pit_push_laps,
        post_pit_push_boost_s=post_pit_push_boost_s,
        use_real_post_pit=use_real_post_pit,
    )

    if out.empty:
        raise ValueError("Simulation returned empty output")

    # ---- build summary (keep it stable over time) ----
    last = out.iloc[-1]

    summary = {
        "race_id": race_id,
        "driver": driver,
        "pit_lap": pit_lap,
        "end_lap": int(last["LapNumber"]),
        "net_delta_end_s": float(last["DeltaCum_s"]),
        "base_pos_end": int(last["BasePos"]),
        "whatif_pos_end": int(last["WhatIfPos"]),
    }

    # optional: include a few useful extras if present
    optional_cols = ["TrafficPenalty_s", "GapAhead_s", "AheadDriver", "RejoinPos", "RejoinGapAhead"]
    for c in optional_cols:
        if c in out.columns:
            val = last[c]
            # convert NaNs safely
            summary[c] = None if pd.isna(val) else (float(val) if isinstance(val, (int, float)) else str(val))

    # ---- JSON-safe laps ----
    laps = out.copy()

    # Convert numpy types -> python types
    laps_dicts = []
    for rec in laps.to_dict("records"):
        clean = {}
        for k, v in rec.items():
            if pd.isna(v):
                clean[k] = None
            elif hasattr(v, "item"):  # numpy scalar
                clean[k] = v.item()
            else:
                clean[k] = v
        laps_dicts.append(clean)

    return {"summary": summary, "laps": laps_dicts}
