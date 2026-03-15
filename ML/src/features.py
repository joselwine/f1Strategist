import pandas as pd
import numpy as np


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add strategy-relevant features:
      - tyre age / wear
      - stint progression
      - race phase
      - contextual pit activity
      - position changes
      - SC/VSC flags
    """
    df_fe = df.copy()
    df_fe = df_fe.sort_values(["RaceId", "Driver", "LapNumber"])

    # Tyre age (already have TyreLife, but make a cleaner alias)
    df_fe["TyreAge"] = df_fe["TyreLife"].fillna(df_fe["LapsSinceLastPit"])

    # Stint position
    g_stint = df_fe.groupby(["RaceId", "Driver", "Stint"])
    df_fe["StintLap"] = g_stint.cumcount() + 1

    stint_min = g_stint["LapNumber"].transform("min")
    stint_max = g_stint["LapNumber"].transform("max")
    df_fe["StintLength"] = stint_max - stint_min + 1
    df_fe["StintFrac"] = df_fe["StintLap"] / df_fe["StintLength"]

    # Tyre wear fraction
    TYPICAL_LIFE = {"SOFT": 15, "MEDIUM": 25, "HARD": 35}
    df_fe["TyreLifeExpected"] = df_fe["Compound"].map(TYPICAL_LIFE)
    df_fe["TyreWearFrac"] = (df_fe["TyreAge"] / df_fe["TyreLifeExpected"]).clip(0, 1)


    # Race phase
    race_max_lap = df_fe.groupby("RaceId")["LapNumber"].transform("max")
    df_fe["LapsRemaining"] = race_max_lap - df_fe["LapNumber"] + 1
    df_fe["RaceFracRemaining"] = df_fe["LapsRemaining"] / race_max_lap

    # Simple lagged + rolling style degradation (example: rolling laps-since-pit)
    g_drv = df_fe.groupby(["RaceId", "Driver"])
    # Rolling average of lap time (seconds) for pace baseline
    df_fe["lap_avg_3"] = (
        g_drv["LapTime_s"]
        .rolling(3, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )


    df_fe["IsOldTyre"] = (df_fe["TyreAge"] > 12).astype(int)

    # Context: how many cars pitted recently
    pits_per_lap = (
        df_fe.groupby(["RaceId", "LapNumber"])["is_pit_lap"]
        .sum()
        .rename("CarsPittedThisLap")
    )
    df_fe = df_fe.merge(pits_per_lap, on=["RaceId", "LapNumber"], how="left")
    df_fe["CarsPittedThisLap"] = df_fe["CarsPittedThisLap"].fillna(0)

    df_fe["CarsPittedPrevLap"] = (
        df_fe.groupby("RaceId")["CarsPittedThisLap"].shift(1).fillna(0)
    )

    df_fe["CarsPittedLast3"] = (
        df_fe.groupby("RaceId")["CarsPittedThisLap"]
        .rolling(3)
        .sum()
        .reset_index(level=0, drop=True)
        .fillna(0)
    )

    # Position change
    df_fe["Pos_prev"] = g_drv["Position"].shift(1)
    df_fe["PosChange1"] = (df_fe["Pos_prev"] - df_fe["Position"]).fillna(0)
    df_fe["PosChange3"] = g_drv["Position"].diff(3).fillna(0)

    # SC / VSC flags (very rough)
    df_fe["TrackStatus"] = df_fe["TrackStatus"].astype(str)
    df_fe["IsSC"] = df_fe["TrackStatus"].str.contains("4|5").astype(int)
    df_fe["IsVSC"] = df_fe["TrackStatus"].str.contains("6").astype(int)

    return df_fe
