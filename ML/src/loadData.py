from pathlib import Path
import os

import fastf1
import pandas as pd

from .config import PROCESSED_DIR, TRAIN_RACES, TEST_RACES

CACHE_DIR = Path(__file__).resolve().parents[1] / "fastf1_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))


def make_lap_dataset_for_session(year: int, event: str, session_type: str = "R") -> pd.DataFrame:
 
    print(f"Loading {year} {event} {session_type}...")
    session = fastf1.get_session(year, event, session_type)
    session.load()

    laps = session.laps.copy()
    laps = laps[~laps["Driver"].isna()].copy()

    # Core identifiers
    laps["Year"] = year
    laps["Track"] = event
    laps["RaceId"] = f"{year}-{event}"


    # LapTime in seconds
    laps["LapTime_s"] = laps["LapTime"].dt.total_seconds()

    # Normalised lap number within race
    max_lap = laps["LapNumber"].max()
    laps["LapNumberNorm"] = laps["LapNumber"] / max_lap

    # Simple pit label
    laps["is_pit_lap"] = laps["PitInTime"].notna().astype(int)

    # Laps since last pit + StopsSoFar
    def add_driver_features(df):
        df = df.sort_values("LapNumber").copy()
        laps_since = []
        counter = 0
        for _, row in df.iterrows():
            counter += 1
            laps_since.append(counter)
            if pd.notna(row["PitInTime"]):
                counter = 0
        df["LapsSinceLastPit"] = laps_since
        df["StopsSoFar"] = df["Stint"] - 1
        return df

    laps = laps.groupby("Driver", group_keys=False).apply(add_driver_features)

    # TyreLife fallback
    if "TyreLife" in laps.columns:
        laps["TyreLife"] = laps["TyreLife"].fillna(laps["LapsSinceLastPit"])
    else:
        laps["TyreLife"] = laps["LapsSinceLastPit"]

    cols = [
        "Year", "Track", "RaceId",
        "Driver", "Team",
        "LapNumber", "LapNumberNorm",
        "LapTime_s",
        "Stint", "LapsSinceLastPit", "StopsSoFar",
        "Compound", "TyreLife", "Position", "TrackStatus",
        "is_pit_lap",
    ]
    df = laps[cols].dropna()
    print(f"  -> {len(df)} rows for {year} {event}")
    return df


def build_full_dataset(outfile: Path | None = None) -> pd.DataFrame:
 
    races = TRAIN_RACES + TEST_RACES
    print(f"Building dataset for {len(races)} races...")

    all_dfs = []

    for (year, event) in races:
        try:
            df_race = make_lap_dataset_for_session(year, event, "R")
            all_dfs.append(df_race)
        except Exception as e:
            print(f"Failed to load {year} {event}: {e}")

    full = pd.concat(all_dfs, ignore_index=True)

    if outfile is None:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        outfile = PROCESSED_DIR / "full_laptime_full.csv"

    full.to_csv(outfile, index=False)
    print("Saved combined dataset to:", outfile)

    return full


if __name__ == "__main__":
    build_full_dataset()
