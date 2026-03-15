import pandas as pd

def add_labels(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """
    Add strategy labels:
      - pit_next_lap: 1 if is_pit_lap on next lap
      - pit_within_<window>: 1 if pit occurs in next `window` laps
    """
    df = df.sort_values(["RaceId", "Driver", "LapNumber"]).copy()

    # Ensure we have is_pit_lap (1/0)
    if "is_pit_lap" not in df.columns:
        raise ValueError("Expected 'is_pit_lap' column in df")

    # Next-lap label
    df["pit_next_lap"] = (
        df.groupby(["RaceId", "Driver"])["is_pit_lap"].shift(-1)
        .fillna(0)
        .astype(int)
    )

    # Pit within N laps
    def pit_within_window(x, w: int):
        out = []
        x = list(x)
        for i in range(len(x)):
            future = x[i + 1 : i + 1 + w]
            out.append(int(sum(future) > 0))
        return out

    df[f"pit_within_{window}"] = (
        df.groupby(["RaceId", "Driver"])["is_pit_lap"]
        .transform(lambda s: pit_within_window(s, window))
        .astype(int)
    )

    return df
