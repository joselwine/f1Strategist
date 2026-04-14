from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import joblib
import pandas as pd

@dataclass
class Assets:
    df_fe: pd.DataFrame
    pace_model: object
    features_pace: list[str]
    feature_rank: pd.DataFrame | None = None  # optional

def load_assets(root: Path) -> Assets:
    data_path = root / "ML" / "data" / "df_fe.parquet"
    model_path = root / "ML" / "models" / "pace_model.joblib"
    feats_path = root / "ML" / "models" / "features_pace.txt"

    df_fe = pd.read_parquet(data_path)
    pace_model = joblib.load(model_path)
    features_pace = (feats_path.read_text().splitlines())

    return Assets(df_fe=df_fe, pace_model=pace_model, features_pace=features_pace)
