from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from sklearn.inspection import permutation_importance

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def main() -> None:
    df_path = PROJECT_ROOT / "ML" / "data" / "df_fe.parquet"
    model_path = PROJECT_ROOT / "ML" / "models" / "pace_model.joblib"
    out_path = PROJECT_ROOT / "ML" / "models" / "feature_rank_pace.csv"

    df = pd.read_parquet(df_path)
    model = joblib.load(model_path)

    from ML.src.config import FEATURES_PACE

    target = "LapDelta"
    df = df.dropna(subset=[target]).copy()

    df_s = df.sample(n=min(8000, len(df)), random_state=42)

    X = df_s[FEATURES_PACE]
    y = df_s[target]

    r = permutation_importance(
        model, X, y,
        n_repeats=5,
        random_state=42,
        scoring="neg_mean_absolute_error"
    )

    imp = pd.DataFrame({
        "feature": FEATURES_PACE,
        "importance_mean": r.importances_mean,
        "importance_std": r.importances_std
    }).sort_values("importance_mean", ascending=False)

    imp.to_csv(out_path, index=False)
    print("Saved:", out_path)
    print(imp.head(10))

if __name__ == "__main__":
    main()
