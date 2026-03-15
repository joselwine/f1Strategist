# src/evaluation.py
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline


def explain_driver_race(
    clf: Pipeline,
    df: pd.DataFrame,
    race_id: str,
    driver_code: str,
    features: list,
    categorical: list,
    numeric: list,
    top_n: int = 10,
) -> pd.DataFrame | None:
    """
    Aggregate SHAP importances over all laps for a given driver in a given race.

    Returns a dataframe with feature and mean absolute importance (for class 1).
    """
    # Filter the full feature df to just this driver & race
    df_sub = df[(df["RaceId"] == race_id) & (df["Driver"] == driver_code)].copy()
    if df_sub.empty:
        print(f"No data for {driver_code} in {race_id} in provided dataframe.")
        return None

    # Use the same feature columns the model was trained on
    X_sub = df_sub[features]

    pre = clf.named_steps["preprocess"]
    model = clf.named_steps["model"]

    X_arr = pre.transform(X_sub)
    feature_names = pre.get_feature_names_out(categorical + numeric)

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_arr)[1]  # class 1 = pit-window

    mean_abs = np.abs(shap_vals).mean(axis=0)

    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": mean_abs
    }).sort_values("importance", ascending=False)

    return df_imp.head(top_n)
