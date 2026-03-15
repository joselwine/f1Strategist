# app/test_sim_service.py
import joblib
import pandas as pd

from app.sim_service import SimContext, run_strategy_sim

def main():
    df_fe = pd.read_parquet("path/to/df_fe.parquet")   # or however you load
    pace_model = joblib.load("path/to/pace_model.pkl")
    pace_features = joblib.load("path/to/pace_features.pkl")  # or a python list

    ctx = SimContext(df_fe=df_fe, pace_model=pace_model, pace_features=pace_features)

    res = run_strategy_sim(
        ctx,
        race_id="2024-Abu Dhabi",
        driver="LEC",
        pit_lap=25,
        horizon_laps=20,
        pit_loss_s=22.0,
        use_real_post_pit=True,
    )

    print(res["summary"])
    print("laps returned:", len(res["laps"]))
    print("first lap keys:", res["laps"][0].keys())

if __name__ == "__main__":
    main()
