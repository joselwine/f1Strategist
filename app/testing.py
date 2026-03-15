from __future__ import annotations
from app.services.strategy_service import StrategyService

def main() -> None:
    svc = StrategyService()
    print("✅ Assets loaded.")

    race_id = "2025-Monaco"
    driver = "VER"
    pit_lap = 28

    sim = svc.simulate(race_id=race_id, driver=driver, pit_lap=pit_lap, horizon_laps=20)
    print("\n--- WHAT-IF SIM (viewer) ---")
    print(sim["summary_viewer"])   # 👈 use viewer summary (you’ll add this below)

    current_lap = 18
    candidates = list(range(current_lap, current_lap + 6))
    reco = svc.recommend_pit_lap(race_id, driver, current_lap, candidates, horizon_laps=20)
    print("\n--- RECOMMENDATION (viewer) ---")
    print(reco["summary_viewer"])  # 👈 same idea

    pits = svc.get_real_pit_laps(race_id, driver)
    print("\n--- REAL PIT EXPLANATION (viewer) ---")
    if pits:
        exp_real = svc.explain_real_pit(race_id, driver, pit_lap=pits[0], horizon_laps=20, window=2, prefer="viewer")
        print(exp_real.get("summary_viewer", exp_real["summary"]) ) # viewer by default in the version I sent

if __name__ == "__main__":
    main()