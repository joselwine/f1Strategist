from app.services.strategy_service import StrategyService

def main():
    service = StrategyService()

    # --- single simulation test ---
    sim = service.simulate(
        race_id="2024-Abu Dhabi",
        driver="LEC",
        pit_lap=20,
        horizon_laps=20
    )

    print("\n--- SIMULATION RESULT ---")
    print(sim["summary"])

    # --- recommendation test ---
    reco = service.recommend_pit_lap(
        race_id="2024-Abu Dhabi",
        driver="LEC",
        current_lap=15,
        candidate_laps=[18, 20, 22, 25],
        horizon_laps=20
    )

    print("\n--- RECOMMENDATION ---")
    print(reco["summary"])

if __name__ == "__main__":
    main()
