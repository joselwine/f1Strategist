from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


ALL_RACES = {
    2022: ["Bahrain", "Saudi Arabia", "Australia", "Emilia-Romagna", "Azerbaijan",
    "Miami", "Monaco", "Spain", "Canada", "France",
    "Austria", "Britain", "Hungary", "Belgium", "Netherlands", "Italy", "Singapore", "Japan", "Qatar", "United States", "Mexico", "Brazil", "Las Vegas", "Abu Dhabi"
    ],
    2023: ["Bahrain", "Saudi Arabia", "Australia", "Azerbaijan", "Emilia-Romagna",
    "Miami", "Monaco", "Spain", "Canada",
    "Austria", "Britain", "Hungary", "Belgium", "Netherlands", "Italy", "Singapore", "Japan", "Qatar", "United States", "Mexico", "Brazil", "Las Vegas", "Abu Dhabi"
    ],
    2024: ["Bahrain", "Saudi Arabia", "Australia", "Azerbaijan", "China", "Emilia-Romagna",
    "Miami", "Monaco", "Spain", "Canada",
    "Austria", "Britain", "Hungary", "Belgium", "Netherlands", "Italy", "Singapore", "Japan", "Qatar", "United States", "Mexico", "Brazil", "Las Vegas", "Abu Dhabi"
    ],
    2025: ["Bahrain", "Saudi Arabia", "Australia", "Azerbaijan",
    "Miami", "Monaco", "Spain", "Canada", "Emilia-Romagna", 
    "Austria", "Britain", "Hungary", "Belgium", "Netherlands", "Italy", "Singapore", "Japan", "United States", "Mexico", "Brazil", "Las Vegas"
    ],   
}

TRAIN_RACES = [
    (year, race)
    for year in [2022, 2023, 2024, 2025]
    for race in ALL_RACES[year][:-2]     # leaves last two as test
]

TEST_RACES = [
    (year, ALL_RACES[year][-1])
    for year in [2022, 2023, 2024, 2025]
]

