from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

INPUT_PATH_LAUNCHES = RAW_DATA_DIR / "spacex_launches.csv"
INPUT_PATH_ROCKETS = RAW_DATA_DIR / "spacex_rockets.csv"

PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH_LAUNCHES = PROCESSED_DATA_DIR / "processed_data.csv"


df = pd.read_csv(INPUT_PATH_LAUNCHES)
rockets = pd.read_csv(INPUT_PATH_ROCKETS)

rockets.rename(
    columns={
        "active": "rocket_active",
        "cost_per_launch": "rocket_cost_per_launch",
        "success_rate": "rocket_success_rate",
        "height_m": "rocket_height_m",
        "diameter_m": "rocket_diameter_m",
        "mass_kg": "rocket_mass_kg",
    },
    inplace=True,
)

rockets.drop(
    ["name", "type", "stages", "first_flight", "description"],
    axis="columns",
    inplace=True,
)

df = df.merge(rockets, left_on="rocket_id", right_on="id", how="left")

df.drop(["id"], axis="columns", inplace=True)

df.drop(df[(df["success"] == 0) & (df["failures"] == 0)].index, inplace=True)

df.drop(
    [
        "flight_number",
        "date_utc",
        "rocket_id",
        "failures",
        "launchpad_id",
        "details",
    ],
    axis="columns",
    inplace=True,
)

# Save the processed data to a new CSV file
df.to_csv(OUTPUT_PATH_LAUNCHES, index=False)
