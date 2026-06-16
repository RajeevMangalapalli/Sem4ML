from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

INPUT_PATH_LAUNCHES = RAW_DATA_DIR / "spacex_launches.csv"

PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH_LAUNCHES = PROCESSED_DATA_DIR / "processed_data.csv"


# Load the dataset into a pandas data frame
df = pd.read_csv(INPUT_PATH_LAUNCHES)
"""
Features
flight_number, name, date_utc, rocket_id,
rocket_name, launchpad_id, launchpad_name, success, failures, 
details, crew_count, payloads_count, cores_reused, landing_success, landing_type

Features to remove:
date_utc, rocket_id, details
"""

df.drop(
    df[(df["success"] == 0) & (df["failures"] == 0)].index,
    inplace=True,
)

df.drop(
    ["date_utc", "rocket_id", "details", "landing_type", "launchpad_id"],
    axis=1,
    inplace=True,
)

# Save the processed data to a new CSV file
df.to_csv(OUTPUT_PATH_LAUNCHES, index=False)
