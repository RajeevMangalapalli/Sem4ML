from pathlib import Path
import pandas as pd

FILE_PATH_LAUNCHES = Path('Project_Work/data/raw/spacex_launches.csv')
#FILE_PATH_ROCKETS = Path('../data/raw/spacex_launches.csv')
OUTPUT_PATH = Path('Project_Work/data/processed/processed_data.csv')

#Load the dataset into a pandas data frame

df = pd.read_csv(FILE_PATH_LAUNCHES)
#rockets = pd.read_csv(FILE_PATH_ROCKETS)
#print(rockets.head())
#print()
#print(df.head())

"""
Features
flight_number, name, date_utc, rocket_id,
rocket_name, launchpad_id, launchpad_name, success, failures, 
details, crew_count, payloads_count, cores_reused, landing_success, landing_type

Features to remove:
date_utc, rocket_id, details

"""

df.drop(["date_utc","rocket_id","details","landing_type","launchpad_id"], axis=1, inplace = True)

#Save the processed data to a new CSV file
df.to_csv(OUTPUT_PATH, index=False)





