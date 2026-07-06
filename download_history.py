import nfl_data_py as nfl

# Define the years you want to pressure-test against
years = [2019, 2020, 2021, 2022, 2023]

print("Downloading Play-by-Play data... (This may take a few minutes)")
pbp_df = nfl.import_pbp_data(years)

print("Downloading historical betting market lines...")
lines_df = nfl.import_schedules(years)

print("Saving to Parquet format...")
pbp_df.to_parquet("historical_pbp.parquet", index=False)
lines_df.to_parquet("historical_lines.parquet", index=False)

print("Download complete. Data saved to your local folder.")
