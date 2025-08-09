# pre processing script for the Aero-delay-prediction project
import os
import pandas as pd
import argparse
from src.helpers.preprocessing_helpers import (
    load_and_filter_dataset,
    removed_diverted_or_cancelled,
    create_delay_flag,
    enrich_with_weather_flags,
    add_temporal_features,
)
from src.params.preprocessing_params import PreprocessingParams

pp_params = PreprocessingParams()

parser = argparse.ArgumentParser(description="Preprocess flight data")
parser.add_argument(
    "--input_flights_table", type=str, required=True, help="Flights table file path"
)
parser.add_argument(
    "--input_airport_table", type=str, required=True, help="Airport table file path"
)
parser.add_argument("--output_path", type=str, required=True, help="Output file path")
parser.add_argument(
    "--cache_dir", type=str, required=True, help="Cache directory for raw weather"
)

args = parser.parse_args()


# Load and filter dataset
df = load_and_filter_dataset(
    file_path=args.input_flights_table, start_date=pp_params.start_date
)

# Remove diverted or cancelled flights
df = removed_diverted_or_cancelled(df)

# Create delay flag
df = create_delay_flag(df, delay_threshold=15)

# Enrich with weather flags
df = enrich_with_weather_flags(
    df_flights=df,
    airports_csv=args.input_airport_table,
    cache_dir=args.cache_dir,
    wind_kmh_threshold=pp_params.wind_kmh_threshold,
    max_workers=pp_params.max_workers,
    force_refresh=pp_params.force_refresh,
)

# Add temporal features
df = add_temporal_features(df)

# Drop unnecessary columns
df_dropped = df.drop(columns=pp_params.columns_to_drop, errors="ignore")
# Save final data
os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
df_dropped.to_csv(args.output_path, index=False)
print(f"Preprocessed data saved to {args.output_path}")
# --- END OF FILE ---
