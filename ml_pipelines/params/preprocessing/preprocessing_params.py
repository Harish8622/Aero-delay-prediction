import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class PreprocessingParams:
    def __init__(self):
        base_path = os.getenv("BASE_PATH")
        if not base_path:
            raise ValueError("BASE_PATH not set in .env")

        self.output_path = os.path.join(
            base_path, "test_data/data/processed/preprocessed_data.csv"
        )
        self.input_flights_table = os.path.join(
            base_path, "data/raw/flights_sample_3m.csv"
        )
        self.input_airport_table = os.path.join(base_path, "data/raw/airports.csv")
        self.cache_dir = os.path.join(base_path, "notebooks/weather_cache")

        # Ensure required directories exist
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.input_flights_table), exist_ok=True)
        os.makedirs(os.path.dirname(self.input_airport_table), exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
