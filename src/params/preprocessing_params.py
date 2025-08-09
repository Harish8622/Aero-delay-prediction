# params for preprocessing
import os


class PreprocessingParams:
    def __init__(self):
        self.start_date = "2021-07-31"  # Start date for filtering flights
        self.wind_kmh_threshold = 25  # Wind speed threshold for "wind" flag
        self.max_workers = 12  # Number of parallel fetch threads
        self.force_refresh = True  # If True, ignore cache and re-download
        self.columns_to_drop = [
            # Identifiers
            "FL_NUMBER",
            "AIRLINE_DOT",
            "AIRLINE_CODE",
            "DOT_CODE",
            # Location names (redundant with IATA codes)
            "ORIGIN_CITY",
            "DEST_CITY",
            # oout of scope / not useful for prediction
            "CRS_DEP_TIME",
            "CRS_ARR_TIME",
            "CRS_ELAPSED_TIME",
            "CANCELLED",
            "DIVERTED",
            "CRS_ELAPSED_TIME"
            # Future / leakage features
            "DEP_TIME",
            "DEP_DELAY",
            "TAXI_OUT",
            "WHEELS_OFF",
            "WHEELS_ON",
            "TAXI_IN",
            "ARR_TIME",
            "ARR_DELAY",
            "CANCELLATION_CODE",
            "ELAPSED_TIME",
            "AIR_TIME",
            "DELAY_DUE_CARRIER",
            "DELAY_DUE_WEATHER",
            "DELAY_DUE_NAS",
            "DELAY_DUE_SECURITY",
            "DELAY_DUE_LATE_AIRCRAFT",
            "DEP_TIME",
            # Raw datetime keys (already used to make features)
            "FL_DATE",
            "dep_hour_dt",
            "arr_hour_dt",
        ]
