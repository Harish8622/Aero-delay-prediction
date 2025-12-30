# import packages
import pandas as pd
from tqdm import tqdm
from meteostat import Point, Hourly
from datetime import datetime
import holidays
import warnings

warnings.simplefilter("ignore")  # silence pandas FutureWarnings

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


# functiont o do this # load and filter dataset for rows after 31-07-2024
def load_and_filter_dataset(file_path, start_date):
    """Loads the dataset and filters for rows after selected date."""
    df = pd.read_csv(file_path)
    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"])
    filtered_df = df[df["FL_DATE"] >= start_date]
    return filtered_df


# we dont want diverted or cancelled flights
def removed_diverted_or_cancelled(df):
    """
    Remove diverted or cancelled flights from the DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing flight data.

    Returns:
    pd.DataFrame: Filtered DataFrame with diverted and cancelled flights removed.
    """
    return df[(df["DIVERTED"] == 0) & (df["CANCELLED"] == 0)]


# now we want to create a delay flag for 15 and 30 minutes delay
def create_delay_flag(df, delay_threshold=15):
    """
    Create a delay flag based on the specified delay threshold.

    Parameters:
    df (pd.DataFrame): DataFrame containing flight data.
    delay_threshold (int): Delay threshold in minutes to flag as delayed.

    Returns:
    pd.DataFrame: DataFrame with an additional 'DELAY_FLAG' column.
    """
    df[f"DELAY_FLAG_{delay_threshold}"] = df["ARR_DELAY"].apply(
        lambda x: 1 if x >= delay_threshold else 0
    )
    return df


# ========= Weather Enrichment (helpers + orchestrator) =========


# ----------------------------- #
# Low-level helpers (pure funcs)
# ----------------------------- #


def _hhmm_to_hour(x) -> int:
    """Convert HHMM-ish values (930, 0930, '1730', 930.0) to an integer hour [0..23]."""
    if pd.isna(x):
        return 0
    s = str(int(float(x))).zfill(4)[:4]
    return int(s[:2])


def _add_hour_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Add dep_hour_dt / arr_hour_dt columns (local-date + hour buckets)."""
    out = df.copy()
    out["FL_DATE"] = pd.to_datetime(out["FL_DATE"])
    dep_h = out["CRS_DEP_TIME"].apply(_hhmm_to_hour)
    arr_h = out["CRS_ARR_TIME"].apply(_hhmm_to_hour)
    out["dep_hour_dt"] = pd.to_datetime(out["FL_DATE"].dt.date) + pd.to_timedelta(
        dep_h, unit="h"
    )
    out["arr_hour_dt"] = pd.to_datetime(out["FL_DATE"].dt.date) + pd.to_timedelta(
        arr_h, unit="h"
    )
    return out


def _validate_flights(df_flights: pd.DataFrame) -> None:
    """Ensure flights DF has the required columns."""
    required = {"FL_DATE", "ORIGIN", "DEST", "CRS_DEP_TIME", "CRS_ARR_TIME"}
    missing = required - set(df_flights.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _load_airports(airports_csv: str) -> pd.DataFrame:
    """Load airports CSV (must include IATA, lat, lon). Returns DF indexed by IATA with ['lat','lon']."""
    ap = pd.read_csv(airports_csv)
    need = {"IATA", "lat", "lon"}
    miss = need - set(ap.columns)
    if miss:
        raise ValueError(f"Airports CSV must contain columns: {need}. Missing: {miss}")
    ap = (
        ap.dropna(subset=["IATA", "lat", "lon"])
        .drop_duplicates("IATA")
        .set_index("IATA")[["lat", "lon"]]
    )
    return ap


def _build_needed_pairs(df: pd.DataFrame, ap: pd.DataFrame) -> pd.DataFrame:
    """From flights with hour keys, build unique (IATA, year) pairs and attach lat/lon."""
    dep_keys = df[["ORIGIN", "dep_hour_dt"]].rename(
        columns={"ORIGIN": "IATA", "dep_hour_dt": "ts"}
    )
    arr_keys = df[["DEST", "arr_hour_dt"]].rename(
        columns={"DEST": "IATA", "arr_hour_dt": "ts"}
    )
    all_keys = (
        pd.concat([dep_keys, arr_keys], ignore_index=True).dropna().drop_duplicates()
    )
    all_keys["year"] = all_keys["ts"].dt.year
    pairs = (
        all_keys[["IATA", "year"]].drop_duplicates().join(ap, on="IATA", how="inner")
    )
    return pairs


def _fetch_airport_year_raw(
    iata: str,
    lat: float,
    lon: float,
    year: int,
    cache_dir: Path,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch RAW hourly weather for one (airport, year) from Meteostat.
    Caches to Parquet as {IATA}_{year}_RAW.parquet with columns: ts, prcp, temp, wspd, IATA
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    fpath = cache_dir / f"{iata}_{year}_RAW.parquet"

    if (not force_refresh) and fpath.exists():
        return pd.read_parquet(fpath)

    p = Point(float(lat), float(lon))
    start = datetime(int(year), 1, 1)
    end = datetime(int(year), 12, 31, 23, 59, 59)

    wx = Hourly(p, start, end).fetch()  # index 'time' (UTC, hourly)
    if wx.empty:
        raw = pd.DataFrame(columns=["ts", "prcp", "temp", "wspd", "IATA"])
    else:
        raw = wx.reset_index().rename(columns={"time": "ts"})
        raw = raw[["ts", "prcp", "temp", "wspd"]]
        raw["IATA"] = iata

    raw.to_parquet(fpath, index=False)
    return raw


def _fetch_all_raw_with_progress(
    pairs: pd.DataFrame, cache_dir: Path, max_workers: int, force_refresh: bool
) -> pd.DataFrame:
    """Parallel fetch raw weather for all (IATA, year) pairs with a progress bar."""
    parts = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(
                _fetch_airport_year_raw,
                r.IATA,
                r.lat,
                r.lon,
                int(r.year),
                cache_dir,
                force_refresh,
            ): (r.IATA, int(r.year))
            for r in pairs.itertuples(index=False)
        }
        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Fetching Meteostat raw (airport-year)",
        ):
            parts.append(fut.result())

    if parts:
        out = pd.concat(parts, ignore_index=True)
        out.drop_duplicates(subset=["IATA", "ts"], inplace=True)
        return out
    return pd.DataFrame(columns=["ts", "prcp", "temp", "wspd", "IATA"])


def _compute_flags_from_raw(
    raw_df: pd.DataFrame, wind_kmh_threshold: int
) -> pd.DataFrame:
    """
    Compute binary flags from raw weather:
      rain = prcp > 0
      ice  = temp <= 0 and prcp > 0
      wind = wspd >= threshold
    Returns: ['ts','rain','ice','wind','IATA']
    """
    if raw_df.empty:
        return pd.DataFrame(columns=["ts", "rain", "ice", "wind", "IATA"])

    out = raw_df.copy()
    out["rain"] = (out["prcp"].fillna(0) > 0).astype("int8")
    out["ice"] = ((out["temp"].fillna(99) <= 0) & (out["prcp"].fillna(0) > 0)).astype(
        "int8"
    )
    out["wind"] = (out["wspd"].fillna(0) >= wind_kmh_threshold).astype("int8")
    return out[["ts", "rain", "ice", "wind", "IATA"]]


def _merge_flags(df: pd.DataFrame, wx_flags: pd.DataFrame) -> pd.DataFrame:
    """Merge dep/arr flags into flights by (IATA, hour-key) and add *_missing columns."""
    # Departure merge
    dep = (
        df[["ORIGIN", "dep_hour_dt"]]
        .rename(columns={"ORIGIN": "IATA", "dep_hour_dt": "ts"})
        .merge(wx_flags, on=["IATA", "ts"], how="left")
        .rename(columns={"rain": "dep_rain", "ice": "dep_ice", "wind": "dep_wind"})
    )
    # Arrival merge
    arr = (
        df[["DEST", "arr_hour_dt"]]
        .rename(columns={"DEST": "IATA", "arr_hour_dt": "ts"})
        .merge(wx_flags, on=["IATA", "ts"], how="left")
        .rename(columns={"rain": "arr_rain", "ice": "arr_ice", "wind": "arr_wind"})
    )

    out = df.copy()
    out["dep_rain"], out["dep_ice"], out["dep_wind"] = (
        dep["dep_rain"].values,
        dep["dep_ice"].values,
        dep["dep_wind"].values,
    )
    out["arr_rain"], out["arr_ice"], out["arr_wind"] = (
        arr["arr_rain"].values,
        arr["arr_ice"].values,
        arr["arr_wind"].values,
    )

    for col in ["dep_rain", "dep_ice", "dep_wind", "arr_rain", "arr_ice", "arr_wind"]:
        out[col] = out[col].fillna(0).astype("int8")

    return out


# ---------------------------------- #
# Orchestrator
# ---------------------------------- #


def enrich_with_weather_flags(
    df_flights: pd.DataFrame,
    airports_csv: str = "../data/raw/airports.csv",  # Must contain IATA, lat, lon
    cache_dir: str = "weather_cache",  # Where to save raw weather data
    wind_kmh_threshold: int = 30,  # Wind speed threshold for "wind" flag
    max_workers: int = 12,  # Parallel fetch threads
    force_refresh: bool = False,  # If True, ignore cache and re-download
) -> pd.DataFrame:
    """
    High-level pipeline:
      1) Validate inputs
      2) Add dep/arr hour keys
      3) Load airports & filter flights to known airports
      4) Build unique (IATA, year) fetch list
      5) Download (or load cached) RAW weather per airport-year
      6) Compute rain/ice/wind flags from RAW
      7) Merge flags back into flights
    """
    # 1) Validate
    _validate_flights(df_flights)

    # 2) Hour keys
    df_keyed = _add_hour_keys(df_flights)

    # 3) Airports & filter
    ap = _load_airports(airports_csv)
    df_keyed = df_keyed[
        df_keyed["ORIGIN"].isin(ap.index) & df_keyed["DEST"].isin(ap.index)
    ].copy()

    # 4) Needed pairs
    pairs = _build_needed_pairs(df_keyed, ap)

    # 5) Fetch RAW with progress
    cache_path = Path(cache_dir)
    raw_weather = _fetch_all_raw_with_progress(
        pairs, cache_path, max_workers, force_refresh
    )

    # 6) Compute flags
    wx_flags = _compute_flags_from_raw(raw_weather, wind_kmh_threshold)

    # 7) Merge to flights
    df_enriched = _merge_flags(df_keyed, wx_flags)
    return df_enriched


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal features to the DataFrame:
      - day_of_week (0=Mon, 6=Sun)
      - month (1-12)
      - hour_of_day (0-23) from CRS_DEP_TIME
      - is_bank_holiday (US federal holiday)

    Parameters:
    df : pd.DataFrame
        DataFrame containing 'FL_DATE' and 'CRS_DEP_TIME' columns.

    Returns:
    pd.DataFrame : DataFrame with added columns.
    """
    df = df.copy()
    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"])

    # Day of week & month
    df["day_of_week"] = df["FL_DATE"].dt.dayofweek
    df["month"] = df["FL_DATE"].dt.month

    # Scheduled departure hour
    df["hour_of_day"] = df["CRS_DEP_TIME"].apply(
        lambda x: int(str(int(x)).zfill(4)[:2]) if pd.notnull(x) else 0
    )

    # US Bank Holidays
    start_year, end_year = df["FL_DATE"].dt.year.min(), df["FL_DATE"].dt.year.max()
    us_holidays = holidays.US(years=range(start_year, end_year + 1))
    df["is_bank_holiday"] = df["FL_DATE"].isin(us_holidays).astype("int8")

    return df
