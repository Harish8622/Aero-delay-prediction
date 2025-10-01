import pandas as pd
import holidays
import os
import requests
from datetime import datetime
from dotenv import load_dotenv

from src.inference.inference import predict

# Load Visual Crossing API key
load_dotenv()
VISUAL_CROSSING_KEY = os.getenv("VISUAL_CROSSING_KEY")

inference_variables = {
    "cat_candidates": ["AIRLINE", "ORIGIN", "DEST"],
    "num_candidates": [
        "DISTANCE", "day_of_week", "month", "hour_of_day", "is_bank_holiday",
        "dep_rain", "dep_ice", "dep_wind", "arr_rain", "arr_ice", "arr_wind"
    ]
}

# change this later need to make consistent with pre processing params
weather_thresholds = {
    "rain_mm": 0,  # Example threshold for rain in mm
    "ice_mm": 0,   # Example threshold for ice in mm
    "wind_kmh": 30.0 # Example threshold for wind in km/h
}



airport = pd.read_csv('data/raw/airports.csv')

# following headers IATA,ICAO,name,lat,lon,tz_db

def distance(lat1, lon1, lat2, lon2):
    """
    Calculate the Haversine distance between two points on the Earth.
    """
    from math import radians, sin, cos, sqrt, atan2

    R = 6371.0  # Radius of the Earth in kilometers

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c

    return distance
def get_distance(origin, destination):
    """
    given two IATA codes, return the distance between them
    """
    try:
        origin_data = airport[airport['IATA'] == origin].iloc[0]
        dest_data = airport[airport['IATA'] == destination].iloc[0]
        lat1, lon1 = origin_data['lat'], origin_data['lon']
        lat2, lon2 = dest_data['lat'], dest_data['lon']
        return distance(lat1, lon1, lat2, lon2)
    except IndexError:
        raise ValueError("Invalid IATA code provided for origin or destination.")
    

def get_temporal_features(timestamp):
    """
    Given a timestamp, return day_of_week, month, hour_of_day, is_bank_holiday 
    think about us time later
    """
    dt = pd.to_datetime(timestamp)
    day_of_week = dt.dayofweek  # Monday=0, Sunday=6
    month = dt.month
    hour_of_day = dt.hour

    us_holidays = holidays.US()
    is_bank_holiday = int(dt.date() in us_holidays)

    return day_of_week, month, hour_of_day, is_bank_holiday




def get_weather(lat: float, lon: float, date: str = "now"):
    """
    Fetch live, historical, or forecast weather from Visual Crossing API.
    
    Args:
        lat (float): Latitude of location.
        lon (float): Longitude of location.
        date (str): Date in 'YYYY-MM-DD' or "now".
                    - "now" → live weather
                    - future date → forecast
                    - past date → historical

    Returns:
        dict: Weather data (temp, rain, snow, wind, conditions)
    """
    base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
    
    # Build API URL
    url = f"{base_url}/{lat},{lon}/{date}"
    params = {
        "key": VISUAL_CROSSING_KEY,
        "unitGroup": "metric",
        "include": "current,hours,days"
    }
    
    resp = requests.get(url, params=params)
    data = resp.json()
    
    if "days" not in data:
        raise ValueError(f"Weather API error: {data}")
    
    # Pick the first day for simplicity
    weather = data["days"][0]
    return {
        "temperature": weather.get("temp", None),
        "wind_speed": weather.get("windspeed", None),
        "rain_mm": weather.get("precip", 0),
        "snow_mm": weather.get("snow", 0),
        "conditions": weather.get("conditions", ""),
        "date": weather.get("datetime", date)
    }

