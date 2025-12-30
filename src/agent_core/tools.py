import requests
import holidays
from dotenv import load_dotenv
from langchain_core.tools import tool

from pydantic import ValidationError
import pandas as pd

from datetime import datetime
from src.inference.inference import predict
from src.agent_core.data_models import(
    FlightParams, 
    FinalPrediction,
    DistanceParams,
    TemporalParams,
    WeatherParams,
    airlines, 
    airports,
)

<<<<<<< HEAD:agent/helpers/tools.py
from agent.helpers.prompts import build_route_confirmation_prompt
from agent.model_config import llm
=======
from src.agent_core.prompts import build_route_confirmation_prompt
from src.agent_core.model_config import llm
>>>>>>> origin/main:src/agent_core/tools.py
import os

load_dotenv()
VISUAL_CROSSING_KEY = os.getenv("VISUAL_CROSSING_KEY")


# -----------------------
# Tool: Route Confirmation
# -----------------------
@tool
def route_confirmation(user_query: str):
    """
    for a user query extract airline, origin, destination and timestamp. 
    for example "I am flying from JFK to LAX on Delta Air Lines next Friday at noon"
    If there is an error in extraction or validation, return the error message.

    args: user_query: str - the user's input query

    returns: dictionary with keys airline, origin, destination, timestamp
    
    """
    now = datetime.now()

    prompt = build_route_confirmation_prompt(
        user_query=user_query,
        airlines=airlines,
        now=now,
    )
    structured_llm = llm.with_structured_output(FlightParams)
    try:
        response: FlightParams = structured_llm.invoke(prompt)
    except ValidationError as e:
        return f"Invalid input: {e.errors()}"
    
    return response

# -----------------------
# Tool: Get Distance
# -----------------------

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

@tool(args_schema=DistanceParams)
def get_distance(origin, destination):
    """
    given two IATA codes, return the distance between them
    """
    try:
        origin_data = airports[airports['IATA'] == origin].iloc[0]
        dest_data = airports[airports['IATA'] == destination].iloc[0]
        lat1, lon1 = origin_data['lat'], origin_data['lon']
        lat2, lon2 = dest_data['lat'], dest_data['lon']
        return distance(lat1, lon1, lat2, lon2)
    except IndexError:
        raise ValueError("Invalid IATA code provided for origin or destination.")


# -----------------------
# Tool: Get Temporal Features
# -----------------------

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


@tool(args_schema=TemporalParams)
def get_temporal_features_node(timestamp: datetime):
    """
    given a timestamp, return day_of_week, month, hour_of_day, is_bank_holiday
    """
    day_of_week, month, hour_of_day, is_bank_holiday = get_temporal_features(timestamp)
    return day_of_week, month, hour_of_day, is_bank_holiday


# -----------------------
# Tool: Get Weather
# -----------------------

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

@tool(args_schema=WeatherParams)
def weather_node(origin: str, destination: str, timestamp: str):
    """ 
    given origin, destination IATA codes and timestamp, return weather conditions at both airports
    timestamp in format %Y-%m-%d %H:%M:%S

    returns:
        dep_rain (int): 1 if rain at departure, else 0
        dep_ice (int): 1 if ice at departure, else 0
        dep_wind (int): 1 if high wind at departure, else 0 
        arr_rain (int): 1 if rain at arrival, else 0
        arr_ice (int): 1 if ice at arrival, else 0  
        arr_wind (int): 1 if high wind at arrival, else 0
        

    
    """
    origin_code = origin
    dest_code = destination
    timestamp = timestamp

    # Extract date only for API
    date = timestamp.strftime("%Y-%m-%d")

    # Get airport lat/lon from CSV
    origin = airports[airports["IATA"] == origin_code].iloc[0]
    dest = airports[airports["IATA"] == dest_code].iloc[0]

    origin_weather = get_weather(origin["lat"], origin["lon"], date)
    dest_weather = get_weather(dest["lat"], dest["lon"], date)

    # Map to inference model features
    dep_rain = 1 if origin_weather["rain_mm"] > 0 else 0
    dep_ice = 1 if origin_weather["temperature"] < 0 else 0  # Simplified
    dep_wind = 1 if origin_weather["wind_speed"] > 30 else 0
    arr_rain = 1 if dest_weather["rain_mm"] > 0 else 0
    arr_ice = 1 if dest_weather["temperature"] < 0 else 0  # Simplified
    arr_wind = 1 if dest_weather["wind_speed"] > 30 else 0

    return dep_rain, dep_ice, dep_wind, arr_rain, arr_ice, arr_wind



# -----------------------
# Tool: Final Prediction
# -----------------------

@tool(args_schema=FinalPrediction)
def final_prediction(airline, origin, destination, distance, day_of_week, month, hour_of_day, is_bank_holiday,
                     dep_rain, dep_ice, dep_wind,       
                     arr_rain, arr_ice, arr_wind):
    
    """
    given all features, return delay probability and prediction"""
        
    proba, prediction = predict(
        airline=airline,
        origin=origin,
        destination=destination,
        distance=distance,
        day_of_week=day_of_week,
        month=month,
        hour_of_day=hour_of_day,
        is_bank_holiday=is_bank_holiday,
        dep_rain=dep_rain,
        dep_ice=dep_ice,
        dep_wind=dep_wind,       
        arr_rain=arr_rain,
        arr_ice=arr_ice,
        arr_wind=arr_wind
    )

    
    return {"delay_probability": proba, "delay_prediction": prediction}

# define a list of tools
tools = [route_confirmation, get_distance, get_temporal_features_node,
         weather_node, final_prediction]