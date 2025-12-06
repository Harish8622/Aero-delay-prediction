import requests
import holidays
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool

from typing_extensions import TypedDict
from pydantic import BaseModel, Field, field_validator, ValidationError
import pandas as pd
import json
from datetime import datetime

from agent.model_config import llm
import os


airlines = ["American Airlines", "Delta Air Lines", "United Airlines", "Southwest Airlines",
        "Alaska Airlines", "JetBlue Airways", "Spirit Airlines", "Frontier Airlines",
        "Hawaiian Airlines", "Allegiant Air", "Sun Country Airlines", "Copa Airlines",
        "Aeromexico", "Air Canada", "WestJet", "British Airways", "Lufthansa",
        "Emirates", "Qatar Airways", "Singapore Airlines"]


airports = pd.read_csv('data/raw/airports.csv')


class FlightParams(BaseModel):
    airline: str = Field(..., description="Selected airline")
    origin: str = Field(..., description="Departure airport IATA code")
    destination: str = Field(..., description="Arrival airport IATA code")
    timestamp: datetime = Field(..., description="Flight departure time in UTC")

    # Validate airline
    @field_validator("airline")
    def airline_supported(cls, v):
        if v not in airlines:
            raise ValueError(f"Unsupported airline: {v}")
        return v

    # Validate origin
    @field_validator("origin")
    def origin_supported(cls, v):
        if v not in airports['IATA'].values:
            raise ValueError(f"Unsupported origin airport: {v}")
        return v

    # Validate destination
    @field_validator("destination")
    def destination_supported(cls, v):
        if v not in airports['IATA'].values:
            raise ValueError(f"Unsupported destination airport: {v}")
        return v

# -----------------------
# Node: Route Confirmation
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


    structured_llm = llm.with_structured_output(FlightParams)
    try:
        response: FlightParams = structured_llm.invoke(
            f"""
            You are an expert travel assistant.
            Extract the airline, origin IATA, destination IATA, and timestamp from:
            '{user_query}'

            - Valid airlines: {', '.join(airlines)}
            - Airports must be valid IATA codes. Infer them if needed.
            - If user says "now", use {now.strftime('%Y-%m-%d %H:%M:%S')}.
            - If relative date, resolve to absolute datetime.
            if user query is vague return best guess ensuring to meet the validation criteria

            Always return timestamp in YYYY-MM-DD HH:MM:SS.
            """
        )

    except ValidationError as e:
        # Handle validation errors directly from Pydantic
        return f"Invalid input: {e.errors()}"
    

    if response.origin not in airports['IATA'].values:
        return f"Invalid input: Unsupported origin airport: {response.origin}"
    if response.destination not in airports['IATA'].values:
        return f"Invalid input: Unsupported destination airport: {response.destination}"

    return response





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

@tool
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


@tool
def get_temporal_features_node(timestamp: datetime):
    """
    given a timestamp, return day_of_week, month, hour_of_day, is_bank_holiday
    """
    day_of_week, month, hour_of_day, is_bank_holiday = get_temporal_features(timestamp)
    return day_of_week, month, hour_of_day, is_bank_holiday
# helper
load_dotenv()
VISUAL_CROSSING_KEY = os.getenv("VISUAL_CROSSING_KEY")

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

@tool
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
    date = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d")

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



from src.inference.inference import predict


@tool
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