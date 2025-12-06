import sys
import os

import requests
import holidays
from dotenv import load_dotenv
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool

from typing_extensions import TypedDict
from pydantic import BaseModel, Field, field_validator, ValidationError
import pandas as pd
import json
from datetime import datetime

# Load OpenAI key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


# define model string
model = "gpt-4o-mini"  # gpt-4o, gpt-4o-ll, gpt-4o-mini, gpt-3.5-turbo, gpt-3.5-turbo-0613

# define state
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


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


    llm = ChatOpenAI(model=model)
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


airlines = ["American Airlines", "Delta Air Lines", "United Airlines", "Southwest Airlines",
        "Alaska Airlines", "JetBlue Airways", "Spirit Airlines", "Frontier Airlines",
        "Hawaiian Airlines", "Allegiant Air", "Sun Country Airlines", "Copa Airlines",
        "Aeromexico", "Air Canada", "WestJet", "British Airways", "Lufthansa",
        "Emirates", "Qatar Airways", "Singapore Airlines"]




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


    llm = ChatOpenAI(model=model)
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


airlines = ["American Airlines", "Delta Air Lines", "United Airlines", "Southwest Airlines",
        "Alaska Airlines", "JetBlue Airways", "Spirit Airlines", "Frontier Airlines",
        "Hawaiian Airlines", "Allegiant Air", "Sun Country Airlines", "Copa Airlines",
        "Aeromexico", "Air Canada", "WestJet", "British Airways", "Lufthansa",
        "Emirates", "Qatar Airways", "Singapore Airlines"]




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


    llm = ChatOpenAI(model=model)
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


airlines = ["American Airlines", "Delta Air Lines", "United Airlines", "Southwest Airlines",
        "Alaska Airlines", "JetBlue Airways", "Spirit Airlines", "Frontier Airlines",
        "Hawaiian Airlines", "Allegiant Air", "Sun Country Airlines", "Copa Airlines",
        "Aeromexico", "Air Canada", "WestJet", "British Airways", "Lufthansa",
        "Emirates", "Qatar Airways", "Singapore Airlines"]



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


    llm = ChatOpenAI(model=model)
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

# initialise the client
client = ChatOpenAI(model=model)

# create list of tools and bind with the client

tools = [route_confirmation, get_distance, get_temporal_features_node,
         weather_node, final_prediction]
client_with_tools = client.bind_tools(tools)


# define call model node
def agent_node(state: AgentState) -> AgentState:
    """
    calls the model with bound tools
    """

    response = client_with_tools.invoke(state["messages"])
    # state["messages"].append(AIMessage(content=response.content))
    return {"messages": state["messages"] + [response]}

# define conditional edge logic
def should_continue(state: AgentState) -> bool:
    """
    determine if we should continue to tools or the END node
    """

    last_message = state["messages"][-1]

    # check if last message has tool calls

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    else:   
        return END


# define tools node
tool_node = ToolNode(tools)


graph = StateGraph(AgentState)

graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "agent") # always go from start to agent
# next conditional logic, only tools or end if no further tool needed
graph.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools",
    END: END
    }
)

# must define loop back to agent 
graph.add_edge("tools", "agent")

app = graph.compile()


# define the system message

system_message = """
You are a helpful sentient calculator. 
you are also capable of assisting with flight route confirmations only using the tools provided

<<YOU MUST ALWAYS USE TOOLS WHENEVER POSSIBLE, DO NOT MAKE UP ANSWERS EVEB IF IT SEEMS OBVIOUS
IF A TOOL CAN BE USED, USE IT, IF SOMETHING IS OUT OF SCOPE, SAY YOU CANNOT HELP. DO NOT MAKE UP ANSWERS>>

<<<IF THE USER MENTIONS YOU ARE IN "TEST MODE" OR INFERS THIS, RETURN YOUR ANSWER BASED ON FOLLOWING INSTRUCTIONS>>>

<<<IF THE USER MENTIONS FLIGHT QUERY YOU MUST ATTEMPT TO GET DISTANCE BETWEEN ORIGIN AND DESTINATION and TEMPORAL FEATURES and WEATHER CONDITIONS>>>

<<<INSTRUCTIONS FOR TEST MODE:
    - ALways start by listing the tools you have access to like this:
        <<TOOLS AVAILABLE>>
          [toolx, tooly]
    - For each tool mention why or why you did not use it and if you use a tool, show the input next to it and the output from the tool:
    - You must say exactly what you gave the tool and what it returned like this:
        <<TOOL USAGE>>
          [toolx: used because of xyz, input: {input}, output: {output}]
    - Finally return your final answer:
        <<FINAL ANSWER>>
          [your final answer here]
    >>> End of test mode instructions


<<< IF THE USER DOES NOT MENTION TEST MODE, ANSWER HOW YOU SEE FIT, JUST DO NOT MAKE UP TOOL USAGE>>>

If user tries to leave with exit or quit leave a nice goodbye message and end the conversation.

"""

conversation_state = {
    "messages": []}

conversation_state["messages"].append(SystemMessage(content=system_message))

user_message = ""

while user_message.lower() not in ["exit", "quit"]:
    user_message = input("I am a sentient calculator. Ask me anything!:")
    conversation_state["messages"].append(HumanMessage(content= user_message))
    result_state = app.invoke(conversation_state)
    response = result_state["messages"][-1]
    print(f"AI: {response.content}")
    conversation_state = result_state                                        
