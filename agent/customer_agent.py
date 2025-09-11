import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


from dotenv import load_dotenv
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
import pandas as pd
import json
from datetime import datetime


from agent.helpers.tools import get_distance, get_temporal_features, get_weather
from src.inference.inference import predict




print('imported packages')

# Load OpenAI key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize the model
llm = ChatOpenAI(model="gpt-3.5-turbo")


# define variables for inference params here


inference_variables = {
    "cat_candidates": ["AIRLINE", "ORIGIN", "DEST"],
    "num_candidates": [
        "DISTANCE", "day_of_week", "month", "hour_of_day", "is_bank_holiday",
        "dep_rain", "dep_ice", "dep_wind", "arr_rain", "arr_ice", "arr_wind"
    ]
}

airlines = ["American Airlines", "Delta Air Lines", "United Airlines", "Southwest Airlines",
            "Alaska Airlines", "JetBlue Airways", "Spirit Airlines", "Frontier Airlines",
            "Hawaiian Airlines", "Allegiant Air", "Sun Country Airlines", "Copa Airlines",
            "Aeromexico", "Air Canada", "WestJet", "British Airways", "Lufthansa",
            "Emirates", "Qatar Airways", "Singapore Airlines"]

airports = pd.read_csv('data/raw/airports.csv')

# following headers IATA,ICAO,name,lat,lon,tz_db



class FlightParams(BaseModel):
    airline: str = Field(..., description="Selected airline")
    origin: str = Field(..., description="Departure airport IATA code")
    destination: str = Field(..., description="Arrival airport IATA code")
    timestamp: str = Field(..., description="in date time format")
# first node should decide if customer has provided airline in scope, origin and destination in scope

# -----------------------
# State definition
# -----------------------
class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

# -----------------------
# Node: Route Confirmation
# -----------------------
def route_confirmation(state: State):
    last_message = state["messages"][-1]

    # Configure LLM to return structured output
    structured_llm = llm.with_structured_output(FlightParams)
    now = datetime.now()
    response = structured_llm.invoke(
        f"""
        You are an expert travel assistant.
        Extract the airline, origin airport (IATA), destination airport (IATA), and timestamp from:
        '{last_message}'

        - Valid airlines: {', '.join(airlines)}
        - Airports must be valid IATA codes.
        - If the user says **"now"**, use {now.strftime('%Y-%m-%d %H:%M:%S')}.
        - If they give a **relative date** (e.g. "tomorrow", "next Friday", "in 3 days"),
        **calculate the exact datetime** based on {now.strftime('%Y-%m-%d %H:%M:%S')}.

        Always return the timestamp in **YYYY-MM-DD HH:MM:SS** format.
        """
    )


    # Check if airline is supported
    
    return {"messages": [{"role": "assistant", "content": f'{{"airline": "{response.airline}", "origin": "{response.origin}", "destination": "{response.destination}", "timestamp": "{response.timestamp}"}}'}]}



def check_eligibility(state: State):
    last_message = json.loads(state["messages"][-1].content)
    # ...existing code...
    if last_message["airline"] in airlines and \
       last_message["origin"] in airports['IATA'].values and \
       last_message["destination"] in airports['IATA'].values:
        return {"messages": [{"role": "assistant", "content": json.dumps({
            "airline": last_message['airline'],
            "origin": last_message['origin'],
            "destination": last_message['destination'],
            "timestamp": last_message['timestamp']
        })}]}
    else:
        return {"messages": [{"role": "assistant", "content": json.dumps({
            "error": f"Sorry, we do not support the airline {last_message['airline']} or one of the airports {last_message['origin']} or {last_message['destination']}. Please provide a different query."
        })}]}

def get_flight_distance(state: State):
    last_message = json.loads(state["messages"][-1].content)
    origin = last_message["origin"]
    destination = last_message["destination"]

    distance = get_distance(origin, destination)
    
    return {"messages": [{"role": "assistant", "content": json.dumps({
        "airline": last_message['airline'],
        "origin": last_message['origin'],
        "destination": last_message['destination'],
        "distance": distance,
        "timestamp": last_message['timestamp']
    })}]}

def get_temporal_features_node(state: State):
    last_message = json.loads(state["messages"][-1].content)
    timestamp = last_message["timestamp"]

    day_of_week, month, hour_of_day, is_bank_holiday = get_temporal_features(timestamp)
    
    return {"messages": [{"role": "assistant", "content": json.dumps({
        "airline": last_message['airline'],
        "origin": last_message['origin'],
        "destination": last_message['destination'],
        "distance": last_message['distance'],
        "timestamp": last_message['timestamp'],
        "day_of_week": day_of_week,
        "month": month,
        "hour_of_day": hour_of_day,
        "is_bank_holiday": is_bank_holiday
    })}]}

# get weather node

def weather_node(state: dict):
    last_message = json.loads(state["messages"][-1].content)
    origin_code = last_message["origin"]
    dest_code = last_message["destination"]
    timestamp = last_message["timestamp"]  # YYYY-MM-DD HH:MM:SS

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

    return {
        "messages": [{"role": "assistant", "content": json.dumps({
            "airline": last_message["airline"],
            "origin": origin_code,
            "destination": dest_code,
            "distance": last_message["distance"],        
            "timestamp": timestamp,
            "day_of_week": last_message["day_of_week"],
            "month": last_message["month"],
            "hour_of_day": last_message["hour_of_day"],
            "is_bank_holiday": last_message["is_bank_holiday"],
            "dep_rain": dep_rain,
            "dep_ice": dep_ice,
            "dep_wind": dep_wind,
            "arr_rain": arr_rain,
            "arr_ice": arr_ice,
            "arr_wind": arr_wind
        })}]
    }

def final_prediction(state: dict):
    last_message = json.loads(state["messages"][-1].content)
    
    proba, prediction = predict(
        airline=last_message["airline"],
        origin=last_message["origin"],
        destination=last_message["destination"],
        distance=last_message["distance"],
        day_of_week=last_message["day_of_week"],
        month=last_message["month"],
        hour_of_day=last_message["hour_of_day"],
        is_bank_holiday=last_message["is_bank_holiday"],
        dep_rain=last_message["dep_rain"],
        dep_ice=last_message["dep_ice"],
        dep_wind=last_message["dep_wind"],
        arr_rain=last_message["arr_rain"],
        arr_ice=last_message["arr_ice"],
        arr_wind=last_message["arr_wind"]
    )
    
    return {
        "messages": [{"role": "assistant", "content": json.dumps({
            "probability_of_delay": proba,
            "delay_prediction": "Delayed" if prediction == 1 else "On Time"
        })}]
    }




# -----------------------
# Build Graph
# -----------------------
graph_builder.add_node("route_confirmation", route_confirmation)
graph_builder.add_node("check_eligibility", check_eligibility)
graph_builder.add_node("get_flight_distance", get_flight_distance)
graph_builder.add_node("get_temporal_features", get_temporal_features_node)
graph_builder.add_node("get_weather", weather_node)
graph_builder.add_node("final_prediction", final_prediction)


graph_builder.add_edge(START, "route_confirmation")
graph_builder.add_edge("route_confirmation", "check_eligibility")
graph_builder.add_edge("check_eligibility", "get_flight_distance")
graph_builder.add_edge("get_flight_distance", "get_temporal_features")
graph_builder.add_edge("get_temporal_features", "get_weather")
graph_builder.add_edge("get_weather", "final_prediction")
graph_builder.add_edge("final_prediction", END)
# graph_builder.add_edge("check_eligibility", END)  # End if not eligible
customer_agent = graph_builder.compile()

# -----------------------
# Run the Agent
# -----------------------
user_input = input("Enter your flight details query: ")
initial_state = {"messages": [{"role": "user", "content": user_input}]}

state = customer_agent.invoke(initial_state)
print("\n🤖 Assistant:", state["messages"][-1].content)
# last_message = json.loads(state["messages"][-1].content)
# origin = last_message["origin"]
# destination = last_message["destination"]

# distance = get_distance(origin, destination)

# print(distance)
    # neeed to add tools so initially it asks customer for flight details
    # then uses tools to determine the required params
    # this is conditional edge depending on if there is enough info
    # use pydanbtic to ensure correct params
    # then uses tool to call inference
    # then returns to user