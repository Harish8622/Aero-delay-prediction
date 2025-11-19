import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


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

from src.inference.inference import predict




print('imported packages')

# Load OpenAI key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize the model
llm = ChatOpenAI(model="gpt-4o-mini")


# define variables for inference params here


inference_variables = {
    "cat_candidates": ["AIRLINE", "ORIGIN", "DEST"],
    "num_candidates": [
        "DISTANCE", "day_of_week", "month", "hour_of_day", "is_bank_holiday",
        "dep_rain", "dep_ice", "dep_wind", "arr_rain", "arr_ice", "arr_wind"
    ]
}

# following headers IATA,ICAO,name,lat,lon,tz_db


# -----------------------
# State definition
# -----------------------
class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

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


tools = [route_confirmation]
model_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools)


def agent_node(state: dict):
    messages = list(state["messages"])
    response = model_with_tools.invoke(messages)

    # Make sure we ALWAYS return the AIMessage itself
    if not isinstance(response, AIMessage):
        raise ValueError("Expected AIMessage from LLM with tools")

    return {"messages": [response]}





def route_after_agent(state: State) -> str:
    """Decide where to go after the agent runs."""
    last_msg = state["messages"][-1]
    
    # If the last message is an AIMessage *and* it includes tool calls → run tool node
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "tool_call"
    
    # Otherwise, we’re done
    return "default"

graph_builder.add_node("agent_node", agent_node)
graph_builder.add_node("tool_node", tool_node)


graph_builder.add_edge(START, "agent_node")


graph_builder.add_conditional_edges(
    "agent_node",
    route_after_agent,   # 👈 callback function
    {
        "tool_call": "tool_node",
        "default": END
    }
)


graph_builder.add_edge("tool_node", "agent_node")

app = graph_builder.compile()


system_message = """
You are an expert travel assistant. 
Your job is to help customers with questions about their flights — including extracting key flight details, estimating the probability of delays, and providing concise, accurate advice.

You have access to one or more tools that you can use to assist with tasks (like extracting structured flight info from a query). 
For **every** answer you provide, you must follow this exact format:

---

**1. Tools Available:**
List all the tools you have access to.

**2. Tool Decisions:**
For each tool, explain whether you will use it and why.  
Use this structure:

- Tool 1: `route_confirmation` – I **will use** this tool because [...]
- Tool 2: `<tool_name>` – I **will NOT use** this tool because [...]

(Do this for *every* tool.)

**3. Final Answer:**
Provide your final answer to the user based on the information you have and any tools you used.

---

Always make your explanations short, clear, and helpful.  
Do **not** skip the tool reasoning step — it’s required in every single response.
You MUST call the `route_confirmation` tool whenever the user's query mentions a flight, origin/destination, or time — do NOT try to parse it yourself.
"""



conversation_state = {
    "messages": [
        SystemMessage(content=system_message),
    ]
}





user_input = ""
while user_input.lower() != "exit":
    user_input = input("Enter your travel query (or 'exit' to quit): ")
    if user_input.lower() == "exit":
        break

    # Add user message
    conversation_state["messages"].append(HumanMessage(content=user_input))

    # Invoke the graph
    result_state = app.invoke(conversation_state)

    # Extract the assistant’s reply (AIMessage)
    assistant_response = result_state["messages"][-1]

    conversation_state["messages"].append(assistant_response)

    print("Assistant:", assistant_response.content)
# now combine everything else apart from prediction into one node


# i will recreate the sentient calculator first

print("Debug output")
print(conversation_state)
# def get_flight_distance(state: State):
#     last_message = json.loads(state["messages"][-1].content)
#     origin = last_message["origin"]
#     destination = last_message["destination"]

#     distance = get_distance(origin, destination)
    
#     return {"messages": [{"role": "assistant", "content": json.dumps({
#         "airline": last_message['airline'],
#         "origin": last_message['origin'],
#         "destination": last_message['destination'],
#         "distance": distance,
#         "timestamp": last_message['timestamp']
#     })}]}

# def get_temporal_features_node(state: State):
#     last_message = json.loads(state["messages"][-1].content)
#     timestamp = last_message["timestamp"]

#     day_of_week, month, hour_of_day, is_bank_holiday = get_temporal_features(timestamp)
    
#     return {"messages": [{"role": "assistant", "content": json.dumps({
#         "airline": last_message['airline'],
#         "origin": last_message['origin'],
#         "destination": last_message['destination'],
#         "distance": last_message['distance'],
#         "timestamp": last_message['timestamp'],
#         "day_of_week": day_of_week,
#         "month": month,
#         "hour_of_day": hour_of_day,
#         "is_bank_holiday": is_bank_holiday
#     })}]}

# # get weather node

# def weather_node(state: dict):
#     last_message = json.loads(state["messages"][-1].content)
#     origin_code = last_message["origin"]
#     dest_code = last_message["destination"]
#     timestamp = last_message["timestamp"]  # YYYY-MM-DD HH:MM:SS

#     # Extract date only for API
#     date = datetime.fromisoformat(timestamp.replace("Z", "+00:00")).strftime("%Y-%m-%d")

#     # Get airport lat/lon from CSV
#     origin = airports[airports["IATA"] == origin_code].iloc[0]
#     dest = airports[airports["IATA"] == dest_code].iloc[0]

#     origin_weather = get_weather(origin["lat"], origin["lon"], date)
#     dest_weather = get_weather(dest["lat"], dest["lon"], date)

#     # Map to inference model features
#     dep_rain = 1 if origin_weather["rain_mm"] > 0 else 0
#     dep_ice = 1 if origin_weather["temperature"] < 0 else 0  # Simplified
#     dep_wind = 1 if origin_weather["wind_speed"] > 30 else 0
#     arr_rain = 1 if dest_weather["rain_mm"] > 0 else 0
#     arr_ice = 1 if dest_weather["temperature"] < 0 else 0  # Simplified
#     arr_wind = 1 if dest_weather["wind_speed"] > 30 else 0

#     return {
#         "messages": [{"role": "assistant", "content": json.dumps({
#             "airline": last_message["airline"],
#             "origin": origin_code,
#             "destination": dest_code,
#             "distance": last_message["distance"],        
#             "timestamp": timestamp,
#             "day_of_week": last_message["day_of_week"],
#             "month": last_message["month"],
#             "hour_of_day": last_message["hour_of_day"],
#             "is_bank_holiday": last_message["is_bank_holiday"],
#             "dep_rain": dep_rain,
#             "dep_ice": dep_ice,
#             "dep_wind": dep_wind,
#             "arr_rain": arr_rain,
#             "arr_ice": arr_ice,
#             "arr_wind": arr_wind
#         })}]
#     }

# def final_prediction(state: dict):
#     last_message = json.loads(state["messages"][-1].content)
    
#     proba, prediction = predict(
#         airline=last_message["airline"],
#         origin=last_message["origin"],
#         destination=last_message["destination"],
#         distance=last_message["distance"],
#         day_of_week=last_message["day_of_week"],
#         month=last_message["month"],
#         hour_of_day=last_message["hour_of_day"],
#         is_bank_holiday=last_message["is_bank_holiday"],
#         dep_rain=last_message["dep_rain"],
#         dep_ice=last_message["dep_ice"],
#         dep_wind=last_message["dep_wind"],
#         arr_rain=last_message["arr_rain"],
#         arr_ice=last_message["arr_ice"],
#         arr_wind=last_message["arr_wind"]
#     )
    
#     return {
#         "messages": [{"role": "assistant", "content": json.dumps({
#             "probability_of_delay": proba,
#             "delay_prediction": "Delayed" if prediction == 1 else "On Time",
#             "airline": last_message["airline"],
#             "origin": last_message["origin"],
#             "destination": last_message["destination"],
#             "distance": last_message["distance"],        
#             "timestamp": last_message["timestamp"],
#             "dep_ice": last_message['dep_ice'],
#             "dep_rain": last_message['dep_rain'],
#             "dep_wind": last_message['dep_wind'],
#             "arr_ice": last_message['arr_ice'],
#             "arr_rain": last_message['arr_rain'],
#             "arr_wind": last_message['arr_wind']

#         })}]
#     }

# def final_response(state: dict):
#     last_message = json.loads(state["messages"][-1].content)
#     airline = last_message["airline"]
#     origin = last_message["origin"]
#     destination = last_message["destination"]
#     timestamp = last_message["timestamp"]
#     probability = last_message["probability_of_delay"]
#     prediction = last_message["delay_prediction"]
#     departure_ice = last_message['dep_ice']
#     departure_rain = last_message['dep_rain']
#     departure_wind = last_message['dep_wind']
#     arrival_ice = last_message['arr_ice']
#     arrival_rain = last_message['arr_rain']
#     arrival_wind = last_message['arr_wind']
#     response = llm.invoke(
#         f"""
#         You are an expert travel assistant.
#         A customer is flying from {origin} to {destination} on {timestamp} on {airline}.
#         Based on weather conditions (departure - ice: {departure_ice}, rain: {departure_rain}, wind: {departure_wind}; arrival - ice: {arrival_ice}, rain
# : {arrival_rain}, wind: {arrival_wind}) and other factors,
#         the probability of delay is {probability:.2%}, and the prediction is that the flight will be {prediction}.
#         Provide a concise summary to the customer about their flight status.
#         Say the full airport name not the IATA code in your response
#         Also mention the weather conditions at departure and arrival airports. for example there is rain at departure but no ice or wind
#         """
#     )
#     return {"messages": [{"role": "assistant", "content": response.content}]}
    



