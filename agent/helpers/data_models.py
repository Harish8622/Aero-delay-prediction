from pydantic import BaseModel, Field, field_validator
from typing import Literal, Annotated
from datetime import datetime
import pandas as pd

airports = pd.read_csv("data/raw/airports.csv")

airlines = [
    "American Airlines",
    "Delta Air Lines",
    "United Airlines",
    "Southwest Airlines",
    "Alaska Airlines",
    "JetBlue Airways",
    "Spirit Airlines",
    "Frontier Airlines",
    "Hawaiian Airlines",
    "Allegiant Air",
    "Sun Country Airlines",
    "Copa Airlines",
    "Aeromexico",
    "Air Canada",
    "WestJet",
    "British Airways",
    "Lufthansa",
    "Emirates",
    "Qatar Airways",
    "Singapore Airlines",
]

# ---- Shared constrained types ----
DayOfWeek = Annotated[int, Field(ge=0, le=6)]  # Monday=0..Sunday=6
MonthInt = Annotated[int, Field(ge=1, le=12)]  # Jan=1..Dec=12
HourOfDay = Annotated[int, Field(ge=0, le=23)]  # 0..23
BinaryFlag = Literal[0, 1]


class RouteValidators(BaseModel):
    """
    define reusable validators for route-related fields
    child classes can inherit from this and if a field tht is validated
    is present, the validator will be applied

    check_field is set to False so if child class does not have the field,
    the validator is simply skipped
    """

    @field_validator("airline", check_fields=False)
    @classmethod
    def airline_supported(cls, v: str) -> str:
        if v not in airlines:
            raise ValueError(f"Unsupported airline: {v}")
        return v

    @field_validator("origin", check_fields=False)
    @classmethod
    def origin_supported(cls, v: str) -> str:
        if v not in airports["IATA"].values:
            raise ValueError(f"Unsupported origin airport: {v}")
        return v

    @field_validator("destination", check_fields=False)
    @classmethod
    def destination_supported(cls, v: str) -> str:
        if v not in airports["IATA"].values:
            raise ValueError(f"Unsupported destination airport: {v}")
        return v


class FlightParams(RouteValidators):
    """
    used to enforce llm output structure for flight route extraction
    """

    airline: str = Field(..., description="Selected airline")
    origin: str = Field(..., description="Departure airport IATA code")
    destination: str = Field(..., description="Arrival airport IATA code")
    timestamp: datetime = Field(..., description="Scheduled departure timestamp")


class DistanceParams(RouteValidators):
    """
    used to validate scheme for distance calculation function call
    """

    origin: str = Field(..., description="Departure airport IATA code")
    destination: str = Field(..., description="Arrival airport IATA code")


class TemporalParams(BaseModel):
    """
    used to validate scheme for temporal feature extraction function call
    """

    timestamp: datetime = Field(..., description="Scheduled departure timestamp")


class WeatherParams(RouteValidators):
    """
    used to validate scheme for weather feature extraction function call
    """

    origin: str = Field(..., description="Departure airport IATA code")
    destination: str = Field(..., description="Arrival airport IATA code")
    timestamp: datetime = Field(..., description="Scheduled departure timestamp")


class FinalPrediction(RouteValidators):
    """
    Used to validation scheme for final prediction function call
    """

    airline: str = Field(..., description="Selected airline")
    origin: str = Field(..., description="Departure airport IATA code")
    destination: str = Field(..., description="Arrival airport IATA code")
    distance: float = Field(
        ..., description="Distance between origin and destination in miles"
    )
    day_of_week: DayOfWeek = Field(..., description="Day of the week, Monday=0")
    month: MonthInt = Field(..., description="Month as an integer, January=1")
    hour_of_day: HourOfDay = Field(..., description="Hour of the day in 24-hour format")
    is_bank_holiday: BinaryFlag = Field(
        ..., description="1 if the flight date is a bank holiday, else 0"
    )
    dep_rain: BinaryFlag = Field(..., description="1 if rain at departure, else 0")
    dep_ice: BinaryFlag = Field(..., description="1 if ice at departure, else 0")
    dep_wind: BinaryFlag = Field(..., description="1 if high wind at departure, else 0")
    arr_rain: BinaryFlag = Field(..., description="1 if rain at arrival, else 0")
    arr_ice: BinaryFlag = Field(..., description="1 if ice at arrival, else 0")
    arr_wind: BinaryFlag = Field(..., description="1 if high wind at arrival, else 0")
