import json
import os
from typing import List, Dict, Any

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _get_data_file(filename: str) -> str:
    """Locate lab data in either the submission or autograder layout."""
    candidates = (
        os.path.join(PROJECT_DIR, "raw-data", filename),
        os.path.join(PROJECT_DIR, "autograder", "raw-data", filename),
    )
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return candidates[0]

def get_flight_info(origin: str, destination: str, max_price: int = 5000000) -> List[Dict[str, Any]]:
    """
    Search for flights matching origin, destination, and budget constraint.
    """
    flight_file = _get_data_file("flight_data.json")
    if not os.path.exists(flight_file):
        return []
    
    with open(flight_file, "r", encoding="utf-8") as f:
        flights = json.load(f)
    
    results = [
        fl for fl in flights
        if fl["origin"].upper() == origin.upper()
        and fl["destination"].upper() == destination.upper()
        and fl["price_vnd"] <= max_price
    ]
    return results

def get_weather_forecast(city_code: str) -> Dict[str, Any]:
    """
    Get weather forecast and outfit recommendation for a city code (e.g. SGN, HAN, DAD).
    """
    weather_file = _get_data_file("weather_data.json")
    if not os.path.exists(weather_file):
        return {"error": "Weather data not found"}
    
    with open(weather_file, "r", encoding="utf-8") as f:
        weather_data = json.load(f)
    
    return weather_data.get(city_code.upper(), {"error": f"No data for {city_code}"})

# Tool Registry for ReAct Agent
TOOL_DEFINITIONS = [
    {
        "name": "get_flight_info",
        "description": "Tìm chuyến bay theo điểm đi, điểm đến và giá tối đa.",
        "parameters": {
            "origin": "Mã sân bay đi (VD: HAN)",
            "destination": "Mã sân bay đến (VD: SGN)",
            "max_price": "Giá vé tối đa dạng số nguyên (VND)"
        }
    },
    {
        "name": "get_weather_forecast",
        "description": "Lấy thông tin thời tiết và gợi ý trang phục theo mã sân bay/thành phố (SGN, HAN, DAD).",
        "parameters": {
            "city_code": "Mã sân bay thành phố (VD: SGN)"
        }
    }
]

TOOL_MAP = {
    "get_flight_info": get_flight_info,
    "get_weather_forecast": get_weather_forecast
}
