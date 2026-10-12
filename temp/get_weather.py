import os
import requests
from langchain_core.tools import tool


@tool
def get_weather(city: str) -> dict:
    """Get current weather for a city, such as Hyderabad,IN."""

    data = requests.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={
            "q": city,
            "appid": os.environ["OPENWEATHER_API_KEY"],
            "units": "metric",
        },
        timeout=15,
    ).json()

    return {
        "city": data["name"],
        "temperature_c": data["main"]["temp"],
        "condition": data["weather"][0]["description"],
    }