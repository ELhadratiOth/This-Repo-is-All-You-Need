# from langchain_community.tools import DuckDuckGoSearchResults

# web_search = DuckDuckGoSearchResults()

import os
import requests


def current_weather(city:str) -> str:
    """
    Retrieves the current weather conditions for a given city.

    Returns:
        str: A string describing the current weather in the given city.
    """
    api_key = os.getenv("WEATHER_KEY")  
    url = f"http://api.openweathermap.org/data/2.5/weather?q=Al+Hoceima,MA&appid={api_key}&units=metric"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        temp = data["main"]["temp"]
        conditions = data["weather"][0]["description"]
        wind_speed = data["wind"]["speed"]
        humidity = data["main"]["humidity"]

        weather_report = (
            f"Current weather in {city} :\n"
            f"Temperature: {temp}°C\n"
            f"Conditions: {conditions.capitalize()}\n"
            f"Wind Speed: {wind_speed} m/s\n"
            f"Humidity: {humidity}%"
        )
        return weather_report

    except requests.RequestException as e:
        return f"Error fetching weather data: {str(e)}"
    
from langchain.tools import Tool

current_weather_tool = Tool(
    name="current_weather",
    func=current_weather,
    description="Useful for getting the current weather in a given city. The input to this tool should be a string, representing the name of the city. For example, 'Al Hoceima' or 'New York'.", 
  )  
  
    
tools = [
    # web_search,
    current_weather_tool,
]