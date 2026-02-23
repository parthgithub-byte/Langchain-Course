from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient

load_dotenv()

tavily = TavilyClient()

@tool
def search(query: str) -> str:
    """
    Tool that searches over the internet.
    Args:
    query: The query to search for
    Returns:
    The result
    """
    print(f"Seaching for {query}")  # buffer display text
    return tavily.search(query=query)


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
tools = [search]
agent = create_agent(model=llm, tools=tools)


def main():
    results = agent.invoke(
        {"messages": HumanMessage(content="What is the weather of Tokyo?")}
    )
    print(results)  # An llm response result contain various components, we mostly want the content attribute only


if __name__ == "__main__":
    main()


# The LangSmith tracer:
# Input:

# Human:
# What is the weather of Tokyo?

# AI:
# search (i.e. sent to search tool)
# query: weather in Tokyo

# Tool:
# search (i.e. came from the search tool)
# {"query": "weather in Tokyo", "follow_up_questions": null, "answer": null, "images": [], "results": [{"title": "Weather in Tokyo", "url": "https://www.weatherapi.com/", "content": "{'location': {'name': 'Tokyo', 'region': 'Tokyo', 'country': 'Japan', 'lat': 35.6895, 'lon': 139.6917, 'tz_id': 'Asia/Tokyo', 'localtime_epoch': 1771851111, 'localtime': '2026-02-23 21:51'}, 'current': {'last_updated_epoch': 1771850700, 'last_updated': '2026-02-23 21:45', 'temp_c': 12.3, 'temp_f': 54.1, 'is_day': 0, 'condition': {'text': 'Partly Cloudy', 'icon': '//cdn.weatherapi.com/weather/64x64/night/116.png', 'code': 1003}, 'wind_mph': 13.6, 'wind_kph': 22.0, 'wind_degree': 51, 'wind_dir': 'NE', 'pressure_mb': 1019.0, 'pressure_in': 30.09, 'precip_mm': 0.0, 'precip_in': 0.0, 'humidity': 88, 'cloud': 25, 'feelslike_c': 10.1, 'feelslike_f': 50.2, 'windchill_c': 11.4, 'windchill_f': 52.5, 'heatindex_c': 13.3, 'heatindex_f': 56.0, 'dewpoint_c': 9.7, 'dewpoint_f': 49.4, 'vis_km': 10.0, 'vis_miles': 6.0, 'uv': 0.0, 'gust_mph': 18.9, 'gust_kph': 30.3}}", "score": 0.9999659, "raw_content": null}, {"url": "https://www.weather25.com/asia/japan/tokyo?page=month&month=February", "title": "Tokyo weather in February 2026 | Tokyo 14 day weather", "content": "Location was added to My Locations. Location was removed from My Locations. # Tokyo weather in February 2026. You can expect about **3 to 8 days of rain** in Tokyo during the month of February. It’s important to get out your snow boots and warm mittens to stay warm while you explore Tokyo. Historic average weather for February. | 1  11° /4° | 2  10° /4° | 3  10° /4° | 4  9° /4° | 5  9° /5° | 6  9° /4° | 7  9° /3° |. | 8  8° /4° | 9  9° /3° | 10  11° /2° | 11  9° /5° | 12  9° /5° | 13  11° /6° | 14  13° /6° |. | January | **9°** / 2° | 5 | 22 | 4 | 90 mm | Bad | Tokyo in January |. | March | **13°** / 6° | 8 | 22 | 1 | 192 mm | Bad | Tokyo in March |. Click on hotel for more details.", "score": 0.9999517, "raw_content": null}, {"url": "https://www.agatetravel.com/japan/tokyo/weather-in-february.html", "title": "Tokyo Weather in February 2026, Cold, Mostly Clear Climate", "content": "# Tokyo Weather in February. The weather of Tokyo in February is still a bit chilly, but the winter feeling is gradually fading overall. ## How much rain does Tokyo get in February? Tokyo gets a total of 56 mm (2.2 in) of precipitation in February from around 6 days. Tokyo receives an average of about 6 hours of sunshine per day in February. ## Is February a good time to visit Tokyo? For tourists who prefer fewer crowds and lower prices, February is a good time to visit Tokyo, as it belongs to the low travel season. The climate of Tokyo in February is only a little cold, and many sunny days serve as good conditions for outdoor traveling. ## What to wear in Tokyo in February? Although it’s not a high season for appreciating cherry blossoms in Tokyo, early sakura can be seen in February, which begins to bloom early the month, and reaches its high season in mid to late February.", "score": 0.99993026, "raw_content": null}, {"url": "https://world-weather.info/forecast/japan/tokyo/february-2026/", "title": "Weather in Tokyo in February 2026 (Tōkyō-to)", "content": "Weather in Tokyo in February 2026. Tokyo Weather Forecast for February 2026 ... Monday, 23 February. Day. +63°. 11. 29.8. 62%. +57°. 06:18 AM. 05:29 PM. Waxing", "score": 0.99987173, "raw_content": null}, {"url": "https://www.japanhighlights.com/japan/weather-in-february", "title": "Japan Weather in February 2026", "content": "# Japan Weather in February. **Winter continues in February in Japan**. Central Japan (Tokyo, Kyoto, and Osaka) stays cold and quiet, while spring shows signs of coming in the warm south (Okinawa). **Along with the benefits of lower crowding**, February is a great time to discover Japan for all travelers: thrilling winter activities, relaxing hot springs to bathe in, and stunning early cherry and plum blossoms. February is cold in Japan. You can expect heavy snow in February on Hokkaido, Japan's northern island**.** The snowy landscape and winter sports appeal to travelers, especially skiing lovers, while the festive atmosphere lights up the city of Sapporo. Just 1½ hours' drive from Tokyo, Hakone is a perfect winter getaway, known for Japan's best hot springs. Okinawa is one of Japan's warmest places in February, averaging highs of 19°C (67°F). Being one of the best times for skiing, **February sees most Japan travelers in Hokkaido.** If you're going for winter sports, **you can expect higher prices and more crowds than usual**.", "score": 0.99971753, "raw_content": null}], "response_time": 1.4, "request_id": "IAMHIDINGTHIS"}

# AI:
# The weather in Tokyo, Japan is currently 12.3°C (54.1°F) and partly cloudy. The wind is blowing from the NE at 22.0 kph (13.6 mph), and the humidity is 88%. It feels like 10.1°C (50.2°F). The local time is 2026-02-23 21:51.
                                                                                                                                                                                   