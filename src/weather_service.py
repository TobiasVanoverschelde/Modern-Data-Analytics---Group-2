import time
import requests
import pandas as pd
from pathlib import Path


class WeatherService:

    def __init__(self):

        self.weather_dir = Path("data/weather")

        self.weather_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    # =====================================================
    # DOWNLOAD WEATHER
    # =====================================================

    def get_weather_data(

        self,

        municipality,
        latitude,
        longitude,

        start_date,
        end_date

    ):

        cache_file = (

            self.weather_dir /
            f"{municipality}.csv"

        )

        # =============================================
        # USE CACHE IF EXISTS
        # =============================================

        if cache_file.exists():

            print(
                f"Using cached weather for {municipality}"
            )

            return pd.read_csv(cache_file)

        print(
            f"Downloading weather for {municipality}"
        )

        url = (

            "https://archive-api.open-meteo.com/v1/archive"

            f"?latitude={latitude}"
            f"&longitude={longitude}"

            f"&start_date={start_date}"
            f"&end_date={end_date}"

            "&daily="
            "temperature_2m_mean,"
            "precipitation_sum,"
            "precipitation_hours,"
            "sunshine_duration,"
            "wind_speed_10m_max"

            "&timezone=Europe%2FBerlin"

        )

        # =============================================
        # API REQUEST
        # =============================================

        response = requests.get(
            url,
            timeout=60
        )

        response.raise_for_status()

        data = response.json()

        # =============================================
        # PARSE DATA
        # =============================================

        daily = data.get("daily")

        if daily is None:

            print(
                f"No daily weather returned for {municipality}"
            )

            return pd.DataFrame()

        weather_df = pd.DataFrame({

            "date": daily.get("time"),

            "temperature_2m_mean":
                daily.get("temperature_2m_mean"),

            "precipitation_sum":
                daily.get("precipitation_sum"),

            "precipitation_hours":
                daily.get("precipitation_hours"),

            "sunshine_duration":
                daily.get("sunshine_duration"),

            "wind_speed_10m_max":
                daily.get("wind_speed_10m_max")

        })

        weather_df["municipality"] = municipality

        # =============================================
        # SAVE CACHE
        # =============================================

        weather_df.to_csv(
            cache_file,
            index=False
        )

        print(
            f"Saved weather for {municipality}"
        )

        # =============================================
        # COOLDOWN
        # =============================================

        print(
            "Cooling down API for 30 seconds..."
        )

        time.sleep(30)

        return weather_df
