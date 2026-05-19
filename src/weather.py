import pandas as pd
import requests


def fetch_open_meteo(lat, lon, start, end,
                     variables=("temperature_2m", "precipitation",
                                "wind_speed_10m", "cloud_cover")):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(variables),
        "timezone": "Europe/Brussels",
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    payload = r.json()["hourly"]
    weather = pd.DataFrame(payload)
    weather["time"] = pd.to_datetime(weather["time"])
    return weather.rename(columns={"time": "timestamp"})