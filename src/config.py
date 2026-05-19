from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

COUNT_COLUMNS = [
    "site_id",
    "direction_id",
    "type",
    "start_time",
    "end_time",
    "count"
]

SITE_COLUMNS = [
    "site_id",
    "site_uuid",
    "longitude",
    "latitude",
    "municipality",
    "street"
]

DIRECTION_COLUMNS = [
    "direction_id",
    "direction_name"
]

START_DATE = "2019-08-01"
END_DATE = "2026-05-31"
WEATHER_LATITUDE = 51.0
WEATHER_LONGITUDE = 4.0
RANDOM_STATE = 42
TEST_START_YEAR = 2025
