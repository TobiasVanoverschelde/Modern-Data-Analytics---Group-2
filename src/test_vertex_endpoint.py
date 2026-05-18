import sys
from pathlib import Path

import numpy as np
import pandas as pd
from google.cloud import aiplatform

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.modeling import ALL_FEATURES

PROJECT_ID = "mda-cycling-version-1"
REGION = "europe-west1"


def predict():
    aiplatform.init(project=PROJECT_ID, location=REGION)

    endpoint_name = (
        Path(__file__).parent.parent / "data" / "processed" / "vertex_endpoint.txt"
    ).read_text().strip()
    endpoint = aiplatform.Endpoint(endpoint_name)

    row = {
        "temperature_2m": 18.0,
        "precipitation": 0.0,
        "wind_speed_10m": 10.0,
        "cloud_cover": 40.0,
        "day_of_week_sin": float(np.sin(2 * np.pi * 2 / 7)),
        "day_of_week_cos": float(np.cos(2 * np.pi * 2 / 7)),
        "month_sin": float(np.sin(2 * np.pi * 6 / 12)),
        "month_cos": float(np.cos(2 * np.pi * 6 / 12)),
        "gemeente": "Leuven",
        "covid_period": "post_covid",
        "is_weekend": False,
        "is_holiday": False,
    }

    # Force column order matching zoals the trained pipeline expects
    df = pd.DataFrame([row])[ALL_FEATURES]
    instances = df.to_dict(orient="records")

    print(f"Sending instance:\n{instances}")
    response = endpoint.predict(instances=instances)
    print(f"\nPredictions: {response.predictions}")


if __name__ == "__main__":
    predict()