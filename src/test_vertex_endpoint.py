from pathlib import Path

from google.cloud import aiplatform

PROJECT_ID = "mda-cycling-version-1"   # match deploy_vertex.py
REGION = "europe-west1"


def predict():
    aiplatform.init(project=PROJECT_ID, location=REGION)
    
    endpoint_name = (
        Path(__file__).parent.parent / "data" / "processed" / "vertex_endpoint.txt"
    ).read_text().strip()
    endpoint = aiplatform.Endpoint(endpoint_name)
    
    # Build an example single-row instance matching our feature schema
    instance = {
        "temperature_2m": 18.0,
        "precipitation": 0.0,
        "wind_speed_10m": 10.0,
        "cloud_cover": 40.0,
        "day_of_week": 2,
        "month": 6,
        "gemeente": "Leuven",
        "covid_period": "post_covid",
        "is_weekend": False,
        "is_holiday": False,
    }
    
    response = endpoint.predict(instances=[instance])
    print("Predictions:", response.predictions)


if __name__ == "__main__":
    predict()