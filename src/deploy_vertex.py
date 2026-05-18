import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from google.cloud import aiplatform

PROJECT_ID = "mda-cycling-flanders"
REGION = "europe-west1"
BUCKET_NAME = "mda-cycling-version-1"
MODEL_DISPLAY_NAME = "cycling-flanders"
ENDPOINT_DISPLAY_NAME = "cycling-flanders-endpoint"

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
LOCAL_MODEL_DIR = PROCESSED_DIR / "vertex_model"


def export_for_vertex():
    LOCAL_MODEL_DIR.mkdir(exist_ok=True)
    
    # Load our existing model
    model = joblib.load(PROCESSED_DIR / "best_model.pkl")
    
    # Save as MLflow sklearn model — produces the right structure for Vertex
    mlflow.sklearn.save_model(
        sk_model=model,
        path=str(LOCAL_MODEL_DIR),
        serialization_format="cloudpickle",
    )
    print(f"Model exported to {LOCAL_MODEL_DIR}")


def upload_to_gcs():
    from google.cloud import storage
    
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)
    
    artifact_uri = f"gs://{BUCKET_NAME}/cycling-flanders/v1"
    
    for local_file in LOCAL_MODEL_DIR.rglob("*"):
        if local_file.is_file():
            relative = local_file.relative_to(LOCAL_MODEL_DIR)
            blob_path = f"cycling-flanders/v1/{relative.as_posix()}"
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(str(local_file))
            print(f"Uploaded {relative} -> {blob_path}")
    
    return artifact_uri


def deploy_to_vertex(artifact_uri):
    aiplatform.init(project=PROJECT_ID, location=REGION)
    
    # Find a sklearn serving container that matches our version
    serving_image = (
        "europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-5:latest"
    )
    
    print(f"Uploading model to Vertex AI registry...")
    model = aiplatform.Model.upload(
        display_name=MODEL_DISPLAY_NAME,
        artifact_uri=artifact_uri,
        serving_container_image_uri=serving_image,
    )
    print(f"Model registered: {model.resource_name}")
    
    print(f"\nDeploying to endpoint (this takes 5-15 min)...")
    endpoint = model.deploy(
        machine_type="n1-standard-2",
        min_replica_count=1,
        max_replica_count=1,
        traffic_percentage=100,
        sync=True,
    )
    print(f"\nEndpoint ready: {endpoint.resource_name}")
    print(f"Endpoint ID: {endpoint.name}")
    return endpoint


if __name__ == "__main__":
    export_for_vertex()
    artifact_uri = upload_to_gcs()
    #artifact_uri = f"gs://{'mda-cycling-version-1'}/cycling-flanders/v1"

    endpoint = deploy_to_vertex(artifact_uri)
    
    # Save endpoint name for later use
    endpoint_info_path = PROCESSED_DIR / "vertex_endpoint.txt"
    endpoint_info_path.write_text(endpoint.resource_name)
    print(f"\nEndpoint info saved to {endpoint_info_path}")
    print(f"\nTo invoke: python -m src.test_vertex_endpoint")