import os
import joblib
import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = "https://dagshub.com/sharifa-mohamed/late_delivery_prediction.mlflow"

os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
os.environ["MLFLOW_TRACKING_USERNAME"] = "sharifa-mohamed"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "6217a14c7055eaf0a3810d6f008e4ad8a66c63ec"

experiment_name = "late_delivery_prediction"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

client = MlflowClient()

BASE_DIR = "registered_models"
os.makedirs(BASE_DIR, exist_ok=True)


def parse_requirements(path):
    reqs = set()

    if not path or not os.path.exists(path):
        return reqs

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                reqs.add(line)

    return reqs


def get_latest_model_version(model_name):
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        return None
    return max(versions, key=lambda v: int(v.version))


def download_latest(model_name, global_reqs):
    latest = get_latest_model_version(model_name)

    if not latest:
        print(f"No versions found for {model_name}")
        return

    version = latest.version
    source = latest.source
    stage = latest.current_stage or "no_stage"

    target_dir = os.path.join(BASE_DIR, model_name, f"version_{version}_{stage}")
    os.makedirs(target_dir, exist_ok=True)

    print(f"\nSaving {model_name} version {version}")

    try:
        mlflow.artifacts.download_artifacts(
            artifact_uri=source,
            dst_path=os.path.join(target_dir, "model_artifacts")
        )
    except Exception as e:
        print(f"Artifact download failed for {model_name}: {e}")

    try:
        model_uri = f"models:/{model_name}/{version}"
        deps_path = mlflow.pyfunc.get_model_dependencies(model_uri)

        model_reqs = parse_requirements(deps_path)
        global_reqs.update(model_reqs)

        if model_reqs:
            with open(os.path.join(target_dir, "requirements.txt"), "w") as f:
                for r in sorted(model_reqs):
                    f.write(r + "\n")

    except Exception as e:
        print(f"Dependency extraction failed for {model_name}: {e}")
        
        
def download_data_preparing_pipeline(target_dir):
    
    # Locate Latest Preprocessing Run Dynamically
    
    preprocessing_exp = mlflow.get_experiment_by_name(experiment_name)

    if preprocessing_exp is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    # Search specifically for the run named 'data_preparation_pipeline'
    runs = mlflow.search_runs(
        experiment_ids=[preprocessing_exp.experiment_id],
        filter_string="tags.mlflow.runName = 'data_preparation_pipeline'",
        order_by=["attributes.start_time DESC"],
        max_results=1
    )

    if runs.empty:
        raise FileNotFoundError(f"No 'data_preparation_pipeline' runs found for experiment: {experiment_name}")

    latest_run_id = runs.iloc[0]["run_id"]

    print(f"\nConnected To Run ID: {latest_run_id}")
    
    pipeline_path = mlflow.artifacts.download_artifacts(
        run_id=latest_run_id,
        artifact_path="late_delivery_pipeline.pkl",
        dst_path=os.path.join(target_dir, "data_preparation_pipeline")
    )

    print(f"Pipeline downloaded to: {pipeline_path}")
    

def main():
    
    print("Downloading data preparation pipeline...")
       
    download_data_preparing_pipeline(BASE_DIR)
    
        
    print("Fetching registered models...")

    models = client.search_registered_models()

    global_requirements = set()

    for m in models:
        download_latest(m.name, global_requirements)

    # Save unified requirements
    unified_path = os.path.join(BASE_DIR, "unified_requirements.txt")

    with open(unified_path, "w") as f:
        for req in sorted(global_requirements):
            f.write(req + "\n")

    
    print(f"\nDone. Unified requirements saved to {unified_path}")
    print("All latest models saved in registered_models folder")


if __name__ == "__main__":
    main()