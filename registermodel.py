import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()

try:
    # Get the latest run from the iris_experiment
    experiment = mlflow.get_experiment_by_name("iris_experiment")
    runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"], max_results=1)
    run_id = runs[0].info.run_id
    model_uri = f"runs:/{run_id}/model"
    
    model_details = mlflow.register_model(model_uri, "IrisRFModel")
    print(f"Model registered with version: {model_details.version}")
    
    client.transition_model_version_stage(
        name="IrisRFModel",
        version=model_details.version,
        stage="Staging"
    )
    print("Model transitioned to Staging")
    
except Exception as e:
    print(f"Error: {e}")
    print(f"Using run_id: {run_id if 'run_id' in locals() else 'Not found'}")
