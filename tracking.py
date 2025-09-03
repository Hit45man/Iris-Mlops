import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from mlflow.tracking import MlflowClient
import os

# Set tracking uri to absolute path in current directory 
tracking_uri = f"file://{os.path.abspath('./mlruns')}"
mlflow.set_tracking_uri(tracking_uri)
print(f"Mlflow tracking URI: {tracking_uri}")
client = MlflowClient()
print(f"Current working directory:{os.getcwd()}")
print(f"MLruns directory exists:{os.path.exists('./mlruns')}")
if not os.path.exists('./mlruns'):
    print("Creating mlruns directory")
    os.makedirs('./mlruns',exist_ok=True)

mlflow.set_experiment("iris_experiment")

with mlflow.start_run(run_name="rf_run3"):
    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    model = RandomForestClassifier(n_estimators=50)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, artifact_path="model")
