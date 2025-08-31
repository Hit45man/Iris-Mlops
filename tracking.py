import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Use local file-based tracking for CI/CD
mlflow.set_tracking_uri("file:./mlruns")

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
