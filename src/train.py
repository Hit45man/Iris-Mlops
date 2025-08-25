import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from preprocess import load_and_split

mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment("Iris_Ml_Project")

def train_model():
    X_train, X_test, y_train, y_test = load_and_split()

    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)

        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

        # ✅ Ensure 'models/' directory exists before saving
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

        print(f"Model saved to models/model.pkl with accuracy: {acc:.4f}")

if __name__ == "__main__":
    train_model()
