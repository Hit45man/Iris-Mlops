import os
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from preprocess import load_and_split

mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment("Iris_Ml_Project")

def train_model():
    X_train, X_test, y_train, y_test = load_and_split()

    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)

        # Log multiple parameters and metrics
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("train_accuracy", model.score(X_train, y_train))
        
        # Create and log confusion matrix plot
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('confusion_matrix.png')
        mlflow.log_artifact('confusion_matrix.png')
        plt.close()
        
        mlflow.sklearn.log_model(model, "model")

        # ✅ Ensure 'models/' directory exists before saving.
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

        print(f"Model saved to models/model.pkl with accuracy: {acc:.4f}")

if __name__ == "__main__":
    train_model()
