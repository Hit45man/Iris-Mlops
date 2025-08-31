from flask import Flask, request, jsonify, render_template
import mlflow.pyfunc
import numpy as np

app = Flask(__name__)
try:
    model = mlflow.pyfunc.load_model("models:/IrisRFModel/Staging")
except:
    # Fallback to latest version if Staging not available
    model = mlflow.pyfunc.load_model("models:/IrisRFModel/latest")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict_web", methods=["POST"])
def predict_web():
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])
    
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)[0]
    
    species = ['setosa', 'versicolor', 'virginica'][prediction]
    return render_template("result.html", species=species)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    preds = model.predict(data["dataframe_split"])
    return jsonify(predictions=preds.tolist())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
