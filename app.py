from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model
model = load_model("flood_prediction_model.h5")

@app.route("/")
def home():
    return "Flood Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get data from POST request (JSON format)
        data = request.json
        monsoon_intensity = float(data["monsoon_intensity"])
        urbanization = float(data["urbanization"])
        drainage_quality = float(data["drainage_quality"])
        climate_change = float(data["climate_change"])

        # Create input array
        sample_input = np.array([[monsoon_intensity, urbanization, drainage_quality, climate_change]])

        # Make prediction
        prediction = model.predict(sample_input)
        predicted_class = int((prediction >= 0.5)[0][0])

        # Return result
        return jsonify({"flood_risk": "High" if predicted_class == 1 else "Low"})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
