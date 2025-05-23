import numpy as np
from tensorflow.keras.models import load_model #type: ignore

model = load_model("flood_prediction_model.h5")
print("Model loaded successfully!")

X_test = np.load("X_test.npy")
print("X_test loaded successfully! Shape:", X_test.shape)

monsoon_intensity = float(input("Enter Monsoon Intensity (0 to 1): "))
urbanization = float(input("Enter Urbanization Level (0 to 1): "))
drainage_quality = float(input("Enter Drainage Quality (0 to 1): "))
climate_change = float(input("Enter Climate Change Impact (0 to 1): "))

sample_input = np.zeros((1, X_test.shape[1]))  

sample_input[0, 0] = monsoon_intensity
sample_input[0, 1] = urbanization
sample_input[0, 2] = drainage_quality
sample_input[0, 3] = climate_change

prediction = model.predict(sample_input)
predicted_class = (prediction >= 0.5).astype(int)

if predicted_class[0][0] == 1:
    print(" Flood Warning! High risk of flooding.")
else:
    print(" No Flood Risk. Conditions are safe.")
