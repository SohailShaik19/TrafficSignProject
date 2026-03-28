import cv2
import numpy as np
from tensorflow.keras.models import load_model

# --- Load your trained model ---
model = load_model(r"D:\myPython\traffic_model.h5")  # absolute path

# --- Manual list of 43 traffic sign names ---
sign_names = [
    "Speed limit 20 km/h", "Speed limit 30 km/h", "Speed limit 50 km/h",
    "Speed limit 60 km/h", "Speed limit 70 km/h", "Speed limit 80 km/h",
    "End of speed limit 80 km/h", "Speed limit 100 km/h", "Speed limit 120 km/h",
    "No passing", "No passing for vehicles over 3.5 tons",
    "Right-of-way at next intersection", "Priority road", "Yield", "Stop",
    "No vehicles", "Vehicles over 3.5 tons prohibited", "No entry",
    "General caution", "Dangerous curve left", "Dangerous curve right",
    "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right",
    "Road work", "Traffic signals", "Pedestrians", "Children crossing",
    "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing",
    "End speed + passing limits", "Turn right ahead", "Turn left ahead",
    "Ahead only", "Go straight or right", "Go straight or left",
    "Keep right", "Keep left", "Roundabout mandatory", "End of no passing",
    "End no passing for vehicles over 3.5 tons"
]

# --- Load and preprocess your image ---
image = cv2.imread(r"C:\Users\Sohai\Downloads\archive\Test\10268.png")        # replace with your image path
image = cv2.resize(image, (32, 32))
image = image / 255.0
image = np.reshape(image, (1, 32, 32, 3))

# --- Predict ---
prediction = model.predict(image)
class_id = np.argmax(prediction)

# --- Output ---
print("Predicted Traffic Sign Class ID:", class_id)
print("Predicted Traffic Sign Name:", sign_names[class_id])