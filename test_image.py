import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("emotion_model.h5")

emotions = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

# Load image
img = cv2.imread("test.jpg")  # add a face image in project folder
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face = cv2.resize(gray, (48,48))
face = face / 255.0
face = face.reshape(1,48,48,1)

# Predict
pred = model.predict(face)
emotion = emotions[np.argmax(pred)]

print("Predicted Emotion:", emotion)
