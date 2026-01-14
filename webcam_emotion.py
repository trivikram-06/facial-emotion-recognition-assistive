import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

# ===============================
# Load trained model
# ===============================
model = load_model("emotion_model.h5")

# Emotion labels
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ===============================
# Supportive / Empathetic Messages
# ===============================
supportive_messages = {
    "Happy": "You look happy! Keep smiling 😊",
    "Sad": "It's okay to feel sad. You are not alone.",
    "Angry": "I sense anger. Let's take a deep breath together.",
    "Fear": "You seem afraid. Everything will be okay.",
    "Surprise": "You look surprised. Stay calm.",
    "Disgust": "You seem uncomfortable. It's okay to step back.",
    "Neutral": "You look calm and relaxed."
}

# ===============================
# Face detector
# ===============================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ===============================
# Emotion smoothing buffer
# ===============================
emotion_buffer = deque(maxlen=5)  # faster response

# ===============================
# Start webcam
# ===============================
cap = cv2.VideoCapture(0)

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:
        # Crop and preprocess face
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face / 255.0
        face = face.reshape(1, 48, 48, 1)

        # Predict emotion
        pred = model.predict(face, verbose=0)
        confidence = np.max(pred)
        raw_emotion = emotions[np.argmax(pred)]

        if confidence < 0.5:
            raw_emotion = "Neutral"

        # Emotion smoothing
        emotion_buffer.append(raw_emotion)
        smooth_emotion = max(
            set(emotion_buffer),
            key=emotion_buffer.count
        )

        message = supportive_messages[smooth_emotion]

        # Draw bounding box
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Emotion label
        cv2.putText(
            frame,
            smooth_emotion,
            (x, y - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        # Supportive message
        cv2.putText(
            frame,
            message,
            (x, y + h + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    cv2.imshow("Emotion Recognition with Support", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ===============================
# Cleanup
# ===============================
cap.release()
cv2.destroyAllWindows()
