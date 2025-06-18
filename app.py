import cv2
import numpy as np
import streamlit as st
import pyttsx3
import time
from camera_input_live import camera_input_live
from ultralytics import YOLO

# Load YOLO model
model = YOLO(r"best1.pt")

# Initialize TTS
tts_engine = pyttsx3.init()
last_tts_time = 0

# UI
st.title("Live Sign Language Detection")
st.markdown("#### Show a hand sign in front of your webcam")

# Webcam input
image = camera_input_live()

if image is not None:
    # Show the raw frame
    st.image(image)

    # Convert image to OpenCV format
    bytes_data = image.getvalue()
    frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # Run YOLO detection
    results = model(frame)

    # Process detections
    if results[0].boxes:
        detected_signs = [model.names[int(box.cls[0])] for box in results[0].boxes]
        detected_signs = ["I love you" if label == "iloveyou" else label for label in detected_signs]
        label_text = ", ".join(detected_signs)

        # Display the detected sign
        st.success(f"Detected: {label_text}")

        # Speak only once every 3 seconds
        current_time = time.time()
        if current_time - last_tts_time >= 3:
            last_tts_time = current_time
            tts_engine.say(label_text)
            tts_engine.runAndWait()
    else:
        st.info("No signs detected.")
