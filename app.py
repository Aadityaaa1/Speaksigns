import cv2
import numpy as np
import streamlit as st
import time
from camera_input_live import camera_input_live
from ultralytics import YOLO

# Load YOLO model
model = YOLO(r"best1.pt")


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
        
    else:
        st.info("No signs detected.")
