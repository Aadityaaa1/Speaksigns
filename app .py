import streamlit as st
from ultralytics import YOLO
import cv2
import time
import pyttsx3  # Library for text-to-speech

# Initialize the TTS engine
tts_engine = pyttsx3.init()

# Load YOLO model
model = YOLO(r"C:\Users\aadit\Downloads\streamlit-signlanguage\best1.pt")

# Speak the detected sign out loud
def speak_text(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

# Process the webcam feed and run the model with TTS every 3 seconds
def webcam_with_yolo():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("ERROR: Failed to open the webcam. Please check your camera settings.")
        return

    st.write("Webcam successfully opened. Starting live video stream...")

    # Create placeholders for the video feed and last detected sign
    video_placeholder = st.empty()
    detected_placeholder = st.empty()

    last_tts_time = 0  # Track the last time TTS was triggered

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Warning: Failed to grab frame from the webcam.")
            break

        # Run YOLO model to get detections
        results = model(frame)

        # Annotate the frame with bounding boxes (if any)
        annotated_frame = results[0].plot()  # YOLO provides a method to render boxes
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # Display the live feed with annotations
        video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

        # Check for detections
        if len(results[0].boxes) > 0:  # If any predictions exist
            # Get detected class labels
            detected_signs = [box.cls for box in results[0].boxes]
            detected_labels = [model.names[int(cls)] for cls in detected_signs]

            # Announce and display detected signs (debounced to trigger every 3 seconds)
            if detected_labels:
                detected_labels = ["I love you" if label == "iloveyou" else label for label in detected_labels]
                detected_text = ", ".join(detected_labels)
                current_time = time.time()
                if current_time - last_tts_time >= 3:
                    last_tts_time = current_time
                    speak_text(detected_text)
            else:
                detected_placeholder.write("No signs detected.")

        # Allow Streamlit to process UI updates
        time.sleep(0.1)

    cap.release()

# Streamlit app
def main():
    st.title("Sign Language Recognition ")

    # Start webcam and YOLO processing
    webcam_with_yolo()

if __name__ == "__main__":
    main()
