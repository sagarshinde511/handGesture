import streamlit as st
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import tensorflow as tf
import gtts
import os
import threading
import playsound
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load trained gesture model
model = tf.keras.models.load_model("gesture_model1.keras")  # Ensure your model file is present

# Get class labels (A-Z)
gesture_labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Initialize Hand Detector
detector = HandDetector(maxHands=1)

# Function to convert text to speech without blocking

def speak_text(text):
    def play_audio():
        tts = gtts.gTTS(text)
        tts.save("output.mp3")
        try:
            playsound.playsound("output.mp3", block=False)
        except Exception as e:
            print("Error playing sound:", e)
        os.remove("output.mp3")
    threading.Thread(target=play_audio, daemon=True).start()

# Streamlit UI
st.title("Hand Gesture Recognition with Speech Output")
st.write("Show a hand gesture, and the system will recognize it and speak the detected letter.")

# Video processing class
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.previous_prediction = None  # Store last prediction to avoid repeating speech
        self.predicted_label = ""  # Store predicted label

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # Mirror effect
        hands, _ = detector.findHands(img, draw=True, flipType=True)  # Detect hand

        if hands:
            hand = hands[0]  # Get first detected hand
            x, y, w, h = hand['bbox']  # Bounding box of the detected hand

            # Ensure bounding box does not exceed frame dimensions
            x, y = max(0, x), max(0, y)
            w, h = min(img.shape[1] - x, w), min(img.shape[0] - y, h)

            # Crop and preprocess the hand image
            hand_img = img[y:y+h, x:x+w]
            hand_img = cv2.resize(hand_img, (224, 224))
            hand_img = np.expand_dims(hand_img, axis=0)  # Add batch dimension
            hand_img = hand_img / 255.0  # Normalize to [0,1]

            # Predict gesture
            prediction = model.predict(hand_img)
            self.predicted_label = gesture_labels[np.argmax(prediction)]

            # Speak only if the detected alphabet changes
            if self.predicted_label != self.previous_prediction:
                speak_text(self.predicted_label)
                self.previous_prediction = self.predicted_label

            # Display detected text on the frame
            cv2.putText(img, f"Detected: {self.predicted_label}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        return img

# Start webcam streamer
webrtc_streamer(key="gesture-detection", video_processor_factory=VideoProcessor)
