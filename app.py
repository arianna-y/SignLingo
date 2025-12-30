import cv2
import numpy as np
import mediapipe as mp
import torch
import random
import streamlit as st
from model import SignLSTM

# Setup 
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


# Load model
model = SignLSTM()
try:
    model.load_state_dict(torch.load('sign_language_model.pth'))
    print("Model loaded successfully!")
except:
    print("Model file not found. Please train the model first.")
    exit()
model.eval()

actions = ['hello', 'thanks', 'yes']
target_word = random.choice(actions)
score = 0

# Buffer to hold last 30 frames
sequence_buffer = []

# Function to extract keypoints
def extract_keypoints(results):
    if results.left_hand_landmarks:
        return np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
    elif results.right_hand_landmarks:
        return np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
    return np.zeros(21 * 3)

def normalize_sequence(sequence_list):
    seq_arr = np.array(sequence_list)
    seq_reshaped = seq_arr.reshape(30, 21, 3)
    wrists = seq_reshaped[:, 0, :].reshape(30, 1, 3)
    seq_norm = seq_reshaped - wrists
    return seq_norm.reshape(30, 63)

# Main loop
# cap = cv2.VideoCapture(0)
use_webcam = st.checkbox("Enable Webcam")

if use_webcam:
    cap = cv2.VideoCapture(0)
else:
    st.info("Webcam is disabled. Please enable it to use the sign language learning feature.")

cooldown = 0
feedback_text = ""

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)  # convert back for rendering

        # draw landmarks
        mp_drawing.draw_landmarks(image_bgr, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image_bgr, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        if cooldown > 0:
            cooldown -= 1
            cv2.rectangle(image_bgr, (0, 0), (640, 60), (0, 255, 0), -1)
            cv2.putText(image_bgr, f"Correct! (+1)", (200, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('Sign Language Learning', image_bgr)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            continue

        # processing
        keypoints = extract_keypoints(results)
        sequence_buffer.append(keypoints)
        sequence_buffer = sequence_buffer[-30:]

        prediction_text = "..."

        if len(sequence_buffer) == 30:
            # normalize live data
            norm_seq = normalize_sequence(sequence_buffer)
            input_tensor = torch.FloatTensor([norm_seq])  # shape (1, 30, 63)

            with torch.no_grad():
                res = model(input_tensor)
                prob = torch.softmax(res, dim=1)
                prob_np = prob.detach().cpu().numpy()[0]

                best_class_idx = np.argmax(prob_np)
                confidence = prob_np[best_class_idx]

                # conf, idx = torch.max(prob, 1)

            if confidence > 0.8:
                predicted_word = actions[best_class_idx]
                prediction_text = f"{predicted_word} ({confidence*100:.1f}%)"

                # score logic
                if predicted_word.lower() == target_word.lower():
                    score += 1
                    # target_word = random.choice(actions)
                    sequence_buffer = []
                    possible = list(actions)
                    if target_word in possible:
                        possible.remove(target_word)
                    target_word = random.choice(possible)

        # UI rendering
        cv2.rectangle(image_bgr, (0,0), (640, 60), (245, 117, 16), -1)
        cv2.putText(image_bgr, f"Please Sign: {target_word.upper()}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.putText(image_bgr, f"Prediction: {prediction_text} | Score: {score}", (20, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Sign Language Learning', image_bgr)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
