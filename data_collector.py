import cv2
import numpy as np
import mediapipe as mp
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(results):
    # Extract left hand (21 points * 3 coords = 63 values)
    if results.left_hand_landmarks:
        left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
        return left_hand
    elif results.right_hand_landmarks:
        right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
        return right_hand
    else:
        return np.zeros(21 * 3)

# Create data directories
for action in ['hello', 'thanks', 'yes']:
    os.makedirs(os.path.join('data', action), exist_ok=True)

# Main loop
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # process frame (rgb for mediapipe)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)
        
        # create display image (bgr for opencv)
        display_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)  # convert back for rendering

        # Draw landmarks
        mp_drawing.draw_landmarks(display_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(display_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        
        # UI instructions
        cv2.rectangle(display_image, (0,0), (640,40), (245,117,16), -1)
        cv2.putText(display_image, "Press 'h' (hello), 't' (thanks), 'y' (yes) to record", (10,30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        cv2.imshow('Data Collection', display_image)
        
        # check keys
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break

        # Check if user pressed a key to record
        action = None
        if key == ord('h'): action = 'hello'
        if key == ord('t'): action = 'thanks'
        if key == ord('y'): action = 'yes'

        if action:
            folder = os.path.join('data', action)
            file_count = len(os.listdir(folder))
            print(f"Recording for action: {action}")
            sequence = []
            for frame_num in range(30):
                ret, frame = cap.read()
                if not ret:
                    break

                # process
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image_rgb)
                
                # create display image for this frame
                display_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

                # draw landmarks
                mp_drawing.draw_landmarks(display_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(display_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                
                # visual feedback start
                cv2.putText(display_image, f"Recording {action.upper}: {frame_num}/30", (50,200), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
               
                cv2.imshow('Data Collection', display_image)
                cv2.waitKey(3)
               
                # save keypoints
                keypoints = extract_keypoints(results)

                if keypoints.max() == 0:
                    cv2.putText(display_image, "No hand detected, try again", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) 
                
                sequence.append(keypoints)

            folder = os.path.join('data', action)
            file_count = len(os.listdir(folder))
            save_path = os.path.join(folder, str(file_count))
            np.save(save_path, np.array(sequence))
            print(f"Saved {action} sequence to {save_path}")

        if key == ord('q'):
            break

        # # Press 'r' to record 30 frames
        # if cv2.waitKey(10) & 0xFF == ord('r'):
        #     print("Recording...")
        #     sequence = []
        #     for _ in range(30):
        #         ret, frame = cap.read()
        #         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #         results = holistic.process(image)
        #         keypoints = extract_keypoints(results)
        #         sequence.append(keypoints)
        #         cv2.waitKey(33)  # Approx 30 FPS

        #     np.save(os.path.join('data', 'hello', '0'), np.array(sequence))
        #     print("Recording complete!")

        # if cv2.waitKey(10) & 0xFF == ord('q'):
        #     break

cap.release()
cv2.destroyAllWindows()