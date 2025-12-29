# 🛠️ SignLingo Development Log

## Phase 1: The Sensor
**Goal:** Capture consistent human gesture data.
* **Challenge:** "All-Zeros" Bug
    * *Issue:* MediaPipe distinguishes between Left and Right hands. My original script only looked for `left_hand_landmarks`. When I signed with my right hand, the model received empty zero-vectors.
    * *Fix:* Implemented a check: `if Left else Right else Zeros`.
* **Challenge:** Data Visualization
    * *Issue:* I couldn't tell if the system was actually recording, leading to bad data start/stop points.
    * *Fix:* Added real-time visual feedback ("RECORDING: 1/30") directly onto the OpenCV frame buffer.

## Phase 2: The Representation
**Goal:** Make the data robust to position changes.
* **Challenge:** Translation Invariance
    * *Theory:* A "Hello" wave in the top-left corner has different $(x,y)$ coordinates than a wave in the center. A raw LSTM would fail to generalize.
    * *Solution:* **wrist-relative normalization**.
    $$P'_{i} = P_{i} - P_{wrist}$$
    * This forces the wrist to always be at $(0,0,0)$, making the gesture identical regardless of where the user stands.

## Phase 3: The Model
**Goal:** Recognize dynamic gestures over time.
* **Architecture:** 2-Layer LSTM (Long Short-Term Memory).
* **Input:** $(Batch, 30, 63)$. 30 frames of 21 landmarks $\times$ 3 coords.
* **Training Results:**
    * Epoch 1 Loss: 1.09 (Random Guessing).
    * Epoch 50 Loss: 0.003 (Convergence).
    * *Insight:* The loss dropped significantly once the "All-Zeros" data was purged.

## Phase 4: The Interaction
**Goal:** A gamified tutor loop.
* **Challenge:** The "Flickering" Prediction.
    * *Issue:* The model would output probabilities like `[0.4, 0.5, 0.1]` causing the label to flicker.
    * *Fix:* Implemented a **Confidence Threshold** (>0.8) and a **Cooldown Buffer** (1.5s) to smooth the user experience.