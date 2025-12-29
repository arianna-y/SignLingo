import mediapipe as mp
import os

print(f"I am importing mediapipe from: {mp.__file__}")

# Check if 'solutions' exists
try:
    print(mp.solutions)
    print("Success! Solutions found.")
except AttributeError:
    print("FAILURE: 'solutions' not found.")