import numpy as np
import os

file_path = os.path.join('data', 'yes', '5.npy')

try:
    data = np.load(file_path)
    print(f"--- Inspecting data from {file_path} ---")
    print(f"Shape: {data.shape}")
    print(f"Max value: {data.max()}")
    print(f"Min value: {data.min()}")
    print(data)

    # check if it's all zeros
    if data.max() == 0 and data.min() == 0:
        print("Warning: Data contains all zeros.")
        print("Mediapipe might not have detected the hand landmarks properly, re-record this")
    else:
        print("Data looks good!")
        print(data[0, :5])

except FileNotFoundError:
    print(f"File {file_path} not found. Please ensure the data has been collected.")