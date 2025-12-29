import torch
from torch.utils.data import Dataset
import numpy as np
import os

class SignLanguageDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.sequences = []
        self.labels = []
        self.label_map = {'hello': 0, 'thanks': 1, 'yes': 2}

        for label_name, label_idx in self.label_map.items():
            path = os.path.join(root_dir, label_name)
            if not os.path.exists(path):
                continue

            for file in os.listdir(path):
                if file.endswith('.npy'):
                    seq = np.load(os.path.join(path, file))
                    # normalization: subtract wrist (point 0) from all points
                    # reshape to (30, 21, 3) -> wrist is at [:, 0, :]
                    seq_reshaped = seq.reshape(30, 21, 3)

                    # get write coordinates (frame 0 of every frame)
                    wrists = seq_reshaped[:, 0, :].reshape(30, 1, 3)

                    # subtract wrist coordinates from all points
                    seq_normalized = seq_reshaped - wrists

                    # flatten back to (30, 63)
                    seq_flat = seq_normalized.reshape(30, 63)
                    
                    self.sequences.append(seq_flat)
                    self.labels.append(label_idx)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.tensor(self.labels[idx])