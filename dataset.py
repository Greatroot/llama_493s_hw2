import os
import json

import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt

class PileDataset(Dataset):
    """Pile dataset."""

    def __init__(self, jsonl_file):
        """
        Arguments:
            jsonl_file (string): Path to the .jsonl file with training samples
                                 from the Pile Dataset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_src = jsonl_file
        self.data = json.load(jsonl_file)

    def __len__(self):
        return 

    def __getitem__(self, idx):
        
        # Read the next line to get the next .jsonl object

        # img_name = os.path.join(self.root_dir,
        #                         self.landmarks_frame.iloc[idx, 0])
        # image = io.imread(img_name)
        # landmarks = self.landmarks_frame.iloc[idx, 1:]
        # landmarks = np.array([landmarks])
        # landmarks = landmarks.astype('float').reshape(-1, 2)
        # sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample
