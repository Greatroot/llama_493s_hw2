import os
import json

import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt

class PileDataSet(Dataset):
    """Pile dataset."""

    def __init__(self, jsonl_file, transform=None):
        """
        Arguments:
            jsonl_file (string): Path to the .jsonl file with training samples
                                 from the Pile Dataset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_src = jsonl_file
        self.data = json.load(jsonl_file)
        self.transform = transform

    def __len__(self):
        return 

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        

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

if __name__ == "__main__":
    with open('./data/tiny_test_set.jsons', 'r') as file:
        data = json.load(file)
        print(f"type: {type(data)}")
        print(f"data[0] = {data[0]}")
        print(f"data[1] = {data[1]}")
    # with jsonlines.open('./data/subset_data.jsonl') as data_reader:
    #     # data_reader = jsonlines.Reader('./data/subset_data.jsonl')
    #     first = data_reader.read()
    #     second = data_reader.read()
    #     print(f"first = {first}")
    #     print(f"second = {second}")

    # print(f"did anything happen?")
