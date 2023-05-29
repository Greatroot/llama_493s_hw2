import os

import torch
from torch.utils.data import Dataset, IterableDataset

import numpy as np
import matplotlib.pyplot as plt
import jsonlines

# TODO: Use for streaming data or remove it
# def count_json_objects(filename):
#     """
#         Counts the number of json objects in a jsonl file without
#         loading in the entire json object into memory (i.e.)
#     """
#     count = 0
#     with jsonlines.open(filename) as reader:
#         for _ in reader:
#             count += 1
#     return count

class PileIterableDataset(IterableDataset):
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
        self.data_reader = jsonlines.Reader(jsonl_file)
        self.transform = transform  # TODO: remove this if not needed.

    def __iter__(self):
        sample_text = ""
        with jsonlines.open('./data/subset_data.jsonl') as data_reader:
            next_sample = self.data_reader.read()
            sample_text = next_sample["text"]
            
        return sample_text

# TODO: Remove this
# if __name__ == "__main__":
#     sample_text = ""
#     with jsonlines.open('./data/subset_data.jsonl') as data_reader:
#         next_sample = data_reader.read()
#         sample_text = next_sample["text"]

#     print(f"sample_text = {sample_text}")
#     # with jsonlines.open('./data/subset_data.jsonl') as data_reader:
#     #     # data_reader = jsonlines.Reader('./data/subset_data.jsonl')
#     #     first = data_reader.read()
#     #     second = data_reader.read()
#     #     print(f"first = {first}")
#     #     print(f"second = {second}")

#     # print(f"did anything happen?")
    
    