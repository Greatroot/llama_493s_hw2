import os
import json
import jsonlines
import itertools
import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt


class PileDataset(Dataset):
    """Pile dataset."""

    def __init__(self, path_to_jsonl, tokenizer, seq_len):
        """
        Arguments:
            path_to_jsonl (string): Path to the .jsonl file with training samples
                                 from the Pile Dataset.
            tokenizer (Tokenizer): The pretrained Tokenizer object that we'll use
                                 to tokenize our data
            seq_len (int): The number of tokens that we want each sample to be
        """
        self.seq_len = seq_len

        # Create our list of sample sentences from our jsonl data file
        sentences = []
        with jsonlines.open(path_to_jsonl, 'r') as input_reader:
            for record in input_reader:
                sentences.append(record['text'])
        
        # Tokenize our sentences
        tokens = itertools.chain.from_iterable([tokenizer.encode(x, bos=True, eos=True) for x in tqdm.tqdm(sentences)])

        # Flatten our tokens so that they can easily be batched with seq_len of our choice
        # (i.e. we won't have to worry about padding)
        self.flattened_tokens = torch.tensor(list(tokens)).flatten()

    def __len__(self):
        return (len(self.flattened_tokens) - 1) // self.seq_len

    def __getitem__(self, idx):
        input = self.flattened_tokens[idx * self.seq_len : (idx + 1) * self.seq_len]
        target = self.flattened_tokens[idx * self.seq_len + 1: (idx + 1) * self.seq_len + 1]

        return input, target
