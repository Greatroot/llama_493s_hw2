# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json
import math

from pathlib import Path
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from datasets import load_dataset

from llama import ModelArgs, Transformer, Tokenizer, LLaMA

# from dataset_stream import PileIterableDataset  TODO: Remove

# device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load(
    model_args: ModelArgs,
    tokenizer_path: str,
    ckpt_dir: str = None,
) -> LLaMA:
    start_time = time.time()
    if ckpt_dir is not None:
        # If there are checkpoints provided, then load them in
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        ckpt_path = checkpoints
        print("Loading")
        checkpoint = torch.load(ckpt_path, map_location="cpu")

    # TODO: Set up params.json or get rid of this
    # with open(Path(ckpt_dir) / "params.json", "r") as f:
    #     params = json.loads(f.read())

    print(f"tokenizer_path in load func = {tokenizer_path}")
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    # torch.set_default_tensor_type(torch.cuda.HalfTensor)  # TODO: ask why they do this
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    if ckpt_dir:
        model.load_state_dict(checkpoint, strict=False)

    return model, tokenizer


def train(model, tokenizer, train_dataloader, val_dataloader, epochs=1, lr=0.01, beta1=0.9, beta2=0.95, decay=0.01, clip=1.0, temperature=0, top_p=0.95, batch_size=32):
# def train(model, tokenizer, train_dataloader, val_dataloader, epochs=1, lr=0.01, beta1=0.9, beta2=0.95, decay=0.01, clip=1.0, temperature=0, top_p=0.95, batch_size=32):
  model.to(device)
  model.train()
  # We set reduction=None to avoid computing mean on losses (so we get raw losses), this allows us to
  # zero any losses that occured from padding before reducing our 
  criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.eos_id) # Is softmax + negative log likelihood
#   optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=decay)
  optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=decay)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dataloader))
  for epoch in range(epochs):
    losses = []
    total_loss = 0.
    log_interval = 1000
    start_time = time.time()
    for i, batch in enumerate(train_dataloader):
        # get the inputs; data is a list of string prompts
        # print(f"batch = {batch}")
        prompts = batch['text']

        # print(f"size of prompts: {len(prompts)}")
        # print(f"prompts: {prompts}")
        # print(f"type of elements in prompts: {type(prompts[0])}")

        params = model.params

        # Dim of prompt_tokens: (batch_size, len_of_each_prompt)
        prompt_tokens = [tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        # print(f"len of first prompt tokens = {len(prompt_tokens[0])}")

        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_prompt_size)
        # print(f"max_prompt_size = {max_prompt_size}")
        # print(f"total_len = {total_len}")

        # Pad our batch of sample tokens so that they are all the same length
        tokens = torch.full((batch_size, total_len), tokenizer.eos_id).to(device).long()
        for k, t in enumerate(prompt_tokens):
            # k represents the kth prompt and t represents the position in that sentence
            # print(f"shape of {k}th sample's lhs: {tokens[k, : len(t)].shape}")
            # print(f"shape of {k}th sample's rhs: {torch.tensor(t).long().shape}")
            if len(t) > params.max_seq_len:
                tokens[k, :] = torch.tensor(t[:total_len]).long()
            else:
                tokens[k, : len(t)] = torch.tensor(t).long()
        
        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]
        # print(f"shape of tokens: {tokens.shape}")
            
        # zero the parameter gradients
        optimizer.zero_grad()

        # print(f"shape of inputs = {inputs.shape}")
        # print(f"shape of targets = {targets.shape}")
        # print(f"inputs = {inputs}")
        # print(f"targets = {targets}")
        logits = model.forward(inputs, 0)

        # Calculate a mask for the loss to mask out loss calculated on target tokens that are just padding (we want to ignore these)
        # loss_mask = targets.reshape(-1) == tokenizer.pad_id  TODO: remove
        # print(f"logits.reshape(-1, vocab_size) shape = {logits.reshape(-1, params.vocab_size).shape}")
        # print(f"logits.reshape(-1, vocab_size) = {logits.reshape(-1, params.vocab_size)}")
        # print(f"targets.reshape(-1) = {targets.reshape(-1)}")
        loss = criterion(logits.reshape(-1, params.vocab_size), targets.reshape(-1))
        # loss = torch.autograd.Variable(loss, requires_grad=True)
        # print(f"unreduced loss = {loss}")

        # loss_mask tensor([ True,  True,  True,  True,  True, False])

        # loss_masked = loss.where(loss_mask, torch.tensor(0.0))
        # loss_masked tensor([ 0.0010, -0.3000,  0.9000,  0.7000,  0.6000,  0.0000])

        # loss = loss_masked.mean()  might be better to just calculate mean() instead of doing it manually below
        # loss = loss_masked.sum() / loss_mask.sum()

        # loss_masked = torch.masked_select(loss, loss_mask)
        # loss_masked.mean()

        loss.backward()  # autograd magic, computes all the partial derivatives

        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step() # takes a step in negative gradient direction

        # Total loss over epoch and print statistics
        total_loss += loss.item()
        if i % log_interval == 0 and i > 0:
            lr = scheduler.get_last_lr()[0]
            # lr = get_lr(optimizer)
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {i:5d}/{len(train_dataloader):5d} batches | '
                  f'lr {lr:02.5f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()
        
        if i != 0 and i % 3000 == 0:
          scheduler.step()

    # At the end of every epoch, test our model against our validation data
    # running_vloss = 0.0
    # for i, vdata in enumerate(val_dataloader):
    #     prompts = vdata['text']
    #     params = model.params

    #     prompt_tokens = [tokenizer.encode(x, bos=True, eos=False) for x in prompts]
    #     max_prompt_size = max([len(t) for t in prompt_tokens])

    #     total_len = min(params.max_seq_len, max_prompt_size)

    #     # Pad our batch of sample tokens so that they are all the same length
    #     tokens = torch.full((batch_size, total_len), tokenizer.eos_id).to(device).long()
    #     for k, t in enumerate(prompt_tokens):
    #         # k represents the kth prompt and t represents the position in that sentence
    #         if len(t) > params.max_seq_len:
    #             tokens[k, :] = torch.tensor(t[:total_len]).long()
    #         else:
    #             tokens[k, : len(t)] = torch.tensor(t).long()
        
    #     inputs = tokens[:, :-1]
    #     targets = tokens[:, 1:]

    #     voutputs = model.forward(inputs, 0)

    #     vloss = criterion(voutputs.reshape(-1, params.vocab_size), targets.reshape(-1))
    #     running_vloss += vloss

    # avg_vloss = running_vloss / (i + 1)
    # print('Avg LOSS train {} valid {}'.format(total_loss / len(train_dataloader), avg_vloss))

  return losses


def main(
    tokenizer_path: str,
    ckpt_dir: str = None
):
    # train_path = 'data/train.jsonl'  # TODO:
    train_path = 'data/subset_data.jsonl'
    # test_path = 'data/test.jsonl'
    # test_path = 'data/' TODO:

    # Model params
    dim: int = 128  # Originally 512
    n_layers: int = 1  # originally 8
    n_heads: int = 1  # originally 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    epochs = 10
    batch_size: int = 2
    lr=3.0e-4

    # Hyperparams for LLama Transformer implementation
    model_args: ModelArgs = ModelArgs(
        vocab_size=vocab_size, 
        dim=dim, 
        n_layers=n_layers, 
        n_heads=n_heads, 
        multiple_of=multiple_of, 
        norm_eps=norm_eps,
        max_seq_len=512  # originally 2048
    )

    print(f"tokenizer_path = {tokenizer_path}")
    model, tokenizer = load(
        model_args, tokenizer_path, ckpt_dir
    )
    
    print(f"tokenizer n_words = {tokenizer.n_words}")

    # Load in our data and split it into train, val, test datasets
    train_dataset, val_dataset, test_dataset = load_dataset('json', data_files=train_path, split=['train[:80%]', 'train[-20%:-10%]', 'train[-10%:]'])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Train the model
    train(model, tokenizer, train_dataloader, val_dataloader, lr=lr, epochs=epochs, batch_size=batch_size)


if __name__ == "__main__":
    tokenizer_path = './tokenizer.model'
    main(tokenizer_path=tokenizer_path)