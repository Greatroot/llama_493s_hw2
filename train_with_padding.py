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
import tqdm
import argparse

from pathlib import Path
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from datasets import load_dataset

from llama import ModelArgs, Transformer, Tokenizer, LLaMA

parser = argparse.ArgumentParser(description='Training code for transformer')
parser.add_argument('tokenizer_path', type=str, help='Path to tokeinizer')
parser.add_argument('save_path', type=str, help='Path to folder to save results')
parser.add_argument('train_path', type=str, help='Path to the data file you want to train on')
parser.add_argument('val_path', type=str, help='Path to the data file you want to validate on')
parser.add_argument('--ckpt_path', type=str, help='Path to checkpoint if you want to load one in')
parser.add_argument('--max_seq_len', type=int, default=256, help='The max number of tokens per sequence')
parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
parser.add_argument('--num_epochs', type=int, default=7, help='Number of epochs to train for')
parser.add_argument('--dim_size', type=int, default=256, help='Embedding dimension for the Embedder in our Transformer')

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def load(
    model_args: ModelArgs,
    tokenizer_path: str,
    ckpt_dir: str = None,
) -> LLaMA:
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


def eval(model, tokenizer, test_loader):
    losses = 0.
    with torch.no_grad():
        criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
        for i, vdata in tqdm.tqdm(enumerate(test_loader), total=len(test_loader)):
            prompts = vdata['text']
            params = model.params
            batch_size = len(prompts)

            prompt_tokens = [tokenizer.encode(x, bos=True, eos=True) for x in prompts]
            max_prompt_size = max([len(t) for t in prompt_tokens])

            total_len = min(params.max_seq_len, max_prompt_size)

            # Pad our batch of sample tokens so that they are all the same length
            tokens = torch.full((batch_size, total_len), tokenizer.pad_id).to(device).long()
            for k, t in enumerate(prompt_tokens):
                # k represents the kth prompt and t represents the position in that sentence
                if len(t) > params.max_seq_len:
                    tokens[k, :] = torch.tensor(t[:total_len]).long()
                else:
                    tokens[k, : len(t)] = torch.tensor(t).long()
            
            inputs = tokens[:, :-1]
            targets = tokens[:, 1:]

            outputs = model.forward(inputs, 0)

            loss = criterion(outputs.reshape(-1, params.vocab_size), targets.reshape(-1))
            losses += loss.item()
        losses /= len(test_loader)
    return losses


def train(model, tokenizer, train_loader, val_loader, epochs=7, lr=0.01, beta1=0.9, beta2=0.95, decay=0.01, clip=1.0, batch_size=16, save_path=None):
  model.to(device)
  model.train()
  pad_id = 0
  criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
  optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=decay)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader))

  train_losses = []
  fine_train_losses = []
  val_losses = []

  for epoch in range(epochs):
    train_loss = 0.
    epoch_train_loss = 0.
    log_interval = 100
    start_time = time.time()
    for i, batch in enumerate(train_loader):
        # get the inputs; batch is a list of string prompts. We convert these to a padded batch of 
        # tokens before passing into our model.
        prompts = batch['text']
        params = model.params

        # Dim of prompt_tokens: (batch_size, len_of_each_prompt)
        prompt_tokens = [tokenizer.encode(x, bos=True, eos=True) for x in prompts]
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_prompt_size)

        # Pad our batch of sample tokens so that they are all the same length
        tokens = torch.full((batch_size, total_len), tokenizer.pad_id)
        for k, t in enumerate(prompt_tokens):
            # k represents the kth prompt and t represents the position in that sentence
            if len(t) > params.max_seq_len:
                tokens[k, :] = torch.tensor(t[:total_len]).long()
            else:
                tokens[k, : len(t)] = torch.tensor(t).long()
        
        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]
            
        # zero the parameter gradients
        optimizer.zero_grad()

        logits = model.forward(inputs.to(device).long(), 0)

        # Calculate a mask for the loss to mask out loss calculated on target tokens that are just padding (we want to ignore these)
        loss = criterion(logits.reshape(-1, params.vocab_size), targets.reshape(-1).to(device).long())

        loss.backward()  # autograd magic, computes all the partial derivatives

        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step() # takes a step in negative gradient direction

        # Total loss over epoch and print statistics
        train_loss += loss.item()
        epoch_train_loss += loss.item()
        if i % log_interval == 0 and i > 0:
            lr = scheduler.get_last_lr()[0]
            # lr = get_lr(optimizer)
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = epoch_train_loss / log_interval
            ppl = math.exp(cur_loss)

            print(f'| epoch {epoch:3d} | {i:5d}/{len(train_loader):5d} batches | '
                  f'lr {lr:02.5f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            fine_train_losses.append(cur_loss)
            epoch_train_loss = 0
            start_time = time.time()
        
        if i != 0 and i % 3000 == 0:
          scheduler.step()

    # At the end of every epoch, test our model against our validation data
    val_loss = eval(model, tokenizer, val_loader)
    val_losses.append(val_loss)
    train_losses.append(train_loss / len(train_loader))
    print('Avg LOSSES | train loss: {} | valid loss: {}'.format(train_loss / len(train_loader), val_loss))

    # save a checkpoint
    # filename format: {epoch}_{training loss}_{validation loss}.pt
    print(f"model_params: {model.params}")
    if save_path is not None:
        # torch.save(model.state_dict(), f'{save_path}/model_epoch_{epoch}_{train_loss / len(train_loader):.3}_{val_loss:.3}.pt')
        torch.save({
                'model_params': model.params,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                }, os.path.join(save_path, '{}_{:.3}_{:.3}.pt'.format(epoch, train_loss / len(train_loader), val_loss)))


  # Create a plot using plt of the total_losses and the epoch_validation and save it to args.save_path
  plt.cla(); plt.clf()
  plt.plot(fine_train_losses, label='train_loss', color='blue')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.xscale('log')
  plt.title("More Fine-grain Training Losses")
  plt.legend()
  plt.savefig(f'{args.save_path}/fine_grain_train_losses.png')

  plt.cla(); plt.clf()
  plt.plot(train_losses, label='train_loss', color='blue')
  plt.plot(val_losses, label='val_loss', color='orange')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.xscale('log')
  plt.title("Train and Validation Losses per Epoch")
  plt.legend()
  plt.savefig(f'{args.save_path}/final_epoch_losses.png')
  
  return train_losses, val_losses


def main():
    print(f"args.save_path = {args}")
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    elif not os.path.isdir(args.save_path):
        raise ValueError(f"{args.save_path} is not a directory")

    # Model params
    lr=3.0e-4
    model_args: ModelArgs = ModelArgs(
        dim=args.dim_size,
        n_layers=2,
        n_heads=2,
        max_seq_len=args.max_seq_len,
        multiple_of=256, # make SwiGLU hidden layer size multiple of large power of 2
        norm_eps=1e-5
    )

    model, tokenizer = load(
        model_args, args.tokenizer_path, args.ckpt_path
    )
    
    model_size = count_parameters(model)
    print(f"Model Summary: "
          f"\tembedding dim = {args.dim_size}"
          f"\tnum_layers = {model_args.n_layers}"
          f"\tn_heads = {model_args.n_heads}"
          f"\tmax_seq_len = {args.max_seq_len}"
          f"\tnum_params (size of model) = {model_size}"
          )

    # Load in our data and split it into train, val, test datasets
    # train_dataset, val_dataset, test_dataset = load_dataset('json', data_files=args.data_path, split=['train[:80%]', 'train[-20%:-10%]', 'train[-10%:]'])
    train_dataset = load_dataset('json', data_files=args.train_path, split='train')
    val_dataset = load_dataset('json', data_files=args.val_path, split='train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    # test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Train the model
    start_time = time.time()
    train_losses, val_losses = train(model, tokenizer, train_loader, val_loader, lr=lr, epochs=args.num_epochs, batch_size=args.batch_size, save_path=args.save_path)
    end_time = time.time()
    with open(f"{model_size}_10000_train_summary.txt", "a") as f:
        print(f"Training on 3090 took {end_time - start_time} seconds")
        print(f"num of model params (model size): {model_size}")
        print(f"Final validation loss on model was: {val_losses[-1]}")
        print(f"\ntrain_losses = {train_losses}")
        print(f"\nval_losses = {val_losses}")


if __name__ == "__main__":
    main()