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
from training_utils import PileDataset

from llama import ModelArgs, Transformer, Tokenizer, LLaMA

parser = argparse.ArgumentParser(description='Training code for transformer')
parser.add_argument('tokenizer_path', type=str, help='Path to tokeinizer')
parser.add_argument('save_path', type=str, help='Path to folder to save results')
parser.add_argument('train_path', type=str, help='Path to the data file you want to train on')
parser.add_argument('val_path', type=str, help='Path to the data file you want to validate on')
parser.add_argument('--ckpt_path', type=str, help='Path to checkpoint if you want to load one in')
parser.add_argument('--seq_len', type=int, default=512, help='The max number of tokens per sequence')
parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
parser.add_argument('--num_epochs', type=int, default=7, help='Number of epochs to train for')
parser.add_argument('--dim_size', type=int, default=128, help='Embedding dimension for the Embedder in our Transformer')

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def batchify(data, bsz):
    """Divides the data into bsz separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Args:
        data: Tensor, shape [N]
        bsz: int, batch size

    Returns:
        Tensor of shape [N // bsz, bsz]
    """
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data.to(device)


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
        criterion = nn.CrossEntropyLoss()
        for i, (inputs, targets) in tqdm.tqdm(enumerate(test_loader), total=len(test_loader)):
            inputs, targets = inputs.to(device), targets.to(device)
            params = model.params

            outputs = model.forward(inputs, 0)

            loss = criterion(outputs.reshape(-1, params.vocab_size), targets.reshape(-1))
            losses += loss.item()
        losses /= len(test_loader)
    return losses


def train(model, tokenizer, train_loader, val_loader, epochs=7, lr=0.001, beta1=0.9, beta2=0.95, decay=0.01, clip=1.0, batch_size=16, save_path=None):
  model.to(device)
  model.train()
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=decay)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader))

  train_losses = []
  fine_train_losses = []
  val_losses = []

  for epoch in range(epochs):
    train_loss = 0.
    epoch_train_loss = 0.
    log_interval = 100  # TODO: change back to 100
    start_time = time.time()
    for i, (inputs, targets) in enumerate(train_loader):
        # get the inputs; batch is a list of string prompts. We convert these to a padded batch of 
        # tokens before passing into our model.
        inputs, targets = inputs.to(device), targets.to(device)
        params = model.params
            
        # zero the parameter gradients
        optimizer.zero_grad()

        logits = model.forward(inputs, 0)

        # Calculate a mask for the loss to mask out loss calculated on target tokens that are just padding (we want to ignore these)
        loss = criterion(logits.reshape(-1, params.vocab_size), targets.reshape(-1))

        loss.backward()  # autograd magic, computes all the partial derivatives

        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step() # takes a step in negative gradient direction

        # Total loss over epoch and print statistics
        train_loss += loss.item()
        epoch_train_loss += loss.item()
        if i % log_interval == 0 and i > 0:
            # prev_lr = scheduler.get_last_lr()[0]
            lr = scheduler.get_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = epoch_train_loss / log_interval
            ppl = math.exp(cur_loss)

            print(f'| epoch {epoch:3d} | {i:5d}/{len(train_loader):5d} batches | '
                  f'lr {lr:03.6f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            fine_train_losses.append(cur_loss)
            epoch_train_loss = 0
            start_time = time.time()

    # At the end of every epoch, test our model against our validation data and decrease learning rate via scheduler
    scheduler.step()
    val_loss = eval(model, tokenizer, val_loader)
    val_losses.append(val_loss)
    train_losses.append(train_loss / len(train_loader))
    print('Avg LOSSES | train loss: {} | valid loss: {}'.format(train_loss / len(train_loader), val_loss))

    # save a checkpoint
    # filename format: {epoch}_{training loss}_{validation loss}.pt
    print(f"model_params: {model.params}")
    if save_path is not None:
        # torch.save(model.state_dict(), f'{save_path}/model_epoch_{epoch}_{train_loss / len(train_loader):.3}_{val_loss:.3}.pt') TODO: Remove
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
  plt.suptitle("More Fine-grain Training Losses")
  plt.title(f"Model Size: {count_parameters(model)} | train_sze=10000 | dim_size={params.dim} | seq_len={params.max_seq_len}")
  plt.legend()
  plt.savefig(f'{args.save_path}/fine_grain_train_losses.png')

  plt.cla(); plt.clf()
  plt.plot(train_losses, label='train_loss', color='blue')
  plt.plot(val_losses, label='val_loss', color='orange')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.xscale('log')
  plt.suptitle("Train and Validation Losses per Epoch")
  plt.title(f"Model Size: {count_parameters(model)} | train_sze=10000 | dim_size={params.dim} | seq_len={params.max_seq_len}")
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
    lr=0.0008
    model_args: ModelArgs = ModelArgs(
        dim=args.dim_size,
        n_layers=3,
        n_heads=4,
        max_seq_len=args.seq_len,
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
          f"\tseq_len = {args.seq_len}"
          f"\tnum_params (size of model) = {model_size}"
          )

    # Load in our data and split it into train, val, test datasets
    # train_dataset, val_dataset, test_dataset = load_dataset('json', data_files=args.data_path, split=['train[:80%]', 'train[-20%:-10%]', 'train[-10%:]'])
    train_dataset = PileDataset(path_to_jsonl=args.train_path, tokenizer=tokenizer, seq_len=args.seq_len)
    val_dataset = PileDataset(path_to_jsonl=args.val_path, tokenizer=tokenizer, seq_len=args.seq_len)
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