# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json

from pathlib import Path
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from datasets import load_dataset

from llama import ModelArgs, Transformer, Tokenizer, LLaMA

from dataset_stream import PileIterableDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def load(
    model_args: ModelArgs,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
    ckpt_dir: str = None,
) -> LLaMA:
    start_time = time.time()
    if ckpt_dir is not None:
        # If there are checkpoints provided, then load them in
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert world_size == len(
            checkpoints
        ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
        ckpt_path = checkpoints[local_rank]
        print("Loading")
        checkpoint = torch.load(ckpt_path, map_location="cpu")

    # TODO: Set up params.json or get rid of this
    # with open(Path(ckpt_dir) / "params.json", "r") as f:
    #     params = json.loads(f.read())

    print(f"tokenizer_path in load func = {tokenizer_path}")
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    if ckpt_dir:
        model.load_state_dict(checkpoint, strict=False)

    return model, tokenizer


def train(model, tokenizer, dataloader, epochs=1, lr=0.01, beta1=0.9, beta2=0.95, decay=0.01, clip=1.0, temperature=0, top_p=0.95, verbose=1):
  model.to(device)
  losses = []
  # We set reduction=None to avoid computing mean on losses (so we get raw losses), this allows us to
  # zero any losses that occured from padding before reducing our 
  criterion = nn.CrossEntropyLoss() # Is softmax + negative log likelihood
#   optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=decay)
  optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=decay)
  for epoch in range(epochs):
    sum_loss = 0.0
    for i, batch in enumerate(dataloader):
        # get the inputs; data is a list of string prompts
        prompts = batch['text']

        print(f"size of prompts: {len(prompts)}")
        print(f"prompts: {prompts}")

        bsz = len(prompts)
        params = model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        max_gen_len = 1  # For training, the model will only be predicting the next token given all the preceeding tokens
                         # the prompt sample. So we will just make this 1 so that if our batch of prompts is smaller than
                         # the max_seq_len, then the total_len for padding will just be 1 + the size of the longest prompt
                         # in our batch.

        # Dim of prompt_tokens: (bsz, len_of_each_prompt)
        prompt_tokens = [tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        # print(f"prompt_tokens = {prompt_tokens}")
        # print(f"len of first prompt tokens = {len(prompt_tokens[0])}")

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)
        # print(f"total_len = {total_len}")

        tokens = torch.full((bsz, total_len), tokenizer.pad_id).to(device).long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()  # k represents the kth prompt and t represents the position in that sentence
        # input_text_mask = tokens != tokenizer.pad_id  TODO: Remove
        prev_pos = 0
        # print(f"shape of tokens: {tokens.shape}")
        for cur_pos in range(1, total_len - 1):
            """ For each token (not including the first token, which we will use first as context)"""
             # zero the parameter gradients
            optimizer.zero_grad()

            print(f"prev_pos:cur_pos = {tokens[:, prev_pos:cur_pos]}")
            logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)  # TODO: Figure out, do we want to start from prev_pos as context or the beginning of the sentence?
            print(f"logits grad_fn = {logits.grad_fn}")
            print(f"logits = {logits}")
            # print(f"tokens[:, cur_pos] = {tokens[:, cur_pos]}")
            loss = criterion(logits, tokens[:, cur_pos+1])
            print(f"current tokens = {tokens[:, cur_pos+1]}")
            print(f"loss func grad func = {loss.grad_fn}")
            print(f"unreduced loss = {loss}")

            # Zero out any losses that were calculated from padding tokens
            # loss_mask = tokens[:, cur_pos] != tokenizer.pad_id
            # # loss_mask tensor([ True,  True,  True,  True,  True, False])

            # # loss_masked = loss.where(loss_mask, torch.tensor(0.0))
            # # loss_masked tensor([ 0.0010, -0.3000,  0.9000,  0.7000,  0.6000,  0.0000])

            # # loss = loss_masked.mean()  might be better to just calculate mean() instead of doing it manually below
            # # loss = loss_masked.sum() / loss_mask.sum()

            # loss_masked = torch.masked_select(loss, loss_mask)
            # loss_masked.mean()

            loss.backward()  # autograd magic, computes all the partial derivatives

            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step() # takes a step in negative gradient direction

            # print statistics
            losses.append(loss.item())
            sum_loss += loss.item()
            if i % 100 == 99:    # print every 100 mini-batches
                if verbose:
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, sum_loss / 100))
                sum_loss = 0.0

            prev_pos = cur_pos

  return losses


def main(
    tokenizer_path: str,
    ckpt_dir: str = None,
    temperature: float = 0.8,
    top_p: float = 0.95,
    # max_seq_len: int = 512,
    # max_batch_size: int = 32,
):
    # train_path = 'data/train.jsonl'  # TODO:
    train_path = 'data/tiny_test_set.jsonl'
    test_path = 'data/train.jsonl'
    # test_path = 'data/' TODO:

    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    # Model params
    dim: int = 512
    n_layers: int = 8  # changed from 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 512

    # Hyperparams for LLama Transformer implementation
    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, 
        vocab_size=vocab_size, 
        dim=dim, n_layers=n_layers, 
        n_heads=n_heads, 
        multiple_of=multiple_of, 
        norm_eps=norm_eps, 
        max_batch_size=max_batch_size  # TODO: change the max_seq_len and max_batch_size(?)
    )

    print(f"tokenizer_path = {tokenizer_path}")
    model, tokenizer = load(
        model_args, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size, ckpt_dir
    )
    print(f"tokenizer n_words = {tokenizer.n_words}")

    # Load in the training and testing dataset TODO
    # train_dataset = PileIterableDataset(train_path)
    train_dataset = load_dataset('json', data_files=train_path, split='train').with_format('torch')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1)
    # test_dataloader = TODO:

    # Train the model TODO
    # It should be as simple as batching up multiple prompts (need to split)
    train(model, tokenizer, train_dataloader, epochs=1)


if __name__ == "__main__":
    fire.Fire(main)
