# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import torch
import time
import argparse

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, TransformerInference, Tokenizer, LLaMA

parser = argparse.ArgumentParser(description='Training code for transformer')
parser.add_argument('tokenizer_path', type=str, help='Path to tokeinizer')
parser.add_argument('ckpt_path', type=str, help='Path to checkpoint for a pretrained LLaMA model')
parser.add_argument('--max_seq_len', type=int, default=512, help='The max number of tokens per sequence')
parser.add_argument('--max_batch_size', type=int, default=16, help='Training batch size')

args = parser.parse_args()


def load(
    ckpt_path: str,
    tokenizer_path: str,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    print("Loading")
    checkpoint = torch.load(ckpt_path)

    # print(f"checkpoint['model_params'] = {checkpoint['model_params']}")
    if 'model_params' in checkpoint:
        model_args: ModelArgs = checkpoint['model_params']
    else:
        model_args: ModelArgs = ModelArgs(
            dim=256,
            n_layers=2,
            n_heads=2,
            max_seq_len=args.max_seq_len,
            multiple_of=256, # make SwiGLU hidden layer size multiple of large power of 2
            norm_eps=1e-5
        )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = TransformerInference(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
):
    generator = load(
        ckpt_dir, tokenizer_path, max_seq_len, max_batch_size
    )

    prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",
        "Simply put, the theory of relativity states that ",
        "Building a website can be done in 10 simple steps:\n",
        # Few shot prompts: https://huggingface.co/blog/few-shot-learning-gpt-neo-and-inference-api
        """Tweet: "I hate it when my phone battery dies."
Sentiment: Negative
###
Tweet: "My day has been 👍"
Sentiment: Positive
###
Tweet: "This is the link to the article"
Sentiment: Neutral
###
Tweet: "This new music video was incredibile"
Sentiment:""",
        """Translate English to French:

sea otter => loutre de mer

peppermint => menthe poivrée

plush girafe => girafe peluche

cheese =>""",
    ]
    results = generator.generate(
        prompts, max_gen_len=256, temperature=temperature, top_p=top_p
    )

    for result in results:
        print(result)
        print("\n==================================\n")


if __name__ == "__main__":
    main(ckpt_dir=args.ckpt_path, tokenizer_path=args.tokenizer_path)
