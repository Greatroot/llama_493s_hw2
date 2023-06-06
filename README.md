# Trainable LLaMA 

This repository is Ben Kosa, Mike Violette, and Alessio Tosolini's trainable LLaMA code for CSE 493s, Homework 2. This code is largely based off of FaceBook Research's offical GitHub repository for the PyTorch LLaMA implementation, though altered to allow for training LLaMA from scratch. The original repository can be found here: [LLaMA GitHub](https://github.com/facebookresearch/llama). The original paper and article for LLaMA can be found here: [LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/) ([arXiv](https://arxiv.org/abs/2302.13971v1)).

## Setup

In a conda env with pytorch / cuda available, run:
```
pip install -r requirements.txt
```
Then in this repository:
```
pip install -e .
```
You may also need to install tqdm, jsonlines, and matplotlib
```
conda install matplotlib tqdm
```
```
conda install -c conda-forge jsonlines
```

## Download

The `tokenizer.model` weights for the pretrained SentencePiece Tokenizer provided by Meta can be found within this repo. The final checkpoints for our best 20M parameter model and 8M and 20M models trained during abolition testing can be found [here]{https://drive.google.com/drive/folders/1kdPKNX2-aNuNTAVbHxEKlKGZX8WAzrs4?usp=sharing}.

Download the data that we used for performing our abolition tests, along with training our best model and running validation here: [Download data here](https://drive.google.com/drive/folders/1dn8QlNtgwVzzgB5_fAF0n9-GEzHIaI5n?usp=sharing)

Be sure to create a folder called `data` to store all data files in there. Data files come compressed as `.zst` files, so be sure to uncompress them with the `zstd -d` command before using them for training and validation. 

## Inference

The provided `example.py` can be run on a single GPU node with `python` and will output completions for mutltiple pre-defined prompt. An example command to run the inference on a model checkpoint is shown below. Refer to the top of `example.py` to see all args that can be passed in.
```
python example.py ./tokenizer.model ./checkpoints/19_3.63_5.73.pt
```
```
python example.py ./<PATH_TO_TOKENIZER_WEIGHTS>.model ./<PATH_TO_CHECKPOINT_FILE>.pt
```

To pass custom prompts into example.py for generative inference, simply add them as strings to the `prompts` list in the `main()` function in `example.py`.

## Training and Validation

The provided `train.py` can be run on a single GPU node with `python` and will output completions for mutltiple pre-defined prompt. An example command to run the training and validation pipeline can be found below. Refer to the top of `train.py` to see all args that can be passed in.
```
python train.py tokenizer.model ./checkpoints ./data/train_100000.jsonl ./data/val_10000.jsonl --dim_size=256 --epochs=12
```
```
python train.py ./<PATH_TO_PRETRAINED_TOKENIZER_WEIGHTS> ./<FILE_YOU_WANT_TO_SAVE_CHECKPOINTS> ./data/<DATA_FILE>.jsonl ./data/<VAL_FILE>.jsonl --dim_size=256 --epochs=12
```

## FAQ

- [1. The download.sh script doesn't work on default bash in MacOS X](FAQ.md#1)
- [2. Generations are bad!](FAQ.md#2)
- [3. CUDA Out of memory errors](FAQ.md#3)
- [4. Other languages](FAQ.md#4)

## Reference

LLaMA: Open and Efficient Foundation Language Models -- https://arxiv.org/abs/2302.13971

```
@article{touvron2023llama,
  title={LLaMA: Open and Efficient Foundation Language Models},
  author={Touvron, Hugo and Lavril, Thibaut and Izacard, Gautier and Martinet, Xavier and Lachaux, Marie-Anne and Lacroix, Timoth{\'e}e and Rozi{\`e}re, Baptiste and Goyal, Naman and Hambro, Eric and Azhar, Faisal and Rodriguez, Aurelien and Joulin, Armand and Grave, Edouard and Lample, Guillaume},
  journal={arXiv preprint arXiv:2302.13971},
  year={2023}
}
```

## Model Card
See [MODEL_CARD.md](MODEL_CARD.md)

## License
See the [LICENSE](LICENSE) file.
