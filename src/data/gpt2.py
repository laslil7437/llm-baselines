from tqdm import tqdm
import numpy as np
import tiktoken
from transformers import GPT2Tokenizer, GPT2Model
from datasets import load_dataset, Dataset
import os
import shutil
import torch

#FINEWEBMINI_DATA_PATH = os.path.join(os.path.dirname(__file__), "datasets/finewebgpt2/")

gpt2_base = tiktoken.get_encoding("gpt2")

dataset = load_dataset("HuggingFaceFW/fineweb", split="train", name="sample-10BT")
dataset = dataset.select_columns(["text", "dump"])
dataset = dataset.take(50)
print("----> dataset loaded")

split_dataset = dataset.train_test_split(
    test_size=0.1, seed=2357, shuffle=True
)
split_dataset["val"] = split_dataset.pop("test")
print("----> dataset split")

seq_len = 512
# special token for each dump
dumps_list = list(set(example["dump"] for example in dataset))
dumps_tokens_dict = {dump: i+gpt2_base.n_vocab for i, dump in enumerate(dumps_list)}

tknzr = tiktoken.Encoding(
    # If you're changing the set of special tokens, make sure to use a different name
    # It should be clear from the name what behaviour to expect.
    name="gpt2_dumps",
    pat_str=gpt2_base._pat_str,
    mergeable_ranks=gpt2_base._mergeable_ranks,
    special_tokens={
        **gpt2_base._special_tokens,
        # each pair of date:token from date_tokens dict w/o listing manually like 'cc1': date_tokens['cc1'],
        **{dump: dumps_tokens_dict[dump] for dump in dumps_list}
        
    }
)

def process(example):
    ids = tknzr.encode_ordinary(
        example["text"]
    )  # encode_ordinary ignores any special tokens

    ids.append(
        tknzr.eot_token
    )  # add the end of text token, e.g. 50256 for gpt2 bpe

    # add the dump token every 512th token
    dump_token_id = tknzr._special_tokens[example["dump"]]
    for i in range(0, len(ids), seq_len):
        ids.insert(i, dump_token_id)
    breakpoint()
    out = {"ids": ids, "len": len(ids)}
    return out

process(dataset[0])

breakpoint()
# tokenize the dataset
tokenized = split_dataset.map(
    process,
    remove_columns=["text", "dump"],
    desc="tokenizing the splits",
    num_proc=40,
)

breakpoint()
