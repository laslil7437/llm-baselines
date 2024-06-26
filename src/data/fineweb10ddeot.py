# dump tokens included per example at end of sequence
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset, Dataset
import os
import shutil
from collections import defaultdict
from urllib.parse import urlparse


FINEWEB10_DATA_PATH = os.path.join(os.path.dirname(__file__), "datasets/fineweb10ddeot/") 

tknzr = tiktoken.get_encoding("gpt2")

def get_fineweb10ddeot_data(num_proc=40):

    if not os.path.exists(os.path.join(FINEWEB10_DATA_PATH, "train.bin")):
        os.makedirs(FINEWEB10_DATA_PATH, exist_ok=True)
        
        dataset = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT")
        dataset = dataset.select_columns(["text", "dump", "url"])
        print("----> dataset loaded")


        # function to return domain name
        def extract_domain(url):
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            ending = domain.split('.')[-1]
            return ending
        
        # edit "url" column to just be domain name
        dataset = dataset.map(lambda x: {"url": extract_domain(x["url"])})
        # edit "url" column to be "other" or one with at least 100 occurrences
        domains = defaultdict(int)
        us_domains = ["edu", "gov", "mil"]
        for example in dataset['train']:
            if len(example["url"]) == 2 or example["url"] in us_domains:
                if example["url"] not in domains:
                    domains[example["url"]] = 1
                else:
                    domains[example["url"]] += 1
        # domains with at least 100 occurrences
        valid_domains = [k for k, v in domains.items() if v >= 100]

        def convert_to_other(example):
            if example["url"] not in valid_domains:
                example["url"] = "other"
            else:
                example["url"] = example["url"]
            return example
        dataset = dataset.map(convert_to_other)
        print("----> dataset modified")
        
        split_dataset = dataset["train"].train_test_split(
            test_size=0.1, seed=2357, shuffle=True
        )
        split_dataset["val"] = split_dataset.pop("test")
        print("----> dataset split")


        # special token for each domain (url) and dump
        dump_tokens = set(example["dump"] for example in dataset["train"])
        valid_domains.append("other")
        # combine into one 
        dd_tokens = valid_domains + list(dump_tokens)
        dd_token_ids = {token: i+50256+1 for i, token in enumerate(dd_tokens)}

        def process(example):
            ids = tknzr.encode_ordinary(
                example["text"]
            )  # encode_ordinary ignores any special tokens
            ids.append(
                tknzr.eot_token
            )  # add the end of text token, e.g. 50256 for gpt2 bpe

            # insert dump_token_id at end of sequence
            domain_token_id = dd_token_ids[example["url"]]
            dump_token_id = dd_token_ids[example["dump"]]
            ids.append(dump_token_id)
            ids.append(domain_token_id)


            out = {"ids": ids, "len": len(ids)}
            return out

        # tokenize the dataset
        tokenized = split_dataset.map(
            process,
            remove_columns=["text", "dump", "url"],
            desc="tokenizing the splits",
            num_proc=num_proc,
        )
        print("----> dataset tokenized")

        
        # concatenate all the ids in each dataset into one large file we can use for training
        for split, dset in tokenized.items():
            arr_len = np.sum(dset["len"])
            filename = os.path.join(FINEWEB10_DATA_PATH, f"{split}.bin")
            dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
            arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
            total_batches = min(1024, len(dset))

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
                # Batch together samples for faster write
                batch = dset.shard(
                    num_shards=total_batches, index=batch_idx, contiguous=True
                ).with_format("numpy")
                arr_batch = np.concatenate(batch["ids"])
                # Write into mmap
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()

    return {'train': os.path.join(FINEWEB10_DATA_PATH, 'train.bin'), 'val': os.path.join(FINEWEB10_DATA_PATH, 'val.bin')}

# 9,318,992,613 training tokens and 1,036,331,430 validation tokens = 10,355,324,043 total tokens (without counting dump tokens)
# 9,344,323,311 training tokens and 1,039,147,285 validation tokens = 10,383,470,596 total tokens (counting dump tokens)
# Note: 
# 10,383,470,596 tokens / 512 tokens per sequence = 20,263,476 sequences
#
# 95 dumps
