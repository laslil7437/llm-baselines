######### for debugging purposes only #########

from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset, Dataset
import os
import shutil

FINEWEBMINI_DATA_PATH = os.path.join(os.path.dirname(__file__), "datasets/finewebmini/")

tknzr = tiktoken.get_encoding("gpt2")

def get_finewebmini_data(num_proc=40):

    if not os.path.exists(os.path.join(FINEWEBMINI_DATA_PATH, "train.bin")):
        os.makedirs(FINEWEBMINI_DATA_PATH, exist_ok=True)
        
        dataset = load_dataset("HuggingFaceFW/fineweb", split="train", name="sample-10BT")
        dataset = dataset.select_columns(["text", "dump"])
        dataset = dataset.take(500)
        print("----> dataset loaded")

        split_dataset = dataset.train_test_split(
            test_size=0.1, seed=2357, shuffle=True
        )
        split_dataset["val"] = split_dataset.pop("test")
        print("----> dataset split")

       
        seq_length = 512
        # special token for each dump
        dump_tokens = set(example["dump"] for example in dataset)
        dump_token_ids = {token: i+50256+1 for i, token in enumerate(dump_tokens)}

        def process(example):
            ids = tknzr.encode_ordinary(
                example["text"]
            )  # encode_ordinary ignores any special tokens
            ids.append(
                tknzr.eot_token
            )  # add the end of text token, e.g. 50256 for gpt2 bpe

            # insert dump_token_id every 512 tokens
            dump_token_id = dump_token_ids[example["dump"]]
            for i in range(0, len(ids), seq_length):
                ids.insert(i, dump_token_id)

            out = {"ids": ids, "len": len(ids)}
            return out

        # tokenize the dataset
        tokenized = split_dataset.map(
            process,
            remove_columns=["text", "dump"],
            desc="tokenizing the splits",
            num_proc=num_proc,
        )
        

        # concatenate all the ids in each dataset into one large file we can use for training
        for split, dset in tokenized.items():
            arr_len = np.sum(dset["len"])
            filename = os.path.join(FINEWEBMINI_DATA_PATH, f"{split}.bin")
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
        
    return {'train': os.path.join(FINEWEBMINI_DATA_PATH, 'train.bin'), 'val': os.path.join(FINEWEBMINI_DATA_PATH, 'val.bin')}

    


        
