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
        dataset = dataset.take(400)
        print("----> dataset loaded")

        split_dataset = dataset.train_test_split(
            test_size=0.1, seed=2357, shuffle=True
        )
        split_dataset["val"] = split_dataset.pop("test")
        print("----> dataset split")

        def process(example):
            ids = tknzr.encode_ordinary(
                example["text"]
            )  # encode_ordinary ignores any special tokens
            ids.append(
                tknzr.eot_token
            )  # add the end of text token, e.g. 50256 for gpt2 bpe
            out = {"ids": ids, "len": len(ids), "dump": example["dump"]}
            return out

        

        # tokenize the dataset
        tokenized = split_dataset.map(
            process,
            remove_columns=["text"],
            desc="tokenizing the splits",
            num_proc=num_proc,
        )

        all_train_data = os.path.join(FINEWEBMINI_DATA_PATH, "train.bin")
        all_val_data = os.path.join(FINEWEBMINI_DATA_PATH, "val.bin")

        # concatenate all the ids in each dataset into one large file we can use for training
        for split, dset in tokenized.items():

            dset_per_dump = dset.to_polars().groupby('dump')
            for dump, dset_dump in dset_per_dump:

                dset_dump_pd = dset_dump.to_pandas()
                dset_dump = Dataset.from_pandas(dset_dump_pd)

                arr_len = np.sum(dset_dump["len"]) # number of tokens in a given dump

                foldername = os.path.join(FINEWEBMINI_DATA_PATH, dump)
                os.makedirs(foldername, exist_ok=True)
                filename = os.path.join(foldername, f"{split}.bin")

                #filename = os.path.join(FINEWEBMINI_DATA_PATH, f"{dump}_{split}.bin")

                dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
                arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
                total_batches = min(1024, len(dset_dump))

                idx = 0
                for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
                    # Batch together samples for faster write
                    batch = dset_dump.shard(
                        num_shards=total_batches, index=batch_idx, contiguous=True
                    ).with_format("numpy")
                    arr_batch = np.concatenate(batch["ids"])
                    # Write into mmap
                    arr[idx : idx + len(arr_batch)] = arr_batch
                    idx += len(arr_batch)
                arr.flush()
                if filename.endswith("train.bin"):
                    with open(all_train_data, 'ab') as outfile:
                        with open(os.path.join(FINEWEBMINI_DATA_PATH, filename), 'rb') as infile:
                            shutil.copyfileobj(infile, outfile)
                elif filename.endswith("val.bin"):
                    with open(all_val_data, 'ab') as outfile:
                        with open(os.path.join(FINEWEBMINI_DATA_PATH, filename), 'rb') as infile:
                            shutil.copyfileobj(infile, outfile)

        # # concatenate all dump_split.bin files into train.bin and val.bin
        # for filename in os.listdir(FINEWEBMINI_DATA_PATH):
        #     if filename.endswith("train.bin"):
        #         with open(all_train_data, 'ab') as outfile:
        #             with open(os.path.join(FINEWEBMINI_DATA_PATH, filename), 'rb') as infile:
        #                 shutil.copyfileobj(infile, outfile)
        #     elif filename.endswith("val.bin"):
        #         with open(all_val_data, 'ab') as outfile:
        #             with open(os.path.join(FINEWEBMINI_DATA_PATH, filename), 'rb') as infile:
        #                 shutil.copyfileobj(infile, outfile)
        
        train_data = np.memmap(all_train_data, dtype=np.uint16, mode="r")
        val_data = np.memmap(all_val_data, dtype=np.uint16, mode="r")

        breakpoint() 

        return {'train': train_data, 'val': val_data}

 



        
