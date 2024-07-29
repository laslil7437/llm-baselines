from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset, Dataset
import os
import shutil
from collections import defaultdict
from urllib.parse import urlparse
import fasttext
from transformers import pipeline, AutoTokenizer

# classifier
model_name = 'cffl/bert-base-styleclassification-subjective-neutral'
classifier_tknzr = AutoTokenizer.from_pretrained(model_name)
classify = pipeline(task='text-classification', model=model_name, top_k=1)

gpt2base = tiktoken.get_encoding("gpt2")
dd_tknzr = tiktoken.Encoding(
            name = "gpt2_domains",
            pat_str = gpt2base._pat_str,
            mergeable_ranks = gpt2base._mergeable_ranks,
            special_tokens = {
                **gpt2base._special_tokens,
                'edu': 50257, 
                'uk': 50258, 
                'ca': 50259, 
                'au': 50260, 
                'de': 50261, 
                'ie': 50262, 
                'gov': 50263, 
                'nz': 50264, 
                'co': 50265, 
                'lb': 50266, 
                'tv': 50267, 
                'cn': 50268, 
                'dk': 50269, 
                'nl': 50270, 
                'se': 50271, 
                'us': 50272, 
                'ro': 50273, 
                'br': 50274, 
                'eu': 50275, 
                'fr': 50276, 
                'za': 50277, 
                'pk': 50278, 
                'me': 50279, 
                'sg': 50280, 
                'at': 50281, 
                'tk': 50282, 
                'jp': 50283,
                'ke': 50284, 
                'id': 50285, 
                'lv': 50286, 
                'ru': 50287, 
                'it': 50288, 
                'no': 50289, 
                'in': 50290, 
                'sk': 50291, 
                'ae': 50292, 
                'ng': 50293, 
                'be': 50294, 
                'my': 50295, 
                'mk': 50296, 
                'sh': 50297, 
                'lu': 50298, 
                'mil': 50299, 
                'pl': 50300, 
                'tw': 50301, 
                'io': 50302, 
                'mn': 50303, 
                'tr': 50304, 
                'ga': 50305, 
                'hk': 50306, 
                'ch': 50307, 
                'bg': 50308, 
                'vn': 50309, 
                'qa': 50310, 
                'fj': 50311, 
                'ar': 50312, 
                'si': 50313, 
                'fi': 50314, 
                'gr': 50315, 
                'ws': 50316, 
                'hr': 50317, 
                'ph': 50318, 
                'mo': 50319, 
                'es': 50320, 
                'pw': 50321, 
                'ua': 50322, 
                'fm': 50323, 
                'cc': 50324, 
                'pt': 50325, 
                'hu': 50326, 
                'gq': 50327, 
                'kr': 50328, 
                'vc': 50329, 
                'cl': 50330, 
                'bb': 50331, 
                'cz': 50332, 
                'np': 50333, 
                'gi': 50334, 
                'is': 50335, 
                'ug': 50336, 
                'ir': 50337, 
                'ps': 50338, 
                'ai': 50339, 
                'ge': 50340, 
                'kg': 50341, 
                'lt': 50342, 
                'bd': 50343, 
                'la': 50344, 
                'li': 50345, 
                'pe': 50346, 
                'om': 50347, 
                'mx': 50348, 
                'ml': 50349, 
                'eg': 50350, 
                'th': 50351, 
                'mt': 50352, 
                'to': 50353, 
                'az': 50354, 
                'ee': 50355, 
                'kz': 50356, 
                'lk': 50357, 
                'am': 50358, 
                'zw': 50359, 
                'cf': 50360, 
                'il': 50361, 
                'do': 50362, 
                'tz': 50363, 
                'ly': 50364, 
                'im': 50365, 
                'cy': 50366, 
                'mu': 50367, 
                'ag': 50368, 
                'ky': 50369, 
                'rw': 50370, 
                'cr': 50371, 
                'ec': 50372, 
                'sa': 50373, 
                'jm': 50374, 
                'rs': 50375, 
                'tt': 50376, 
                'mm': 50377, 
                'cm': 50378, 
                'su': 50379, 
                'af': 50380, 
                'je': 50381, 
                'nu': 50382, 
                'by': 50383, 
                'gh': 50384, 
                've': 50385, 
                'ma': 50386, 
                'gd': 50387, 
                'bm': 50388, 
                'sb': 50389, 
                'uz': 50390, 
                'st': 50391, 
                'gg': 50392, 
                'na': 50393, 
                'bs': 50394, 
                'bz': 50395, 
                'cx': 50396, 
                'tm': 50397, 
                'pr': 50398, 
                'ba': 50399, 
                'al': 50400, 
                'kh': 50401, 
                'md': 50402, 
                'ac': 50403, 
                'mv': 50404, 
                'kw': 50405, 
                'va': 50406, 
                'gm': 50407, 
                're': 50408, 
                'bt': 50409, 
                'sc': 50410, 
                'fo': 50411, 
                'iq': 50412, 
                'cd': 50413, 
                'et': 50414, 
                'cu': 50415, 
                'jo': 50416, 
                'sv': 50417, 
                'tl': 50418, 
                'bw': 50419, 
                'vu': 50420, 
                'gs': 50421, 
                'so': 50422, 
                'gy': 50423, 
                'uy': 50424, 
                'ms': 50425, 
                'mw': 50426, 
                'bn': 50427, 
                'zm': 50428, 
                'dj': 50429, 
                'tn': 50430, 
                'pn': 50431, 
                'pg': 50432, 
                'bh': 50433, 
                'as': 50434, 
                'vg': 50435, 
                'other': 50436,
                'CC-MAIN-2013-20':	50437,
                'CC-MAIN-2013-48':	50438,
                'CC-MAIN-2014-10':	50439,
                'CC-MAIN-2014-15':	50440,
                'CC-MAIN-2014-23':	50441,
                'CC-MAIN-2014-35':	50442,
                'CC-MAIN-2014-41':	50443,
                'CC-MAIN-2014-42':	50444,
                'CC-MAIN-2014-49':	50445,
                'CC-MAIN-2014-52':	50446,
                'CC-MAIN-2015-06':	50447,
                'CC-MAIN-2015-11':	50448,
                'CC-MAIN-2015-14':	50449,
                'CC-MAIN-2015-18':	50450,
                'CC-MAIN-2015-22':	50451,
                'CC-MAIN-2015-27':	50452,
                'CC-MAIN-2015-32':	50453,
                'CC-MAIN-2015-35':	50454,
                'CC-MAIN-2015-40':	50455,
                'CC-MAIN-2015-48':	50456,
                'CC-MAIN-2016-07':	50457,
                'CC-MAIN-2016-18':	50458,
                'CC-MAIN-2016-22':	50459,
                'CC-MAIN-2016-26':	50460,
                'CC-MAIN-2016-30':	50461,
                'CC-MAIN-2016-36':	50462,
                'CC-MAIN-2016-40':	50463,
                'CC-MAIN-2016-44':	50464,
                'CC-MAIN-2016-50':	50465,
                'CC-MAIN-2017-04':	50466,
                'CC-MAIN-2017-09':	50467,
                'CC-MAIN-2017-13':	50468,
                'CC-MAIN-2017-17':	50469,
                'CC-MAIN-2017-22':	50470,
                'CC-MAIN-2017-26':	50471,
                'CC-MAIN-2017-30':	50472,
                'CC-MAIN-2017-34':	50473,
                'CC-MAIN-2017-39':	50474,
                'CC-MAIN-2017-43':	50475,
                'CC-MAIN-2017-47':	50476,
                'CC-MAIN-2017-51':	50477,
                'CC-MAIN-2018-05':	50478,
                'CC-MAIN-2018-09':	50479,
                'CC-MAIN-2018-13':	50480,
                'CC-MAIN-2018-17':	50481,
                'CC-MAIN-2018-22':	50482,
                'CC-MAIN-2018-26':	50483,
                'CC-MAIN-2018-30':	50484,
                'CC-MAIN-2018-34':	50485,
                'CC-MAIN-2018-39':	50486,
                'CC-MAIN-2018-43':	50487,
                'CC-MAIN-2018-47':	50488,
                'CC-MAIN-2018-51':	50489,
                'CC-MAIN-2019-04':	50490,
                'CC-MAIN-2019-09':	50491,
                'CC-MAIN-2019-18':	50492,
                'CC-MAIN-2019-22':	50493,
                'CC-MAIN-2019-26':	50494,
                'CC-MAIN-2019-30':	50495,
                'CC-MAIN-2019-35':	50496,
                'CC-MAIN-2019-39':	50497,
                'CC-MAIN-2019-43':	50498,
                'CC-MAIN-2019-47':	50499,
                'CC-MAIN-2019-51':	50500,
                'CC-MAIN-2020-05':	50501,
                'CC-MAIN-2020-10':	50502,
                'CC-MAIN-2020-16':	50503,
                'CC-MAIN-2020-24':	50504,
                'CC-MAIN-2020-29':	50505,
                'CC-MAIN-2020-34':	50506,
                'CC-MAIN-2020-40':	50507,
                'CC-MAIN-2020-45':	50508,
                'CC-MAIN-2020-50':	50509,
                'CC-MAIN-2021-04':	50510,
                'CC-MAIN-2021-10':	50511,
                'CC-MAIN-2021-17':	50512,
                'CC-MAIN-2021-21':	50513,
                'CC-MAIN-2021-25':	50514,
                'CC-MAIN-2021-31':	50515,
                'CC-MAIN-2021-39':	50516,
                'CC-MAIN-2021-43':	50517,
                'CC-MAIN-2021-49':	50518,
                'CC-MAIN-2022-05':	50519,
                'CC-MAIN-2022-21':	50520,
                'CC-MAIN-2022-27':	50521,
                'CC-MAIN-2022-33':	50522,
                'CC-MAIN-2022-40':	50523,
                'CC-MAIN-2022-49':	50524,
                'CC-MAIN-2023-06':	50525,
                'CC-MAIN-2023-14':	50526,
                'CC-MAIN-2023-23':	50527,
                'CC-MAIN-2023-40':	50528,
                'CC-MAIN-2023-50':	50529,
                'CC-MAIN-2024-10':	50530,
                'CC-MAIN-2019-13':	50531,
                'SUBJECTIVE': 50532,
                'NEUTRAL': 50533,
            }
        )

def modify_data_domains(dataset):
    
    # hard-coded list of ccTLDs with >= 100 entries, plus edu/mil/gov
    valid_domains = ['edu','uk','ca','au','de','ie','gov','nz','co','lb','tv','cn','dk',
        'nl','se','us','ro','br','eu','fr','za','pk','me','sg','at','tk','jp','ke','id',
        'lv','ru','it','no','in','sk','ae','ng','be','my','mk','sh','lu','mil','pl','tw',
        'io','mn','tr','ga','hk','ch','bg','vn','qa','fj','ar','si','fi','gr','ws','hr',
        'ph','mo','es','pw','ua','fm','cc','pt','hu','gq','kr','vc','cl','bb','cz','np',
        'gi','is','ug','ir','ps','ai','ge','kg','lt','bd','la','li','pe','om','mx','ml',
        'eg','th','mt','to','az','ee','kz','lk','am','zw','cf','il','do','tz','ly','im',
        'cy','mu','ag','ky','rw','cr','ec','sa','jm','rs','tt','mm','cm','su','af','je',
        'nu','by','gh','ve','ma','gd','bm','sb','uz','st','gg','na','bs','bz','cx','tm',
        'pr','ba','al','kh','md','ac','mv','kw','va','gm','re','bt','sc','fo','iq','cd',
        'et','cu','jo','sv','tl','bw','vu','gs','so','gy','uy','ms','mw','bn','zm','dj',
        'tn','pn','pg','bh','as','vg'] 
    
    # edit "url" column to just be domain, or "other"
    def process_url(example):
        parsed_url = urlparse(example["url"])
        domain = parsed_url.netloc
        ending = domain.split('.')[-1]
        if ending not in valid_domains:
            example["url"] = "other"
        else:
            example["url"] = ending
        return example
    dataset = dataset.map(process_url)
    print("----> dataset modified")
    return dataset 

def get_fwclassifier_data(args, num_proc=40):
    global FINEWEB10_DATA_PATH 
    FINEWEB10_DATA_PATH = os.path.join(os.path.dirname(__file__), f"datasets/{args.filename}/") 

    if not os.path.exists(os.path.join(FINEWEB10_DATA_PATH, "train.bin")):
        os.makedirs(FINEWEB10_DATA_PATH, exist_ok=True)
        
        dataset = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT")
        dataset = dataset.select_columns(["text", "dump", "url"])
        print("----> dataset loaded")

        dataset = modify_data_domains(dataset) 
        
        # classification
        def truncate_classify(example):
            tokens = classifier_tknzr(example["text"], truncation=True, max_length=512)
            truncated_text = classifier_tknzr.decode(tokens['input_ids'], skip_special_tokens=True)
            label = classify(truncated_text)[0][0]['label']
            example["label"] = label
            return example
        dataset = dataset.map(truncate_classify)
        print("----> dataset truncated and labeled")

        split_dataset = dataset["train"].train_test_split(
            test_size=0.1, seed=2357, shuffle=True
        )
        split_dataset["val"] = split_dataset.pop("test")
        print("----> dataset split")

        def process(example):
            ids = dd_tknzr.encode_ordinary(
                example["text"]
            )  # encode_ordinary ignores any special tokens
            ids.append(
                dd_tknzr.eot_token
            )  # add the end of text token, e.g. 50256 for gpt2 bpe

            ids.insert(0, dd_tknzr.encode(example["dump"], allowed_special="all")[0])
            ids.insert(1, dd_tknzr.encode(example["url"], allowed_special="all")[0])

            # add the label token at start
            ids.insert(2, dd_tknzr.encode(example["label"], allowed_special="all")[0])
            
            out = {"ids": ids, "len": len(ids)}
            return out

        # tokenize the dataset
        tokenized = split_dataset.map(
            process,
            remove_columns=["text", "dump", "url", "label"],
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