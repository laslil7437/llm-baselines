from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset, Dataset
import os
import shutil
from collections import defaultdict
from urllib.parse import urlparse
import fasttext

gpt2base = tiktoken.get_encoding("gpt2")

dd_tknzr = tiktoken.Encoding(
            name = "gpt2_domains",
            pat_str = gpt2base._pat_str,
            mergeable_ranks = gpt2base._mergeable_ranks,
            special_tokens = {
                **gpt2base._special_tokens,
                'uk': 50257,
                'au': 50258,
                'ca': 50259,
                'in': 50260,
                'us': 50261,
                'nz': 50262,
                'de': 50263,
                'ie': 50264,
                'eu': 50265,
                'za': 50266,
                'co': 50267,
                'ru': 50268,
                'nl': 50269,
                'me': 50270,
                'io': 50271,
                'tv': 50272,
                'it': 50273,
                'sg': 50274,
                'fr': 50275,
                'pl': 50276,
                'se': 50277,
                'ch': 50278,
                'tk': 50279,
                'ng': 50280,
                'my': 50281,
                'es': 50282,
                'jp': 50283,
                'ph': 50284,
                'pk': 50285,
                'be': 50286,
                'dk': 50287,
                'cn': 50288,
                'cz': 50289,
                'no': 50290,
                'gr': 50291,
                'at': 50292,
                'fi': 50293,
                'fm': 50294,
                'br': 50295,
                'hk': 50296,
                'ae': 50297,
                'ke': 50298,
                'ro': 50299,
                'cc': 50300,
                'id': 50301,
                'vn': 50302,
                'ua': 50303,
                'il': 50304,
                'tr': 50305,
                'hu': 50306,
                'pt': 50307,
                'tw': 50308,
                'is': 50309,
                'ir': 50310,
                'kr': 50311,
                'ee': 50312,
                'ai': 50313,
                'lk': 50314,
                'to': 50315,
                'ws': 50316,
                'sk': 50317,
                'mx': 50318,
                'si': 50319,
                'az': 50320,
                'cl': 50321,
                'ug': 50322,
                'lt': 50323,
                'lv': 50324,
                'hr': 50325,
                'th': 50326,
                'am': 50327,
                'ga': 50328,
                'zw': 50329,
                'rs': 50330,
                'nu': 50331,
                'ar': 50332,
                'ml': 50333,
                'cf': 50334,
                'bg': 50335,
                'la': 50336,
                'pw': 50337,
                'bd': 50338,
                'gh': 50339,
                'sa': 50340,
                'gq': 50341,
                'lu': 50342,
                'ge': 50343,
                'ly': 50344,
                'mt': 50345,
                'gg': 50346,
                'np': 50347,
                'kz': 50348,
                'eg': 50349,
                'by': 50350,
                'im': 50351,
                'ag': 50352,
                'cy': 50353,
                'md': 50354,
                'lb': 50355,
                'su': 50356,
                'tt': 50357,
                'mk': 50358,
                'na': 50359,
                'st': 50360,
                'pe': 50361,
                'bz': 50362,
                'qa': 50363,
                'tl': 50364,
                'va': 50365,
                'tz': 50366,
                'com': 50367,
                'org': 50368,
                'net': 50369,
                'edu': 50370,
                'gov': 50371,
                'info': 50372,
                'blog': 50373,
                'biz': 50374,
                'xyz': 50375,
                'news': 50376,
                'app': 50377,
                'online': 50378,
                'club': 50379,
                'mil': 50380,
                'int': 50381,
                'pro': 50382,
                'top': 50383,
                'shop': 50384,
                'life': 50385,
                'asia': 50386,
                'live': 50387,
                'site': 50388,
                'today': 50389,
                'travel': 50390,
                'media': 50391,
                'coop': 50392,
                'store': 50393,
                'scot': 50394,
                'website': 50395,
                'tech': 50396,
                'mobi': 50397,
                'space': 50398,
                'world': 50399,
                'london': 50400,
                'jobs': 50401,
                'name': 50402,
                'global': 50403,
                'church': 50404,
                'cat': 50405,
                'art': 50406,
                'dev': 50407,
                'wiki': 50408,
                'nyc': 50409,
                'link': 50410,
                'agency': 50411,
                'aero': 50412,
                'guru': 50413,
                'guide': 50414,
                'design': 50415,
                'wales': 50416,
                'one': 50417,
                'law': 50418,
                'network': 50419,
                'com:443': 50420,
                'digital': 50421,
                'africa': 50422,
                'press': 50423,
                'rocks': 50424,
                'studio': 50425,
                'review': 50426,
                'solutions': 50427,
                'academy': 50428,
                'win': 50429,
                'games': 50430,
                'cloud': 50431,
                'work': 50432,
                'buzz': 50433,
                'reviews': 50434,
                'tips': 50435,
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
            }
        )

def modify_data_domains(dataset):
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

def get_fwtime_data(args, num_proc=40):
    global FINEWEB10_DATA_PATH 
    FINEWEB10_DATA_PATH = os.path.join(os.path.dirname(__file__), f"datasets/{args.filename}/") 

    if not os.path.exists(os.path.join(FINEWEB10_DATA_PATH, "train.bin")):
        os.makedirs(FINEWEB10_DATA_PATH, exist_ok=True)
        
        dataset = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT")

        dataset = dataset.select_columns(["text", "dump", "url"])
        print("----> dataset loaded")

        if args.fw_domains == True:
            dataset = modify_data_domains(dataset) 

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

            if args.token_placement == 'start':
                if args.fw_dumps == True:
                    if args.fw_domains == True:
                        # add both tokens at start
                        ids.insert(0, dd_tknzr.encode(example["dump"], allowed_special="all")[0])
                        ids.insert(1, dd_tknzr.encode(example["url"], allowed_special="all")[0])

            ids.insert(0, dd_tknzr.encode(example["dump"], allowed_special="all")[0])
            
            out = {"ids": ids, "len": len(ids)}
            return out

        # tokenize the dataset
        tokenized = split_dataset.map(
            process,
            remove_columns=["text", "dump", "url"],
            desc="tokenizing the splits",
            #num_proc=num_proc,
        )
        print("----> dataset tokenized")

        # further split the validation set into even thirds, based on first token value between 50437 to 50531

        # val1_dumps = ["CC-MAIN-2013-20", "CC-MAIN-2019-39"]
        # split_dataset["val3"] = split_dataset["val"].filter(lambda x: x["dump"] in val1_dumps)
        tokenized["val1"] = tokenized["val"].filter(lambda x: 50437 <= x["ids"][0] < 50484)
        tokenized["val2"] = tokenized["val"].filter(lambda x: 50484 <= x["ids"][0] < 50508)
        tokenized["val3"] = tokenized["val"].filter(lambda x: 50508 <= x["ids"][0] < 50532)

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
    if args.timeval_num == 0:
        return {'train': os.path.join(FINEWEB10_DATA_PATH, 'train.bin'), 'val': os.path.join(FINEWEB10_DATA_PATH, 'val.bin')}
    elif args.timeval_num == 1:
        return {'train': os.path.join(FINEWEB10_DATA_PATH, 'train.bin'), 'val': os.path.join(FINEWEB10_DATA_PATH, 'val1.bin')}
    elif args.timeval_num == 2:
        return {'train': os.path.join(FINEWEB10_DATA_PATH, 'train.bin'), 'val': os.path.join(FINEWEB10_DATA_PATH, 'val2.bin')}
    elif args.timeval_num == 3:
        return {'train': os.path.join(FINEWEB10_DATA_PATH, 'train.bin'), 'val': os.path.join(FINEWEB10_DATA_PATH, 'val3.bin')}
