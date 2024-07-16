# slightly modified from base.py to include special tokens for domains and dumps of FineWeb10 dataset
# also masks special tokens in the loss calculation

"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect

import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.sequence_length, config.sequence_length))
                                        .view(1, 1, config.sequence_length, config.sequence_length))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    

class GPT2FWDomainMasking(nn.Module):
    # GPT-2 for the FineWeb10 dataset that includes dump and domain special tokens
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.sequence_length is not None
        self.config = config
        gpt2base = tiktoken.get_encoding("gpt2")
        self.tokenizer = tiktoken.Encoding(
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

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.sequence_length, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, get_logits=False):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.sequence_length, f"Cannot forward sequence of length {t}, block size is only {self.config.sequence_length}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            if self.training:
                # consider special tokens only during training
                loss = F.cross_entropy(self.lm_head(x.view(-1, x.size(-1))), targets.view(-1))
            else: 
                # mask dump/domain special tokens during final evaluation
                special_tokens = torch.arange(50257, 50532) # all special tokens 
                special_token_mask = torch.isin(targets, special_tokens.to(device))
                targets[special_token_mask] = -1
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None
        logits = logits if get_logits else None
        return {'logits': logits, 'loss': loss}

    def crop_sequence_length(self, sequence_length):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert sequence_length <= self.config.sequence_length
        self.config.sequence_length = sequence_length
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:sequence_length])
        for block in self.transformer.h:
            block.attn.bias = block.attn.bias[:,:,:sequence_length,:sequence_length]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        # TODO
        pass

    def get_parameter_group_specs(self):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        # need to do import here to avoid circular import (since llama imports from base here)
        from .utils import BLACKLIST_WEIGHT_MODULES

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, BLACKLIST_WEIGHT_MODULES):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
        # will appear in the no_decay and decay sets respectively after the above.
        # In addition, because named_parameters() doesn't return duplicates, it
        # will only return the first occurence, key'd by 'transformer.wte.weight', below.
        # so let's manually remove 'lm_head.weight' from decay set. This will include
        # this tensor into optimization via transformer.wte.weight only, and not decayed.
        decay.remove("lm_head.weight")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        return [
            {"params": sorted(list(decay))},
            {"params": sorted(list(no_decay)), "weight_decay": 0.0},
        ]


    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at sequence_length
            idx_cond = idx if idx.size(1) <= self.config.sequence_length else idx[:, -self.config.sequence_length:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond, get_logits=True)['logits']
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    
    @torch.no_grad()
    def generate_from_string(self, in_str, max_new_tokens, temperature=1.0, top_k=None):
        idx = torch.tensor(self.tokenizer.encode(in_str, allowed_special="all")).view(1,-1).to(self.lm_head.weight.device) # changed to allow all special tokens
        out_idx = self.generate(idx, max_new_tokens, temperature, top_k).view(-1).to('cpu').numpy()
        return self.tokenizer.decode(out_idx)
