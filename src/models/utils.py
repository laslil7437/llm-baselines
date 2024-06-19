import torch
from .llama import Llama, RMSNorm
from .base import GPTBase, LayerNorm
from .gpt2dumps import gpt2dumps, LayerNorm


BLACKLIST_WEIGHT_MODULES = (
    torch.nn.LayerNorm,
    LayerNorm,
    RMSNorm,
    torch.nn.Embedding,
)
BLACKLIST2_WEIGHT_MODULES = [
       # ln's that don't get categorized appropriately in fw10base model
    'transformer.h.9.ln_1.weight', 
    'transformer.h.4.ln_2.weight', 
    'transformer.h.11.ln_2.weight', 
    'transformer.h.9.ln_2.weight', 
    'transformer.ln_f.weight', 
    'transformer.h.8.ln_1.weight', 
    'transformer.h.1.ln_2.weight', 
    'transformer.h.5.ln_1.weight', 
    'transformer.h.3.ln_1.weight', 
    'transformer.h.10.ln_1.weight', 
    'transformer.h.0.ln_1.weight', 
    'transformer.h.2.ln_1.weight', 
    'transformer.h.7.ln_2.weight', 
    'transformer.h.2.ln_2.weight', 
    'transformer.h.6.ln_1.weight', 
    'transformer.h.4.ln_1.weight', 
    'transformer.h.0.ln_2.weight', 
    'transformer.h.5.ln_2.weight', 
    'transformer.h.1.ln_1.weight', 
    'transformer.h.6.ln_2.weight', 
    'transformer.h.7.ln_1.weight', 
    'transformer.h.11.ln_1.weight', 
    'transformer.h.8.ln_2.weight', 
    'transformer.h.10.ln_2.weight', 
    'transformer.h.3.ln_2.weight'
]

def get_model(args):
    """ Return the right model """
    if args.model == 'base':
        model = GPTBase(args)
        return model
    elif args.model == 'llama2':
        model = Llama(args)
        return model
    elif args.model == 'gpt2dumps':
        model = gpt2dumps(args)
        return model
    else:
        raise KeyError(f"Unknown model '{args.model}'.")
