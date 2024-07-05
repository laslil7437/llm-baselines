import torch
from .llama import Llama, RMSNorm
from .base import GPTBase, LayerNorm as BaseLayerNorm
from .fwmodel import GPT2FineWebDD, LayerNorm as FWLayerNorm
from .fwmasking import GPT2FWMasking, LayerNorm as FWMaskingLayerNorm

BLACKLIST_WEIGHT_MODULES = (
    torch.nn.LayerNorm,
    BaseLayerNorm,
    FWLayerNorm,
    FWMaskingLayerNorm,
    RMSNorm,
    torch.nn.Embedding,
)

def get_model(args):
    """ Return the right model """
    if args.model == 'base':
        model = GPTBase(args)
        return model
    elif args.model == 'llama2':
        model = Llama(args)
        return model
    elif args.model == 'gpt2dd':
        model = GPT2FineWebDD(args)
        return model
    elif args.model == 'fwmasking':
        model = GPT2FWMasking(args)
        return model
    else:
        raise KeyError(f"Unknown model '{args.model}'.")
