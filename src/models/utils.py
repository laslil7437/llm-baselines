import torch
from .llama import Llama, RMSNorm
from .base import GPTBase, LayerNorm as BaseLayerNorm
from .gpt2dumps import gpt2dumps, LayerNorm as GPT2DumpsLayerNorm
from .gpt2domains import gpt2domains, LayerNorm as GPT2DomainsLayerNorm


BLACKLIST_WEIGHT_MODULES = (
    torch.nn.LayerNorm,
    BaseLayerNorm,
    GPT2DumpsLayerNorm,
    GPT2DomainsLayerNorm,
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
    elif args.model == 'gpt2dumps':
        model = gpt2dumps(args)
        return model
    elif args.model == 'gpt2domains':
        model = gpt2domains(args)
        return model
    else:
        raise KeyError(f"Unknown model '{args.model}'.")
