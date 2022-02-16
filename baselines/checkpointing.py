import datetime

import torch
from torch.nn import DataParallel


def load_torch_model_from_checkpoint(checkpoint: str, model: torch.nn.Module):
    map_location = None
    if not torch.cuda.is_available():
        map_location = "cpu"

    checkpoint = torch.load(checkpoint, map_location=map_location)
    # if run with data parallel, wrap
    if isinstance(checkpoint, DataParallel):
        model = DataParallel(model)

    if "state_dict" in dir(checkpoint) and callable(checkpoint.state_dict):
        # plain model checkpoint
        return model.load_state_dict(checkpoint.state_dict())
    else:
        # checkpoint saved by `save_torch_model_to_checkpoint` below
        return model.load_state_dict(checkpoint["model"])


def save_torch_model_to_checkpoint(model: torch.nn.Module, model_str: str, epoch: int):
    tstamp = datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d%H%M")
    outpath = f"{model_str}_{epoch:04}_{tstamp}.pt"

    save_dict = {"epoch": epoch, "model": model.state_dict()}

    torch.save(save_dict, outpath)
