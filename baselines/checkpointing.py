import datetime
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Union

import torch
from torch.nn import DataParallel


def load_torch_model_from_checkpoint(checkpoint: Union[str, Path], model: torch.nn.Module, map_location: str = None) -> torch.nn.Module:
    if not torch.cuda.is_available():
        map_location = "cpu"
    state_dict = torch.load(checkpoint, map_location=map_location)
    if isinstance(state_dict, DataParallel):
        logging.debug("     [torch-DataParallel]:")
        state_dict = state_dict.state_dict()
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k[:7] == "module.":
                k = k[7:]  # remove `module.` if trained with data parallelism
            new_state_dict[k] = v
        state_dict = new_state_dict
    elif isinstance(state_dict, dict) and "model" in state_dict:
        # Is that what ignite does?
        logging.debug("     [torch-model-attr]:")
        state_dict = state_dict["model"]
    elif isinstance(state_dict, dict) and "state_dict" in state_dict:
        # That's what lightning does.
        logging.debug("     [torch-state_dict-attr]:")
        state_dict = state_dict["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k[:6] == "model.":
                k = k[6:]  # remove `module.` if trained with data parallelism
            new_state_dict[k] = v
        state_dict = new_state_dict
    model.load_state_dict(state_dict)
    return model


def save_torch_model_to_checkpoint(model: torch.nn.Module, model_str: str, epoch: int):
    tstamp = datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d%H%M")
    outpath = f"{model_str}_{epoch:04}_{tstamp}.pt"

    save_dict = {"epoch": epoch, "model": model.state_dict()}

    torch.save(save_dict, outpath)
