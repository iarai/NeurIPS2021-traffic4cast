import datetime

import torch


def load_torch_model_from_checkpoint(checkpoint: str, model: torch.nn.Module):
    map_location = None
    if not torch.cuda.is_available():
        map_location = "cpu"

    state_dict = torch.load(checkpoint, map_location=map_location)
    model.load_state_dict(state_dict["model"])


def save_torch_model_to_checkpoint(model: torch.nn.Module, model_str: str, epoch: int):
    tstamp = datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d%H%M")
    outpath = f"{model_str}_{epoch:04}_{tstamp}.pt"

    save_dict = {"epoch": epoch, "model": model.state_dict()}

    torch.save(save_dict, outpath)
