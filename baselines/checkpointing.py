import torch


def load_torch_model_from_checkpoint(checkpoint: str, model: torch.nn.Module):
    map_location = None
    if not torch.cuda.is_available():
        map_location = "cpu"
    model.load_state_dict(torch.load(checkpoint, map_location=map_location)["model"])
