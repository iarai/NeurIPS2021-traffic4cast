import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import MSELoss

VOL_CHANNELS = [0, 2, 4, 6]
SPEED_CHANNELS = [1, 3, 5, 7]


def mse_loss_wiedemann(
    input: Tensor,
    target: Tensor,
    reduction: str = "mean",
) -> Tensor:
    n = torch.count_nonzero(target[..., VOL_CHANNELS] != 0) + torch.count_nonzero(target[..., VOL_CHANNELS] == 0)
    f = torch.count_nonzero(target[..., VOL_CHANNELS] != 0) + n
    mask = ((target[..., VOL_CHANNELS] != 0)).float()
    target[..., SPEED_CHANNELS] = target[..., SPEED_CHANNELS] * mask
    input[..., SPEED_CHANNELS] = input[..., SPEED_CHANNELS] * mask
    return F.mse_loss(input, target, reduction=reduction) / f * 2 * n


class MSELossWiedemann(MSELoss):
    def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super(MSELossWiedemann, self).__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return mse_loss_wiedemann(input, target, reduction=self.reduction)
