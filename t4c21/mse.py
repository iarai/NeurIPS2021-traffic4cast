#  Copyright 2021 Institute of Advanced Research in Artificial Intelligence (IARAI) GmbH.
#  IARAI licenses this file to You under the Apache License, Version 2.0
#  (the "License"); you may not use this file except in compliance with
#  the License. You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# N.B. partial copy of metrics/mse.py to keep t4c21 self-contained
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
