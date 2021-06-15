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
import numpy as np
import torch


class NaiveWeightedAverage(torch.nn.Module):  # noqa
    def __init__(self):
        """Returns prediction consisting of a weighted average of the last
        hour."""
        super(NaiveWeightedAverage, self).__init__()
        # Initialize the weights for averaging the 12 input time frames
        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  equivalent to repeat last
        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  repeat avg of last two
        # [1, 1, 1, 1, 1, 2, 2, 3, 3, 5, 7, 9]  weighted steep
        # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  equivalent to normal mean
        # [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]  weighted flat
        # [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5, 6]  weighted medium
        self.weights = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5, 6]

    def forward(self, x: torch.Tensor, *args, **kwargs):
        x = x.numpy()

        x = np.average(x, axis=1, weights=self.weights)
        x = np.expand_dims(x, axis=1)
        x = np.repeat(x, repeats=6, axis=1)
        # Convert the float values from the average operation back to uint8
        x = x.astype(np.uint8)
        # Set all speeds to 0 where there is no volume in the corresponding heading
        x[:, :, :, :, 1] = x[:, :, :, :, 1] * (x[:, :, :, :, 0] > 0)
        x[:, :, :, :, 3] = x[:, :, :, :, 3] * (x[:, :, :, :, 2] > 0)
        x[:, :, :, :, 5] = x[:, :, :, :, 5] * (x[:, :, :, :, 4] > 0)
        x[:, :, :, :, 7] = x[:, :, :, :, 7] * (x[:, :, :, :, 6] > 0)

        x = torch.from_numpy(x).float()
        return x
