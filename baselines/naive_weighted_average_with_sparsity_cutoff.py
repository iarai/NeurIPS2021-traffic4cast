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

from baselines.naive_weighted_average import NaiveWeightedAverage


class NaiveWeightedAverageWithSparsityCutoff(NaiveWeightedAverage):
    def __init__(self, num_slots_all_volumes_zero_cutoff: int = 0):
        """Returns prediction consisting of a weighted average of the last hour
        without counting sparse cells.

        Parameters
        ----------
        num_slots_all_volumes_zero_cutoff
            number of slots that need to have zero volume in all directions for the pixel to count as sparse and to be zeroed.
        """
        super(NaiveWeightedAverageWithSparsityCutoff, self).__init__()
        self.num_slots_all_volumes_zero_cutoff = num_slots_all_volumes_zero_cutoff

    def forward(self, x: torch.Tensor, *args, **kwargs):
        x = x.numpy()

        # sparse if all volume channels are == 0
        # `sparse.shape == (8, 12, 495, 436)`
        sparse = (x[:, :, :, :, 0] == 0) & (x[:, :, :, :, 2] == 0) & (x[:, :, :, :, 4] == 0) & (x[:, :, :, :, 6] == 0)
        assert np.count_nonzero(sparse) == np.count_nonzero(sparse)

        # `number_of_sparse_slots.shape == (8, 495, 436)`
        number_of_sparse_slots = np.sum(sparse, axis=1)

        # mask: `(8,495,436) -> (8,12,495,436) -> (8, 12, 495, 436, 8)`
        mask = np.where(number_of_sparse_slots >= self.num_slots_all_volumes_zero_cutoff, 0, 1)
        mask = np.expand_dims(mask, axis=1)
        mask = np.repeat(mask, repeats=12, axis=1)
        mask = np.expand_dims(mask, axis=-1)
        mask = np.repeat(mask, repeats=8, axis=-1)

        x = x * mask

        x = torch.from_numpy(x).float()

        return super(NaiveWeightedAverageWithSparsityCutoff, self).forward(x)
