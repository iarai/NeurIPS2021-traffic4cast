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
import pytest
import torch

from baselines.naive_weighted_average import NaiveWeightedAverage
from baselines.naive_weighted_average_with_sparsity_cutoff import NaiveWeightedAverageWithSparsityCutoff
from data.data_layout import volume_channel_indices


@pytest.mark.parametrize(
    "num_zero_volume_slots,num_zeroize_cells,num_slots_all_volumes_zero_cutoff,expected_same",
    [(3, 0, 2, True), (3, 20, 2, False), (2, 20, 2, False), (4, 20, 5, True), (5, 20, 5, False), (8, 20, 5, False)],
)
def test_test_naive_weighted_average_with_zero_sparsity_cutoff_is_same_as_naive_weighted_average(
    num_zero_volume_slots: int, num_zeroize_cells: int, num_slots_all_volumes_zero_cutoff: int, expected_same: bool
):
    x = torch.randint(low=0, high=256, size=(8, 12, 495, 436, 8))
    for _ in range(num_zeroize_cells):
        batch_idx = np.random.choice(8)
        r = np.random.choice(495)
        c = np.random.choice(436)
        time_indices = np.random.choice(12, size=num_zero_volume_slots, replace=False)
        for time_idx in time_indices:
            for ch in volume_channel_indices:
                x[batch_idx, time_idx, r, c, ch] = 0

    m = NaiveWeightedAverageWithSparsityCutoff(num_slots_all_volumes_zero_cutoff=num_slots_all_volumes_zero_cutoff)
    y = m.forward(x)
    y_baseline = NaiveWeightedAverage().forward(x)
    expected_shape = (8, 6, 495, 436, 8)
    assert y.shape == expected_shape, f"expected shape {expected_shape}, actual {y.shape}"
    assert np.allclose(y, y_baseline) == expected_same
