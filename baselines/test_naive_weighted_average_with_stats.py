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
import tempfile
from pathlib import Path

import numpy as np
import torch

from baselines.naive_weighted_average_from_stats import NaiveWeightedAverageWithStats
from baselines.naive_weighted_average_from_stats import spatio_temporal_cities
from baselines.naive_weighted_average_from_stats import temporal_cities
from competition.competition_constants import MAX_TEST_SLOT_INDEX
from util.h5_util import write_data_to_h5


def test_test_naive_weighted_average_with_stats():
    x = torch.randint(low=0, high=256, dtype=torch.uint8, size=(8, 12, 495, 436, 8))
    additional_data = torch.cat(
        (torch.randint(low=0, high=7, dtype=torch.uint8, size=(8, 1)), torch.randint(low=0, high=MAX_TEST_SLOT_INDEX, dtype=torch.uint8, size=(8, 1))), axis=1
    )
    assert additional_data.shape == (8, 2)
    with tempfile.TemporaryDirectory() as tempdir:
        tempdir_path = Path(tempdir)
        print("creating fake data")
        for d in range(7):
            for city in spatio_temporal_cities + temporal_cities:
                write_data_to_h5(np.random.randint(0, 255, size=(24, 495, 436, 8)), filename=str(tempdir_path / f"{city}_{d}_means.h5"), compression_level=0)
                write_data_to_h5(np.random.randint(0, 12, size=(24, 495, 436, 8)), filename=str(tempdir_path / f"{city}_{d}_zeros.h5"), compression_level=0)
        print("fake data created")

        m = NaiveWeightedAverageWithStats(stats_dir=tempdir_path)

        m(x, city="VIENNA", additional_data=additional_data)
