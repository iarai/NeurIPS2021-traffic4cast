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

from competition.prepare_test_data.prepare_test_data import prepare_test
from data.data_layout import channel_labels
from data.data_layout import speed_channel_indices
from data.data_layout import static_channel_labels
from data.data_layout import volume_channel_indices


@pytest.mark.parametrize("offset", [0, 5, 6])
def test_prepare_test(offset):
    data = np.random.rand(24 + offset, 495, 436, 8)
    test_data, ground_truth_prediction = prepare_test(data=data, offset=offset)
    assert test_data.shape == (12, 495, 436, 8)
    assert ground_truth_prediction.shape == (6, 495, 436, 8)
    ub = offset + 12
    assert (test_data == data[offset:ub]).all()
    # 5,10,15,30,45 and 60 into the future
    assert (ground_truth_prediction[0] == data[offset + 11 + 5 // 5]).all()
    assert (ground_truth_prediction[1] == data[offset + 11 + 10 // 5]).all()
    assert (ground_truth_prediction[2] == data[offset + 11 + 15 // 5]).all()
    assert (ground_truth_prediction[3] == data[offset + 11 + 30 // 5]).all()
    assert (ground_truth_prediction[4] == data[offset + 11 + 45 // 5]).all()
    assert (ground_truth_prediction[5] == data[offset + 11 + 60 // 5]).all()


def test_channel_labels():
    print(channel_labels)
    print(static_channel_labels)
    print(volume_channel_indices)
    print(speed_channel_indices)
