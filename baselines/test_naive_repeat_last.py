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
import torch

from baselines.naive_repeat_last import NaiveRepeatLast


def test_naive_repeat_last_batch():
    x = torch.randint(low=0, high=256, size=(8, 12, 495, 436, 8))
    m = NaiveRepeatLast()
    y = m.forward(x)
    expected_shape = (8, 6, 495, 436, 8)
    assert y.shape == expected_shape, f"expected shape {expected_shape}, actual {y.shape}"
    for i in range(y.shape[0]):
        for j in range(x.shape[1] - 1):
            assert not (x[i, j] == x[i, -1]).all(), f"input data not properly random"
        for j in range(expected_shape[1]):
            assert (x[i, 11] == y[i, j]).all()
