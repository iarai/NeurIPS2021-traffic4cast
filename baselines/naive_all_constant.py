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


class NaiveAllConstant(torch.nn.Module):  # noqa
    def __init__(self, fill_value):
        """Returns prediction consisting of only zero values."""
        self.fill_value = fill_value
        super(NaiveAllConstant, self).__init__()

    def forward(self, x, *args, **kwargs):  # noqa
        output_shape = list(x.shape)
        output_shape[1] = 6
        x = np.full(output_shape, self.fill_value, dtype=np.uint8)
        return x
