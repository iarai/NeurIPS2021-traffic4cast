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

from t4c21.vanilla_unet import UNet


def test_pretransform_nobatch():
    q = torch.rand(495, 436, 8)

    output = UNet.unet_pre_transform(data=torch.rand(12, 495, 436, 8), static_data=q, zeropad2d=None, batch_dim=False)
    assert output.shape == (12 * 8 + 8, 495, 436)


def test_pretransform_batch():

    batch_size = 5

    q = torch.rand(batch_size, 495, 436, 7)

    output = UNet.unet_pre_transform(data=torch.rand(batch_size, 12, 495, 436, 8), static_data=q, zeropad2d=None, batch_dim=True)

    assert output.shape == (batch_size, 12 * 8 + 7, 495, 436)
