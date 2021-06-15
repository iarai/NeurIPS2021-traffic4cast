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

from baselines.unet import UNetTransfomer


def test_dataset_stack_channels_on_time():
    input_data = torch.randint(0, 256, size=(12, 495, 436, 8))
    input_data = UNetTransfomer.transform_stack_channels_on_time(input_data, batch_dim=False)
    assert input_data.shape == (12 * 8, 495, 436)
    output_data = torch.randint(0, 256, size=(6, 495, 436, 8))
    output_data = UNetTransfomer.transform_stack_channels_on_time(output_data, batch_dim=False)
    assert output_data.shape == (6 * 8, 495, 436)


def test_transform_stack_channels_on_time():
    data = torch.randint(255, dtype=torch.uint8, size=(12, 495, 436, 8))
    transformed_data = UNetTransfomer.transform_stack_channels_on_time(data, batch_dim=False)
    assert transformed_data.shape == (12 * 8, 495, 436)
    for t in range(12):
        for ch in range(8):
            index = t * 8 + ch
            assert np.allclose(data[t, :, :, ch], transformed_data[index]), f"t={t} ch={ch} index={index}"
    unstacked_data = UNetTransfomer.transform_unstack_channels_on_time(transformed_data)
    assert unstacked_data.shape == (12, 495, 436, 8)
    for t in range(12):
        for ch in range(8):
            index = t * 8 + ch
            assert np.allclose(transformed_data[index], unstacked_data[t, :, :, ch]), f"t={t} ch={ch} index={index}"
    assert torch.sum(data - unstacked_data) == 0


def test_transform_unstack_channels_on_time():
    data = torch.randint(255, dtype=torch.uint8, size=(12 * 8, 495, 436))
    unstacked_data = UNetTransfomer.transform_unstack_channels_on_time(data, batch_dim=False)
    unstacked_data.numpy()
    assert unstacked_data.shape == (12, 495, 436, 8)
    for t in range(12):
        for ch in range(8):
            index = t * 8 + ch
            assert np.allclose(data[index], unstacked_data[t, :, :, ch]), f"t={t} ch={ch} index={index}"


def test_unet_pretransform():
    data = torch.randint(255, size=(96, 495, 436))
    conformed_data = UNetTransfomer.unet_pre_transform(data, batch_dim=False)
    assert type(conformed_data) is torch.Tensor
    assert conformed_data.shape == (96, 495, 436)

    conformed_data = UNetTransfomer.unet_pre_transform(data, zeropad2d=(6, 6, 1, 0), batch_dim=False)
    assert type(conformed_data) is torch.Tensor
    assert conformed_data.shape == (96, 496, 448)
    unconformed_data = UNetTransfomer.unet_post_transform(conformed_data, crop=(6, 6, 1, 0))
    assert type(unconformed_data) is torch.Tensor
    assert unconformed_data.shape == (96, 495, 436)

    conformed_data = UNetTransfomer.unet_pre_transform(data, zeropad2d=(6, 6, 1, 0))
    assert type(conformed_data) is torch.Tensor
    assert conformed_data.shape == (96, 496, 448)
    unconformed_data = UNetTransfomer.unet_post_transform(conformed_data, crop=(6, 6, 1, 0))
    assert type(unconformed_data) is torch.Tensor
    assert unconformed_data.shape == (96, 495, 436)
