"""UNet implementation from https://github.com/jvanvugt/pytorch-unet.

Copied from https://github.com/mie-lab/traffic4cast/blob/aea6f90e8884c01689c84255c99e96d2b58dc470/models/unet.py with permission
"""
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
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import torch
from torch import nn


class UNet(nn.Module):
    @staticmethod
    def unet_pre_transform(
        data: Union[np.ndarray, torch.Tensor],
        static_data: Union[np.ndarray, torch.Tensor],
        zeropad2d: Optional[Tuple[int, int, int, int]] = None,
        batch_dim: bool = False,
        from_numpy: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Transform data from `T4CDataset` be used by UNet:

        - put time and channels into one dimension
        - padding
        """

        if from_numpy:
            data = torch.from_numpy(data).float()

        if not batch_dim:
            data = torch.unsqueeze(data, 0)
            static_data = torch.unsqueeze(static_data, 0)

        # k = data.shape[0] #noqa
        # (k, 12,495,436,8) -> (k, 96, 495, 436)
        data = UNet.transform_stack_channels_on_time(data, batch_dim=True)

        # (k, 495,436,m) -> (k, m, 495, 436)
        static_data = torch.moveaxis(static_data, 3, 1)

        # (k, 96, 495,436) + (k, m, 495, 436) -> (k, 96+m, 495, 436)
        data = torch.cat([data, static_data], dim=1)

        if zeropad2d is not None:
            zeropad2d = torch.nn.ZeroPad2d(zeropad2d)
            data = zeropad2d(data)
        if not batch_dim:
            data = torch.squeeze(data, 0)
        return data

    @staticmethod
    def unet_post_transform(
        data: torch.Tensor,
        crop: Optional[Tuple[int, int, int, int]] = None,
        batch_dim: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Bring data from UNet back to `T4CDataset` format:

        - separates common dimension for time and channels
        - cropping
        """
        if not batch_dim:
            data = torch.unsqueeze(data, 0)

        if crop is not None:
            _, _, height, width = data.shape
            left, right, top, bottom = crop
            right = width - right
            bottom = height - bottom
            data = data[:, :, top:bottom, left:right]
        data = UNet.transform_unstack_channels_on_time(data, batch_dim=True)
        if not batch_dim:
            data = torch.squeeze(data, 0)
        return data

    @staticmethod
    def transform_stack_channels_on_time(data: torch.Tensor, batch_dim: bool = False):
        """
        `(k, 12, 495, 436, 8) -> (k, 12 * 8, 495, 436)`
        """

        if not batch_dim:
            # `(12, 495, 436, 8) -> (1, 12, 495, 436, 8)`
            data = torch.unsqueeze(data, 0)
        num_time_steps = data.shape[1]
        num_channels = data.shape[4]

        # (k, 12, 495, 436, 8) -> (k, 12, 8, 495, 436)
        data = torch.moveaxis(data, 4, 2)

        # (k, 12, 8, 495, 436) -> (k, 12 * 8, 495, 436)
        data = torch.reshape(data, (data.shape[0], num_time_steps * num_channels, 495, 436))

        if not batch_dim:
            # `(1, 12, 495, 436, 8) -> (12, 495, 436, 8)`
            data = torch.squeeze(data, 0)
        return data

    @staticmethod
    def transform_unstack_channels_on_time(data: torch.Tensor, num_channels=8, batch_dim: bool = False):
        """
        `(k, 12 * 8, 495, 436) -> (k, 12, 495, 436, 8)`
        """
        if not batch_dim:
            # `(12, 495, 436, 8) -> (1, 12, 495, 436, 8)`
            data = torch.unsqueeze(data, 0)

        num_time_steps = int(data.shape[1] / num_channels)
        # (k, 12 * 8, 495, 436) -> (k, 12, 8, 495, 436)
        data = torch.reshape(data, (data.shape[0], num_time_steps, num_channels, 495, 436))

        # (k, 12, 8, 495, 436) -> (k, 12, 495, 436, 8)
        data = torch.moveaxis(data, 2, 4)

        if not batch_dim:
            # `(1, 12, 495, 436, 8) -> (12, 495, 436, 8)`
            data = torch.squeeze(data, 0)
        return data

    def __init__(self, in_channels=96, n_classes=48, depth=5, wf=6, padding=True, batch_norm=True, up_mode="upconv", zeropad2d=(6, 6, 1, 0), crop=(6, 6, 1, 0)):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597
        Using the default arguments will yield the exact version used
        in the original paper
        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(UNet, self).__init__()
        assert up_mode in ("upconv", "upsample")
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(UNetConvBlock(prev_channels, 2 ** (wf + i), padding, batch_norm))
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm))
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)
        self.zeropad2d = zeropad2d
        self.crop = crop

    def forward(self, x, *args, **kwargs):
        x = self.unet_pre_transform(data=x[0], static_data=x[1], batch_dim=True, zeropad2d=self.zeropad2d)
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = torch.nn.functional.max_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        x = self.last(x)
        x = self.unet_post_transform(data=x, batch_dim=True, crop=self.crop)
        return x


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):  # noqa
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == "upconv":
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == "upsample":
            self.up = nn.Sequential(
                nn.Upsample(mode="bilinear", scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        diff_y_target_size_ = diff_y + target_size[0]
        diff_x_target_size_ = diff_x + target_size[1]
        return layer[:, :, diff_y:diff_y_target_size_, diff_x:diff_x_target_size_]

    def forward(self, x, bridge):  # noqa
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out
