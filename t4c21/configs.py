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
from functools import partial

from t4c21.vanilla_unet import UNet
from t4c21.vanilla_unet import UNetTransfomer

configs = {
    "unet": {
        "model_class": UNet,
        # zeropad2d the input data with 0 to ensure same size after upscaling by the network inputs [495, 436] -> [496, 448]
        "model_config": {"in_channels": 12 * 8, "n_classes": 6 * 8, "depth": 5, "wf": 6, "padding": True, "up_mode": "upconv", "batch_norm": True},
        "dataset_config": {"transform": partial(UNetTransfomer.unet_pre_transform, stack_channels_on_time=True, zeropad2d=(6, 6, 1, 0), batch_dim=False)},
        "pre_transform": partial(UNetTransfomer.unet_pre_transform, stack_channels_on_time=True, zeropad2d=(6, 6, 1, 0), batch_dim=True, from_numpy=True),
        "post_transform": partial(UNetTransfomer.unet_post_transform, stack_channels_on_time=True, crop=(6, 6, 1, 0), batch_dim=True),
    },
}
