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

from baselines.graph_models import Graph_resnet
from baselines.naive_all_constant import NaiveAllConstant
from baselines.naive_average import NaiveAverage
from baselines.naive_repeat_last import NaiveRepeatLast
from baselines.naive_weighted_average import NaiveWeightedAverage
from baselines.naive_weighted_average_with_sparsity_cutoff import NaiveWeightedAverageWithSparsityCutoff
from baselines.unet import UNet
from baselines.unet import UNetTransfomer
from data.dataset.dataset_geometric import GraphTransformer

configs = {
    "unet": {
        "model_class": UNet,
        # zeropad2d the input data with 0 to ensure same size after upscaling by the network inputs [495, 436] -> [496, 448]
        "model_config": {"in_channels": 12 * 8, "n_classes": 6 * 8, "depth": 5, "wf": 6, "padding": True, "up_mode": "upconv", "batch_norm": True},
        "dataset_config": {"transform": partial(UNetTransfomer.unet_pre_transform, stack_channels_on_time=True, zeropad2d=(6, 6, 1, 0), batch_dim=False)},
        "pre_transform": partial(UNetTransfomer.unet_pre_transform, stack_channels_on_time=True, zeropad2d=(6, 6, 1, 0), batch_dim=True, from_numpy=True),
        "post_transform": partial(UNetTransfomer.unet_post_transform, stack_channels_on_time=True, crop=(6, 6, 1, 0), batch_dim=True),
    },
    "naive_all_zero": {"model_class": NaiveAllConstant, "model_config": {"fill_value": 0}},
    "naive_all_10": {"model_class": NaiveAllConstant, "model_config": {"fill_value": 10}},
    "naive_all_255": {"model_class": NaiveAllConstant, "model_config": {"fill_value": 255}},
    "naive_repeat_last": {"model_class": NaiveRepeatLast},
    "naive_average": {"model_class": NaiveAverage},
    "naive_weighted_average": {"model_class": NaiveWeightedAverage},
    "naive_weighted_average_with_sparsity_cutoff": {"model_class": NaiveWeightedAverageWithSparsityCutoff},
    "gcn": {
        "model_class": Graph_resnet,
        "model_config": {"nh": 80, "depth": 5, "K": 4, "K_mix": 2, "inout_skipconn": True, "p": 0, "bn": True, "num_features": 96, "num_classes": 48},
        "optimizer_config": {"lr": 0.01, "weight_decay": 0.0001},
        "pre_transform": GraphTransformer.pre_transform,
        "post_transform": GraphTransformer.post_transform,
        "dataloader_config": {"drop_last": True, "shuffle": True},
        "geometric": True,
    },
}
