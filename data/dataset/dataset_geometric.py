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
import glob
import os.path
import os.path as osp
import pickle
from os import makedirs
from pathlib import Path

import h5py
import numpy as np
import torch
import torch_geometric
import tqdm
from overrides import overrides

from competition.competition_constants import MAX_TEST_SLOT_INDEX
from competition.prepare_test_data.prepare_test_data import prepare_within_day_indices_for_ground_truth
from data.graph_utils import reconstruct_graph
from util.h5_util import load_h5_file


# https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html#creating-larger-datasets
# https://github.com/mie-lab/traffic4cast-Graph-ResNet
class T4CGeometricDataset(torch_geometric.data.Dataset):
    def __init__(self, root, transform=None, pre_transform=None, file_filter: str = "**/*8ch.h5"):
        """
        torch geometric data set. Processes all files in advance from `raw` subfolder into `processed` subfolder (one .pt file per training file).
        TODO: maybe this should be done on the fly? cache only the static data? Offer an option for that.

        Parameters
        ----------
        root
            data root folder, by convention should be `data` (without `raw`!), see `data/README.md`. All `**/training/*8ch.h5` will be added to the dataset.
        file_filter: str
            Defaults to "**/*8ch.h5".
        transform
            see :obj:`torch_geometric.data.Dataset`
        pre_transform
            see :obj:`torch_geometric.data.Dataset`
        """
        self.networkx_city_graphs = {}
        self.file_filter = file_filter
        super(T4CGeometricDataset, self).__init__(root, transform, pre_transform)

    @overrides  # noqa
    def _process(self):
        try:
            makedirs(self.processed_dir)
        except FileExistsError:
            pass

        self.process()

    @property
    def processed_dir(self):
        return osp.join(self.root, "processed", "graphs")

    @property  # noqa
    def raw_file_names(self):
        filtered_files = glob.glob(f"{self.root}/raw/{self.file_filter}", recursive=True)
        filtered_files = [f.replace(f"{self.root}/raw/", "") for f in filtered_files]
        return filtered_files

    @property
    def processed_file_names(self):
        filtered_files = glob.glob(f"{self.processed_dir}/**/{self.file_filter}", recursive=True)
        return filtered_files

    @property
    def num_classes(self):
        r"""Alias for :py:attr:`~num_node_features`."""
        return self[0].y.shape[-1]

    def process(self):  # noqa
        r"""Processes the dataset to the :obj:`self.processed_dir` folder. One `.pt` file per training file. The slot extraction will be done in `get`."""
        for _, raw_path in enumerate(tqdm.tqdm(self.raw_paths)):

            filename = os.path.basename(raw_path)
            city = filename.split("_")[1]
            # check if static data is in cache
            if city not in self.networkx_city_graphs:
                # check if static data was already processed
                city_graph_information = T4CGeometricDataset.process_city(city=city, processed_dir=self.processed_dir, raw_dir=self.raw_dir)

                self.networkx_city_graphs[city] = city_graph_information

            # check if file was already preprocessed
            if osp.exists(osp.join(self.processed_dir, city, filename)):
                continue

            # get dynamic data
            dynamic_data = load_h5_file(raw_path)

            # get static data from cache
            node_to_int_mapping = self.networkx_city_graphs[city]["node_to_int_mapping"]

            # This can also be vectorized to greatly increase performance
            node_coords = np.array(list(node_to_int_mapping.keys()))
            x = dynamic_data[:, node_coords[:, 0], node_coords[:, 1], :]

            if self.pre_filter is not None and not self.pre_filter(x):
                continue

            if self.pre_transform is not None:
                x = self.pre_transform(x)

            Path(f"{self.processed_dir}/{city}").mkdir(parents=True, exist_ok=True)

            filename = os.path.basename(raw_path)
            filename_proc = osp.join(self.processed_dir, city, filename)

            f_target = h5py.File(filename_proc, "w")
            f_target.create_dataset("array", shape=x.shape, chunks=(1, x.shape[1], 8), dtype="uint8", data=x, compression="lzf")

            f_target.close()

    @staticmethod
    def process_city(city, processed_dir, raw_dir):
        try:
            city_graph_information = pickle.load(open(osp.join(processed_dir, city + "_graph.pkl"), "rb"))
        except FileNotFoundError:
            static_file = osp.join(f"{raw_dir}", city, city + "_static.h5")
            city_graph_information = T4CGeometricDataset._process_city(city=city, static_file=static_file, processed_dir=processed_dir)
        return city_graph_information

    @staticmethod
    def _process_city(city: str, static_file: str, processed_dir: str):
        G = reconstruct_graph(load_h5_file(static_file), city=city)
        # Every node in the graph has its pixel coordinates as ID (tuple).
        # This assigns every node id tuple and integer (by using enumerate) and stores them in a dict of the
        # form {Node_id_tuple: i}.
        # This should be cached as well
        node_to_int_mapping = {v: k for k, v in enumerate(G.nodes())}
        # I think this has to be reversed as well (e.g., you are storing only 1 direction at the moment).
        edge_index = torch.tensor([[node_to_int_mapping[n] for n, _ in G.edges], [node_to_int_mapping[n] for _, n in G.edges]], dtype=torch.long)
        # switch axis and concatenate to account for both directions:
        edge_index_ = torch.cat((edge_index[1:, :], edge_index[0:1, :]), axis=0)
        edge_index = torch.cat((edge_index, edge_index_), axis=1)
        # we could also save the rest of the graph data but we do not really need it...
        city_graph_information = {"edge_index": edge_index, "node_to_int_mapping": node_to_int_mapping, "G": G}
        pickle.dump(city_graph_information, open(osp.join(processed_dir, city + "_graph.pkl"), "wb"))
        return city_graph_information

    def len(self):

        return len(self.processed_file_names) * MAX_TEST_SLOT_INDEX

    def get(self, idx):
        """Load process data and extract time slot for this data set item.

        Called from called from `__getitem__` in parent!
        """
        file_idx = idx // MAX_TEST_SLOT_INDEX
        filename = self.processed_file_names[file_idx]

        city = os.path.basename(filename).split("_")[1]

        # static data
        if city not in self.networkx_city_graphs:
            self.networkx_city_graphs[city] = pickle.load(open(osp.join(self.processed_dir, city + "_graph.pkl"), "rb"))
        edge_index = self.networkx_city_graphs[city]["edge_index"]

        # dynamic_data
        within_day_offset = idx % MAX_TEST_SLOT_INDEX

        ground_truth_offsets = prepare_within_day_indices_for_ground_truth(within_day_offset)

        f = h5py.File(filename, "r")
        dynamic_data = f.get("array")

        x = torch.from_numpy(dynamic_data[within_day_offset : within_day_offset + 12, :, :]) / 255
        y = torch.from_numpy(dynamic_data[ground_truth_offsets, :, :])

        num_nodes = dynamic_data.shape[1]
        # collapse timestamps into channels
        # todo: this reshaping operation is actually really dangerous as you have to make sure that the data is not
        #  scrampled while doing it. It usually requires a move axis so that the "constant" dimension is up front
        x = torch.moveaxis(x, 1, 0)
        x = x.reshape(num_nodes, 12 * x.shape[-1])

        y = torch.moveaxis(y, 1, 0)
        y = y.reshape(num_nodes, len(ground_truth_offsets) * y.shape[-1])

        data = torch_geometric.data.Data(x=x.float(), edge_index=edge_index, y=y.float())

        return data

    @staticmethod
    def pre_transform(test_data, gt, city, device, **kwargs):
        input_data_part = torch.tensor(gt.image_to_graph(test_data, city)).to(device).float()[0]
        data_obj = gt.graph_to_data_object(input_data_part, city).to(device)
        return data_obj

    @staticmethod
    def post_transform(prediction_part_graph, gt, city, **kwargs):
        a = prediction_part_graph.to("cpu")
        b = a.numpy()
        prediction_part_image = gt.graph_to_image(b, city)
        return gt.unstack_prediction(prediction_part_image)


class GraphTransformer:
    """Transform between `T4CDataset` and model.

    TODO: This is a basically a cheap version of a data loader for the test data.
    """

    def __init__(self, processed_dir, raw_dir, batch_size):
        self.processed_dir = processed_dir
        self.networkx_city_graphs = {}
        self.raw_dir = raw_dir
        self.batch_size = batch_size

    def image_to_graph(self, image_data, city):
        # get static data if not cached
        self.load_cache_city(city)

        # copy from dataset_geometric
        node_to_int_mapping = self.networkx_city_graphs[city]["node_to_int_mapping"]

        # This could be vectorized to greatly increase performance
        node_coords = np.array(list(node_to_int_mapping.keys()))

        # image data has extra dim for batch size
        graph_data = image_data[:, :, node_coords[:, 0], node_coords[:, 1], :]

        return graph_data

    def graph_to_image(self, graph_data, city):
        squeeze = False
        if graph_data.ndim == 3:
            squeeze = True
            graph_data = np.expand_dims(graph_data, 0)

        # time channels will stay collapsed in the last dimension
        # (:, 153504, 48) -> (:, 495, 436, 48))
        # get static data if not cached
        # todo: maybe a better test for this would be nice, these transformations are cunning

        self.load_cache_city(city)

        node_to_int_mapping = self.networkx_city_graphs[city]["node_to_int_mapping"]
        node_coords = np.array(list(node_to_int_mapping.keys()))

        image_data = np.zeros((self.batch_size, 495, 436, graph_data.shape[-1]))
        image_data[:, node_coords[:, 0], node_coords[:, 1], :] = graph_data
        if squeeze:
            image_data = np.squeeze(image_data)
        return image_data

    def graph_to_data_object(self, graph_data: torch.Tensor, city) -> torch_geometric.data.Data:
        # function needs torch objects as input
        squeeze = False

        if city not in self.networkx_city_graphs:
            self.networkx_city_graphs[city] = pickle.load(open(osp.join(self.processed_dir, city + "_graph.pkl"), "rb"))
        edge_index = self.networkx_city_graphs[city]["edge_index"]

        x = graph_data / 255
        if x.ndim == 3:
            squeeze = True
            x = torch.unsqueeze(x, 0)

        num_nodes = graph_data.shape[-2]

        # (:, 12, 127850, 8) --> (:, 127850, 12, 8)
        x = torch.moveaxis(x, 2, 1)
        # (:, 127850, 12, 8) --> (:, 127850, 48)
        x = x.reshape(self.batch_size, num_nodes, 12 * 8)
        if squeeze:
            x = torch.squeeze(x)
        return torch_geometric.data.Data(x=x.float(), edge_index=edge_index)

    def unstack_prediction(self, x):
        squeeze = False
        if x.ndim == 3:
            squeeze = True
            x = np.expand_dims(x, 0)

        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 6, 8)
        x = torch.moveaxis(torch.tensor(x), 3, 1)
        if squeeze:
            x = torch.squeeze(x)
        # (2, 495, 436, 6, 8) ->  [2, 6, 495, 436, 8]
        return x

    def load_cache_city(self, city):
        if city not in self.networkx_city_graphs:
            self.networkx_city_graphs[city] = T4CGeometricDataset.process_city(city=city, processed_dir=self.processed_dir, raw_dir=self.raw_dir)
