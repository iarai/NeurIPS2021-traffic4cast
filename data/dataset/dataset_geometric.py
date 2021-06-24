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
import logging
import os.path
import os.path as osp
import pickle
from multiprocessing import Pool
from pathlib import Path
from typing import Dict
from typing import Optional
from typing import Tuple

import h5py
import numpy as np
import torch
import torch_geometric
import tqdm
from torch import Tensor

from competition.competition_constants import MAX_TEST_SLOT_INDEX
from competition.prepare_test_data.prepare_test_data import prepare_within_day_indices_for_ground_truth
from data.graph_utils import reconstruct_graph
from util.h5_util import load_h5_file

# TODO make namedtuple?
CityGraphInformation = dict
Node = Tuple[int, int]
NodeToIntMapping = Dict[Node, int]


# https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html#creating-larger-datasets
# https://github.com/mie-lab/traffic4cast-Graph-ResNet
class T4CGeometricDataset(torch_geometric.data.Dataset):
    def __init__(self, root, transform=None, pre_transform=None, file_filter: str = None, limit: Optional[int] = None, num_workers: int = 1):
        """torch geometric data set. Processes 8ch.h5 files upon first access
        into `processed` subfolder (one .pt file per training file).

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
        self.networkx_city_graphs: Dict[str, CityGraphInformation] = {}
        if file_filter is None:
            file_filter = "**/*8ch.h5"
        self.file_filter = file_filter
        self.limit = limit
        super(T4CGeometricDataset, self).__init__(root, transform, pre_transform)

        # let's preprocess, as multiple workers might end up in concurrency problems :-(
        for raw_path in tqdm.tqdm(self.raw_paths, desc="process_city_graph_information"):
            filename = os.path.basename(raw_path)
            city = filename.split("_")[1]
            if city not in self.networkx_city_graphs:
                # check if static data was already processed
                city_graph_information: CityGraphInformation = T4CGeometricDataset.process_city_graph_information(
                    city=city, processed_dir=self.processed_dir, raw_dir=self.raw_dir
                )
                self.networkx_city_graphs[city] = city_graph_information
        logging.warning(f"num_workers={num_workers}")
        with Pool(processes=num_workers) as pool:
            for _ in tqdm.tqdm(pool.imap_unordered(self.process_raw_path, self.raw_paths), total=len(self.raw_paths), desc="process_raw_path"):
                pass

    @property
    def raw_dir(self):
        return Path(self.root) / "raw"

    @property
    def processed_dir(self):
        return Path(self.root) / "processed" / "graphs"

    @property  # noqa
    def raw_file_names(self):
        filtered_files = glob.glob(f"{self.root}/raw/{self.file_filter}", recursive=True)
        filtered_files = [f.replace(f"{self.root}/raw/", "") for f in filtered_files]
        return filtered_files

    @property
    def num_classes(self):
        r"""Alias for :py:attr:`~num_node_features`."""
        return self[0].y.shape[-1]

    def process_raw_path(self, raw_path):
        filename = os.path.basename(raw_path)
        city = filename.split("_")[1]
        # check if static data is in cache
        if city not in self.networkx_city_graphs:
            # check if static data was already processed
            city_graph_information: CityGraphInformation = T4CGeometricDataset.process_city_graph_information(
                city=city, processed_dir=self.processed_dir, raw_dir=self.raw_dir
            )

            self.networkx_city_graphs[city] = city_graph_information

        # check if file was already preprocessed
        if osp.exists(osp.join(self.processed_dir, city, filename)):
            return

        # get dynamic data
        dynamic_data = load_h5_file(raw_path)

        # get static data from cache
        city_graph_information = self.networkx_city_graphs[city]
        x = self.image_to_graph(city_graph_information=city_graph_information, dynamic_data=dynamic_data)

        Path(f"{self.processed_dir}/{city}").mkdir(parents=True, exist_ok=True)

        filename = os.path.basename(raw_path)
        filename_proc = osp.join(self.processed_dir, city, filename)

        f_target = h5py.File(filename_proc, "w")
        f_target.create_dataset("array", shape=x.shape, chunks=(1, x.shape[1], 8), dtype="uint8", data=x, compression="lzf")

        f_target.close()

    @staticmethod
    def image_to_graph(city_graph_information: CityGraphInformation, dynamic_data: np.ndarray) -> np.ndarray:
        """

        Parameters
        ----------
        city_graph_information
        dynamic_data

        Returns
        -------

        """
        node_to_int_mapping: NodeToIntMapping = city_graph_information["node_to_int_mapping"]
        # This could also be vectorized to greatly increase performance
        node_coords = np.array(list(node_to_int_mapping.keys()))

        num_nodes = len(node_coords)
        x = dynamic_data[:, node_coords[:, 0], node_coords[:, 1], :]
        expected_shape = (dynamic_data.shape[0], num_nodes, 8)
        assert x.shape == expected_shape, f"found {x.shape}, expected {expected_shape}"
        return x

    @staticmethod
    def process_city_graph_information(city: str, processed_dir: Path, raw_dir: Path) -> CityGraphInformation:
        city_graph_information_pkl = processed_dir / f"{city}_graph.pkl"
        if city_graph_information_pkl.exists():
            with city_graph_information_pkl.open("rb") as pkl:
                city_graph_information = pickle.load(pkl)
        else:
            static_file = raw_dir / city / f"{city}_static.h5"
            city_graph_information = T4CGeometricDataset._process_city_graph_information(city=city, static_file=static_file, processed_dir=processed_dir)
        return city_graph_information

    @staticmethod
    def _process_city_graph_information(city: str, static_file: Path, processed_dir: Path) -> CityGraphInformation:
        G = reconstruct_graph(load_h5_file(static_file), city=city)
        # Every node in the graph has its pixel coordinates as ID (tuple).
        # This assigns every node id tuple and integer (by using enumerate) and stores them in a dict of the form {Node_id_tuple: i}.
        node_to_int_mapping: Dict[Node, int] = {v: k for k, v in enumerate(G.nodes())}
        edge_index: Tensor = torch.tensor([[node_to_int_mapping[n] for n, _ in G.edges], [node_to_int_mapping[n] for _, n in G.edges]], dtype=torch.long)
        # switch axis and concatenate to account for both directions:
        edge_index_ = torch.cat((edge_index[1:, :], edge_index[0:1, :]), axis=0)
        edge_index = torch.cat((edge_index, edge_index_), axis=1)
        city_graph_information = {"edge_index": edge_index, "node_to_int_mapping": node_to_int_mapping, "G": G}
        processed_dir.mkdir(exist_ok=True, parents=True)
        with (processed_dir / f"{city}_graph.pkl").open("wb") as pkl:
            pickle.dump(city_graph_information, pkl)
        return city_graph_information

    def len(self):
        size_240_slots_a_day = len(self.raw_file_names) * MAX_TEST_SLOT_INDEX
        if self.limit is not None:
            return min(size_240_slots_a_day, self.limit)
        return size_240_slots_a_day

    def get(self, idx):
        """Load process data and extract time slot for this data set item.

        Called from called from `__getitem__` in parent!
        """
        file_idx = idx // MAX_TEST_SLOT_INDEX
        filename = self.raw_file_names[file_idx]

        city = os.path.basename(filename).split("_")[1]

        # static data
        self.networkx_city_graphs[city] = T4CGeometricDataset.process_city_graph_information(city=city, processed_dir=self.processed_dir, raw_dir=self.raw_dir)

        edge_index = self.networkx_city_graphs[city]["edge_index"]

        # dynamic_data
        within_day_offset = idx % MAX_TEST_SLOT_INDEX

        ground_truth_offsets = prepare_within_day_indices_for_ground_truth(within_day_offset)

        if not os.path.exists(filename):
            self.process_raw_path(self.raw_paths[file_idx])

        processed_filename = os.path.join(self.processed_dir, city, os.path.basename(filename))

        f = h5py.File(processed_filename, "r")
        dynamic_data = f.get("array")

        x = dynamic_data[within_day_offset : within_day_offset + 12, :, :]
        y = dynamic_data[ground_truth_offsets, :, :]

        data = self.graph_to_data_object(x=x, y=y, edge_index=edge_index)
        return data

    @staticmethod
    def graph_to_data_object(x: np.ndarray, y: np.ndarray, edge_index: torch.Tensor) -> torch_geometric.data.Data:
        input_len = 12
        num_channels = x.shape[-1]
        num_nodes = x.shape[1]
        output_len = 6

        x = torch.from_numpy(x) / 255
        y = torch.from_numpy(y)
        # collapse timestamps into channels
        x = torch.moveaxis(x, 1, 0)
        x = x.reshape(num_nodes, input_len * num_channels)
        y = torch.moveaxis(y, 1, 0)
        y = y.reshape(num_nodes, output_len * num_channels)
        data = torch_geometric.data.Data(x=x.float(), edge_index=edge_index, y=y.float())
        return data


class GraphTransformer:
    """Transform between `T4CDataset` and model."""

    def __init__(self, processed_dir: str, raw_dir: str, batch_size):
        self.processed_dir = Path(processed_dir)
        self.networkx_city_graphs = {}
        self.batch_size = batch_size
        self.raw_dir = Path(raw_dir)

    def graph_to_image(self, graph_data, city):
        squeeze = False
        if graph_data.ndim == 3:
            squeeze = True
            graph_data = np.expand_dims(graph_data, 0)

        # time channels will stay collapsed in the last dimension
        # (:, 153504, 48) -> (:, 495, 436, 48))
        # get static data if not cached

        self.load_cache_city(city)

        node_to_int_mapping = self.networkx_city_graphs[city]["node_to_int_mapping"]
        node_coords = np.array(list(node_to_int_mapping.keys()))

        image_data = np.zeros((self.batch_size, 495, 436, graph_data.shape[-1]))
        image_data[:, node_coords[:, 0], node_coords[:, 1], :] = graph_data
        if squeeze:
            image_data = np.squeeze(image_data)
        return image_data

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

    def load_cache_city(self, city) -> CityGraphInformation:
        if city not in self.networkx_city_graphs:
            self.networkx_city_graphs[city] = T4CGeometricDataset.process_city_graph_information(
                city=city, processed_dir=self.processed_dir, raw_dir=self.raw_dir
            )
        return self.networkx_city_graphs[city]

    @staticmethod
    def pre_transform(test_data: np.ndarray, gt, city, **kwargs) -> torch_geometric.data.Data:
        static_city_information = gt.load_cache_city(city)
        input_data = T4CGeometricDataset.image_to_graph(city_graph_information=static_city_information, dynamic_data=test_data[0])
        data_obj = T4CGeometricDataset.graph_to_data_object(
            x=input_data, edge_index=static_city_information["edge_index"], y=np.zeros(shape=(6, *input_data.shape[1:]))
        )
        return data_obj

    @staticmethod
    def post_transform(prediction_part_graph, gt, city, **kwargs) -> np.ndarray:
        a = prediction_part_graph.to("cpu")
        b = a.numpy()
        prediction_part_image = gt.graph_to_image(b, city)

        return gt.unstack_prediction(prediction_part_image)
