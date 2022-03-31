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
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import torch
from overrides import overrides
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import SubsetRandomSampler

from t4c21.util.h5_util import load_h5_file

MAX_TEST_SLOT_INDEX = 240  # since a test goes over 2 hours, the latest possibility is 10p.m. However, `22*12 > 256 = 2^8` and so does not fit into uint8. Therefore, we (somewhat arbitrarily) chose to start the last test slot at 8-10p.m.


def prepare_test(data: np.ndarray, offset=0, to_torch: bool = False) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    """Extracts an hour of test data for one hour and ground truth prediction
    5,10,15,30,45 and 60 minutes into the future.

    Parameters
    ----------

    data
        tensor of shape (24+, 495, 436, 8) of  type uint8
    offset
    to_torch:bool
        convert to torch float tensor.

    Returns
    -------
        test_data
            tensor of shape (12, 495, 436, 8) of  type uint8
        ground_truth_prediction
            tensor of shape (6, 495, 436, 8) of  type uint8
    """
    offsets = prepare_within_day_indices_for_ground_truth(offset)

    if isinstance(data, torch.Tensor):
        data = data.numpy()

    ub = offset + 12
    model_input = data[offset:ub]
    model_output = data[offsets]
    if to_torch:
        model_input = torch.from_numpy(model_input).float()
        model_output = torch.from_numpy(model_output).float()
    return model_input, model_output


def prepare_within_day_indices_for_ground_truth(offset: int) -> np.ndarray:
    """

    Parameters
    ----------
    offset: int

    Returns
    -------
        the 6 indices for the prediction horizon, i.e. offset+12, offset+13, ...., offset+23
    """
    return np.add([1, 2, 3, 6, 9, 12], 11 + offset)


class T4CDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        file_filter: str = None,
        limit: Optional[int] = None,
        transform: Optional[Callable[[torch.Tensor, bool], torch.Tensor]] = None,
        use_npy: bool = False,
        static_data: Dict[str, List[str]] = None,
    ):
        """torch dataset from training data.

        Parameters
        ----------
        root_dir
            data root folder, by convention should be `data/raw`, see `data/README.md`. All `**/training/*8ch.h5` will be added to the dataset.
        file_filter: str
            filter files under `root_dir`, defaults to `"**/training/*ch8.h5`
        limit
            truncate dataset size
        transform
            transform applied to both the input and label
        static_data:
            per-city static data of shape `(495, 436, channels)` per file.
        """
        self.root_dir = root_dir
        self.limit = limit
        self.files = []
        self.file_filter = file_filter
        self.use_npy = use_npy
        if self.file_filter is None:
            self.file_filter = "**/training/*8ch.h5"
            if self.use_npy:
                self.file_filter = "**/training_npy/*.npy"
        self.transform = transform
        self._load_dataset()
        self.static_data = None
        if static_data is not None:
            self.static_data = {}
            for city, h5_files in static_data.items():
                self.static_data[city] = torch.cat([load_h5_file(p, to_torch=True) for p in h5_files], dim=2)

    def _load_dataset(self):
        self.files = list(Path(self.root_dir).rglob(self.file_filter))

    def _load_h5_file(self, fn, sl: Optional[slice]):
        if self.use_npy:
            return np.load(fn)
        else:
            return load_h5_file(fn, sl=sl)

    def __len__(self):
        size_240_slots_a_day = len(self.files) * MAX_TEST_SLOT_INDEX
        if self.limit is not None:
            return min(size_240_slots_a_day, self.limit)
        return size_240_slots_a_day

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        if idx > self.__len__():
            raise IndexError("Index out of bounds")

        file_idx = idx // MAX_TEST_SLOT_INDEX
        start_hour = idx % MAX_TEST_SLOT_INDEX

        two_hours = self._load_h5_file(self.files[file_idx], sl=slice(start_hour, start_hour + 12 * 2 + 1))

        input_data, output_data = prepare_test(two_hours)

        input_data = self._to_torch(input_data)
        output_data = self._to_torch(output_data)

        if self.static_data is None:
            static_data = torch.zeros(size=(495, 436, 0))
        else:
            city = self.files[file_idx].name.split("_")[1].upper()
            static_data = self.static_data[city]

        return (input_data, static_data), output_data

    def _to_torch(self, data):
        data = torch.from_numpy(data)
        data = data.to(dtype=torch.float)
        return data


class T4CDataModule(LightningDataModule):
    def __init__(
        self,
        val_train_split: float,
        batch_size: Dict[str, int],
        num_workers: int,
        dataset_cls: type,
        dataset_parameters: Dict[str, str],
        dataloader_config: Dict[str, str],
        *args,  # noqa
        **kwargs,  # noqa
    ):
        super().__init__()

        self.val_train_split = val_train_split
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset_cls = dataset_cls
        self.dataset_parameters = dataset_parameters
        self.dataloader_config = dataloader_config

    @overrides
    def setup(self, *args, **kwargs) -> None:
        self.dataset = self.dataset_cls(**self.dataset_parameters)

        full_dataset_size = len(self.dataset)

        indices = list(range(full_dataset_size))
        np.random.shuffle(indices)
        num_train_items = max(int(np.floor(self.val_train_split * full_dataset_size)), self.batch_size["train"],)
        num_val_items = max(int(np.floor((1 - self.val_train_split) * full_dataset_size)), self.batch_size["val"],)

        self.train_indices, self.dev_indices = (
            indices[:num_train_items],
            indices[num_train_items : num_train_items + num_val_items],
        )

    @overrides
    def train_dataloader(self) -> DataLoader:
        train_sampler = SubsetRandomSampler(self.train_indices)

        train_loader = DataLoader(
            self.dataset, batch_size=self.batch_size["train"], num_workers=self.num_workers, sampler=train_sampler, **self.dataloader_config,
        )
        return train_loader

    @overrides
    def val_dataloader(self) -> DataLoader:
        dev_sampler = SubsetRandomSampler(self.dev_indices)
        val_loader = DataLoader(self.dataset, batch_size=self.batch_size["val"], num_workers=self.num_workers, sampler=dev_sampler, **self.dataloader_config,)
        return val_loader
