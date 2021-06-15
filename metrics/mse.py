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
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from torch.nn.functional import mse_loss as torch_mse


def _torch_mse(actual: np.ndarray, expected: np.ndarray, mask: Optional[np.ndarray] = None, mask_norm: bool = True, indices: Optional[List] = None):
    # The torch mse below is significantly faster than (np.square(np.subtract(actual, expected))).mean(axis=axis)
    # Results from performance comparison:
    # %timeit mse(prediction, ground_truth_prediction)
    # 17.9 ms ± 168 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    # %timeit mse(prediction, ground_truth_prediction, use_np=True)
    # 70.1 ms ± 1.93 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    # %timeit mse(prediction, ground_truth_prediction, mask=static_mask)
    # 52.4 ms ± 1.15 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    # %timeit mse(prediction, ground_truth_prediction, mask=static_mask, use_np=True)
    # 112 ms ± 1.08 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    actual_i = torch.from_numpy(actual[:]).float()
    expected_i = torch.from_numpy(expected[:]).float()
    if indices is not None:
        actual_i = actual_i[..., indices]
        expected_i = expected_i[..., indices]
    if mask is not None:
        mask_i = torch.from_numpy(mask[:]).float()
        if indices is not None:
            mask_i = mask_i[..., indices]
        actual_i = actual_i * mask_i
        expected_i = expected_i * mask_i
        if mask_norm:
            return torch_mse(expected_i, actual_i).numpy() / mask_ratio(mask)
    return torch_mse(expected_i, actual_i).numpy()


def _np_mse(
    actual: np.ndarray,
    expected: np.ndarray,
    mask: Optional[np.ndarray] = None,
    mask_norm: bool = True,
    axis: Optional[Tuple] = None,
    indices: Optional[List] = None,
):
    if indices is not None:
        actual = actual[..., indices]
        expected = expected[..., indices]
    if mask is not None:
        actual = actual * mask
        expected = expected * mask
    actual_i = actual.astype(np.float)
    expected_i = expected.astype(np.float)
    if mask is not None and mask_norm:
        return (np.square(np.subtract(actual_i, expected_i))).mean(axis=axis) / mask_ratio(mask)
    return (np.square(np.subtract(actual_i, expected_i))).mean(axis=axis)


def mse(
    actual: np.ndarray,
    expected: np.ndarray,
    mask: Optional[np.ndarray] = None,
    mask_norm: bool = True,
    axis: Optional[Tuple] = None,
    indices: Optional[List] = None,
    use_np: bool = False,
):
    if axis is not None or use_np:
        return _np_mse(actual, expected, mask, mask_norm, axis, indices)
    return _torch_mse(actual, expected, mask, mask_norm, indices)


def mask_ratio(mask):
    return np.count_nonzero(mask) / mask.size
