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

from util.h5_util import load_h5_file


def create_static_mask(static_roads, num_slots=0):
    static_mask_once = np.where(static_roads[0] > 0, 1, 0)
    assert static_mask_once.shape == (495, 436)
    if num_slots > 0:
        static_mask = np.broadcast_to(static_mask_once, (num_slots, 495, 436))
        assert (static_mask[0] == static_mask_once).all()
        static_mask = np.repeat(static_mask, 8).reshape(num_slots, 495, 436, 8)
        assert static_mask.shape == (num_slots, 495, 436, 8), f"{static_mask.shape}"
        assert (static_mask[0, :, :, 0] == static_mask_once).all()
    else:
        static_mask = np.repeat(static_mask_once, 8).reshape(495, 436, 8)
        assert static_mask.shape == (495, 436, 8), f"{static_mask.shape}"
        assert (static_mask[:, :, 0] == static_mask_once).all()
        # Alternatively could also do expand_dims(axis=2) -> repeat(repeats=8, axis=2)
    return static_mask


def get_static_mask(city, base_folder, num_slots=0):  # noqa
    return create_static_mask(load_h5_file(f"{base_folder}/{city}/{city}_static.h5"), num_slots)
