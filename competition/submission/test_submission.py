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
import tempfile
from pathlib import Path

import numpy as np

from util.h5_util import load_h5_file
from util.h5_util import write_data_to_h5


def test_assign_reload_floats():
    data = np.random.random(size=(5, 5, 5)) * 520 - 255
    assert data.dtype != np.uint8
    assigned = np.zeros(shape=(5, 5, 5), dtype=np.uint8)
    assigned[:] = np.clip(data, 0, 255)
    assert assigned.dtype == np.uint8
    too_large = np.argwhere(data > 255)
    too_small = np.argwhere(data < 0)
    with tempfile.TemporaryDirectory() as temp_dir:
        myh5 = Path(temp_dir) / "my.h5"
        write_data_to_h5(data, filename=myh5)
        reloaded = load_h5_file(myh5)
        for k in list(too_small) + list(too_large):
            print(f"{k}: data={data[k[0], k[1], k[2]]} - reloaded={reloaded[k[0], k[1], k[2]]} - assigned={assigned[k[0], k[1], k[2]]}")
        assert (reloaded == assigned).all(), f"assigned={assigned}, reloaded={reloaded}"
