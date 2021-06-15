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
import itertools

offset_map = {"N": (-1, 0), "NE": (-1, 1), "E": (0, 1), "SE": (1, 1), "S": (1, 0), "SW": (1, -1), "W": (0, -1), "NW": (-1, -1)}
layer_indices_from_offset = {v: i + 1 for i, v in enumerate(offset_map.values())}  # noqa

channel_labels = list(itertools.chain.from_iterable([[f"volume_{h}", f"speed_{h}"] for h in ["NE", "NW", "SE", "SW"]])) + ["incidents"]
static_channel_labels = ["base_map"] + [f"connectivity_{d}" for d in offset_map.keys()]


volume_channel_indices = [ch for ch, l in enumerate(channel_labels) if "volume" in l]
speed_channel_indices = [ch for ch, l in enumerate(channel_labels) if "speed" in l]
