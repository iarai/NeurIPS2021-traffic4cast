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
import networkx as nx
import numpy as np

from data.data_layout import offset_map


def reconstruct_graph(data: np.ndarray, city: str) -> nx.Graph:
    """Reconstructs the graph as specified by the connectivity layers: adds an
    edge for all neighbors specified by the connectivity layers 1,..,8.

    Parameters
    ----------
    data: np.ndarray
        static data `(9, 495, 436)`
    city: str
        city name (goes into the `name` attribute of the `nx.Graph`.

    Returns
    -------
    """
    assert data.shape == (9, 495, 436), f"{data.shape}"
    assert data.dtype == np.uint8, f"{data.dtype}"
    g_reconstructed = nx.Graph(name=city)
    offsets = list(offset_map.values())

    for i in range(8):
        start_coordinates = np.argwhere(data[i + 1, ...] > 0)
        start_coordinates = [(r, c) for r, c in start_coordinates]
        dr, dc = offsets[i]
        end_coordinates = [(r + dr, c + dc) for r, c in start_coordinates]
        g_reconstructed.add_edges_from(zip(start_coordinates, end_coordinates))
    return g_reconstructed


def poor_man_graph_visualization(g: nx.Graph) -> np.ndarray:  # noqa
    """Represent the graph as a high-res image 4950x4360. This allows to easily
    focus down on a pixel area of interest.

    Parameters
    ----------
    g: nx.Graph

    Returns
    -------
        image as `np.ndarray`
    """
    height = 4950
    width = 4360

    im = np.zeros(shape=(height, width))
    for n in g.nodes:
        r, c = n[:2]  # Only unpack the coordinates and not additional data
        for dr in [0, 1]:
            for dc in [0, 1]:
                im[r * 10 + 4 + dr, c * 10 + 4 + dc] = 255
    for e in g.edges:
        (r1, c1) = e[0][:2]  # Only unpack the coordinates and not additional data
        (r2, c2) = e[1][:2]  # Only unpack the coordinates and not additional data
        # horizontal right
        if r1 == r2 and c1 < c2:
            offsets = [(0, c) for c in range(10)]
        # horizontal left
        elif r1 == r2 and c1 > c2:
            offsets = [(0, -c) for c in range(10)]
        # vertical down
        elif c1 == c2 and r1 < r2:
            offsets = [(r, 0) for r in range(10)]
        # vertical up
        elif c1 == c2 and r1 > r2:
            offsets = [(-r - 1, 0) for r in range(10)]
        # diagonal right down
        elif r1 < r2 and c1 < c2:
            offsets = [(r, r) for r in range(10)]
        # diagonal left up
        elif r1 > r2 and c1 > c2:
            offsets = [(-r - 1, -r - 1) for r in range(10)]
        # diagonal left down
        elif r1 < r2 and c1 > c2:
            offsets = [(r, -r) for r in range(10)]
        # diagonal right up
        elif r1 > r2 and c1 < c2:
            offsets = [(-r, r) for r in range(10)]
        else:
            raise Exception(f"{(r1, c1), (r2, c2)} {g.name}")

        for offset in offsets:
            dr, dc = offset
            r = r1 * 10 + dr + 4
            c = c1 * 10 + dc + 4
            if r >= 4950 or c >= 4360 or r < 0 or c < 0:
                continue
            im[r, c] = 255
    return im
