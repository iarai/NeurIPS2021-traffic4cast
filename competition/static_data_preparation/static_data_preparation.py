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
import argparse
import logging
import re
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import psutil
import tqdm

from data.data_layout import layer_indices_from_offset
from util.h5_util import load_h5_file
from util.h5_util import write_data_to_h5


def build_neighbor_graph(channel_data: np.ndarray, empty_value=0) -> nx.Graph:
    """(3) build "pixel graph" (Moore neighborhood in pixel -> introduce edge)
    from BW renderings (4) aggregation: introduce edge between high-res nodes
    and their corresponding lo-res node.

    Parameters
    ----------
    channel_data
        two-dimensional array
    empty_value

    Returns
    -------
        two-dimensional array
    """
    g = nx.Graph()
    # add edges if both pixels are not empty.
    i_bound = channel_data.shape[0]
    j_bound = channel_data.shape[1]
    # traverse pixels right,right-down,down
    for i in range(i_bound):
        for j in range(j_bound):
            me = (i, j)
            if channel_data[me] == empty_value:
                continue
            for di in [0, 1]:
                for dj in [0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    neighbour_i = i + di
                    neighbour_j = j + dj
                    neighbour = (neighbour_i, neighbour_j)
                    if neighbour_i >= i_bound or neighbour_j >= j_bound:
                        continue
                    if channel_data[neighbour] == empty_value:
                        continue
                    g.add_edge(me, neighbour)
    return g


def build_intermediate_graph(g: nx.Graph) -> nx.DiGraph:
    """(5) add edge in g_aggr at low res if there is a path of length <= 7 in
    the intermediate (corresponds to path length <= 5 in high res). Avoid name
    clashes of the nodes at the two labels.

    Parameters
    ----------
    g:nx.Graph


    Returns
    -------
    """
    factor = 10
    # N.B. we need this to directed in order to prevent bypassing through multiple centroids
    # (we want paths from centroid to centroid with no intermediate centroid in the path)
    g_fine_with_coarse_nodes = nx.DiGraph()
    for n1, n2 in g.edges:
        g_fine_with_coarse_nodes.add_edge(n1, n2)
        g_fine_with_coarse_nodes.add_edge(n2, n1)
    # add directed edges from low res (centroids) to high res nodes
    for n in g.nodes:
        r, c = n
        coarse_n = (r // factor, c // factor, "coarse")
        g_fine_with_coarse_nodes.add_edge(n, coarse_n)
    return g_fine_with_coarse_nodes


def aggregate_graph(g_intermediate: nx.DiGraph, city: str, cutoff: int = 9) -> nx.Graph:
    """(6) add coarse pixels.

    Parameters
    ----------
    g_intermediate
    city
    cutoff

    Returns
    -------
    """
    g_aggr = nx.Graph()
    coarse_nodes = [n for n in g_intermediate.nodes if len(n) == 3]

    for source in coarse_nodes:
        outgoing_edges = [(source, n_fine) for n_fine in g_intermediate.predecessors(source)]
        g_intermediate.add_edges_from(outgoing_edges)
        for target in nx.single_source_shortest_path(g_intermediate, source=source, cutoff=cutoff):
            if len(target) == 3 and source != target:
                g_aggr.add_edge(source[:2], target[:2])
                try:
                    assert source[0] - target[0] in [-1, 0, 1], f"{city} {source}-{target}"
                    assert source[1] - target[1] in [-1, 0, 1], f"{city} {source}-{target}"
                except AssertionError as e:
                    print(nx.shortest_path(g_intermediate, source, target))
                    raise e
        g_intermediate.remove_edges_from(outgoing_edges)
    return g_aggr


def add_all_coarse_nodes(g_aggr: nx.Graph, g_coarse: nx.Graph, city):
    """(7) export grey_scale 100m res as first level and connectitivty of
    g_aggr (8 layers)

    Parameters
    ----------
    g_aggr
    g_coarse
    city
    """
    added = []
    for n_coarse in g_coarse.nodes:
        if n_coarse not in g_aggr.nodes:
            added.append(n_coarse)
    g_aggr.add_nodes_from(added)
    for n_coarse in added:
        r_coarse, c_coarse = n_coarse
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                neighbor = (r_coarse + dr, c_coarse + dc)
                if neighbor not in g_aggr.nodes:
                    continue
                g_aggr.add_edge(n_coarse, neighbor)


def export_static_files(g, layer0, high_res_data, output_folder, city):
    """(8) plausibility checking: neighbor degrees in g_aggr seem to be
    plausible. Notice that the node encode pixel position so we can derive the
    direction and orientation of each edge.

    Parameters
    ----------
    g
    layer0
    high_res_data
    output_folder
    city

    Returns
    -------
    """
    static_content = np.zeros(shape=(9, 495, 436), dtype=np.uint8)

    # take the gray-scale coarse image as layer 0, but use 0 for white (no road)
    static_content[0] = layer0

    for (r1, c1), (r2, c2) in g.edges:
        offset = (r2 - r1, c2 - c1)
        dr, dc = offset
        assert r1 + dr == r2
        assert c1 + dc == c2
        assert offset[0] in [-1, 0, 1], f"{(r1, c1), (r2, c2)}"
        assert offset[1] in [-1, 0, 1], f"{(r1, c1), (r2, c2)}"
        assert offset != (0, 0), f"{(r1, c1), (r2, c2)}"
        opposite_offset = (r1 - r2, c1 - c2)
        dr_, dc_ = opposite_offset
        assert r2 + dr_ == r1
        assert c2 + dc_ == c1

        layer = layer_indices_from_offset[offset]
        layer_opposite = layer_indices_from_offset[opposite_offset]
        static_content[layer][(r1, c1)] = 255
        static_content[layer_opposite][(r2, c2)] = 255

    assert np.count_nonzero(static_content[1:]) == len(g.edges) * 2
    write_data_to_h5(data=static_content, filename=f"{output_folder}/{city.upper()}_static.h5")
    write_data_to_h5(data=high_res_data, filename=f"{output_folder}/{city.upper()}_map_high_res.h5")
    return static_content


def generate_connectivity_layers(city, coarse_city_, fine_city, output_folder):
    # `(3)` / `(4)`
    g_coarse = build_neighbor_graph(coarse_city_, empty_value=255)
    g_fine = build_neighbor_graph(fine_city, empty_value=255)
    # `(5)`
    g_intermediate = build_intermediate_graph(g_fine)
    # `(6)`
    g_aggr = aggregate_graph(g_intermediate, city)
    # `(7)`
    add_all_coarse_nodes(g_aggr, g_coarse, city)
    # `(8)`
    export_static_files(g_aggr, 255 - coarse_city_, 255 - fine_city, output_folder, city)


def main(raw_folder: str, output_folder: str):
    raw_folder = Path(raw_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    files = list(raw_folder.rglob("**/*_static.h5"))
    logging.info(files)
    p_bar = tqdm.tqdm(files)
    for f in p_bar:
        f = str(f)
        city = re.search(r"/([A-Z]+)_static", f).group(1)
        p_bar.set_description(f"{city}, RAM memory used {psutil.virtual_memory()[2]}")
        coarse_city_ = 255 - load_h5_file(f)[0]
        fine_city = 255 - load_h5_file(f.replace("static", "map_high_res"))

        # check that this reproduces the result
        assert coarse_city_.shape == (495, 436)
        assert fine_city.shape == (4950, 4360)
        generate_connectivity_layers(city, coarse_city_, fine_city, output_folder)

        expected = load_h5_file(f)

        out_file = output_folder / f"{city}_static.h5"
        actual = load_h5_file(str(out_file))
        assert (expected == actual).all()


def create_parser() -> argparse.ArgumentParser:
    """Creates the argument parser for this program."""
    parser = argparse.ArgumentParser(description=("This programs creates the connectivity layers."))
    parser.add_argument("--raw_folder", type=str, help="Points to extracted data.", required=True, default="./data/raw")
    parser.add_argument("--output_folder", type=str, help="Where to put the files created", required=False, default=".")
    return parser


if __name__ == "__main__":
    parser = create_parser()
    try:
        params = parser.parse_args()
        main(**vars(params))
    except Exception as e:
        print(f"There was an error during execution, please review: {e}")
        parser.print_help()
        exit(1)
if __name__ == "__main__":
    main(*sys.argv[1:])
