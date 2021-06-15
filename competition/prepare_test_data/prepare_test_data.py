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
import gc
import glob
import json
import logging
import os
import random
import re
import shutil
import tarfile
import tempfile
import zipfile
from typing import Dict
from typing import Tuple
from typing import Union

import numpy as np
import psutil
import torch

from competition.competition_constants import MAX_TEST_SLOT_INDEX
from util.data_range import generate_date_range
from util.data_range import weekday_parser
from util.h5_util import load_h5_file
from util.h5_util import write_data_to_h5
from util.logging import t4c_apply_basic_logging_config
from util.monitoring import disk_usage_human_readable


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


def _generate_test_slots(**kwargs):
    from_date = kwargs["from_date"]
    to_date = kwargs["to_date"]
    n = kwargs["n"]

    test_slots = {}
    dates = generate_date_range(from_date=from_date, to_date=to_date)

    # `slots[date][123]` contains the boolean of whether starting a test at the 123rd time step on date is still possible.
    slots = {date: {t: True for t in range(288)} for date in dates}
    for i in range(n):
        placed = False
        while not placed:
            key = random.choice(dates)
            time = random.randint(0, MAX_TEST_SLOT_INDEX)
            available = True
            for i in range(48):
                if slots[key][time + i] is False:
                    available = False
            if available:
                placed = True
                # update test_slots dict
                if key in test_slots:
                    test_slots[key].append(time)
                else:
                    test_slots[key] = [time]
                # update slots.
                for i in range(48):
                    slots[key][time + i] = False
    return test_slots


def _create_test_manifest(config, test_slot_directory):
    for competition, competition_config in config.items():
        competition_slots = {}
        for city, city_configs in competition_config.items():
            competition_slots.setdefault(city, {})
            for city_config in city_configs:
                slots = _generate_test_slots(**city_config)
                competition_slots[city] = dict(competition_slots[city], **slots)
            competition_slots_sorted = {k: v for k, v in sorted(competition_slots[city].items(), key=lambda t: t[0])}
            competition_slots[city] = competition_slots_sorted
        print(competition)
        print(competition_slots)
        with open(os.path.join(test_slot_directory, f"test_slots_{competition}.json"), "w") as fp:
            json.dump(competition_slots, fp)


def _check_test_manifest(config, test_slot_directory):
    for competition, _ in config.items():
        manifest_file = os.path.join(test_slot_directory, f"test_slots_{competition}.json")
        if not os.path.exists(manifest_file):
            print(f"Test manifest {manifest_file} not found, please use --update_test_manifest")
            return False
    return True


def _check_exists_dir(city_output_directory):
    if not os.path.exists(city_output_directory):
        os.makedirs(city_output_directory)


def _get_contents_from_tarballs_and_directories(dynamic_data_path, check_tars=True, check_dirs=True) -> Dict[Tuple[str, str], Tuple[str, str]]:
    print("Indexing dynamic input data ...")
    city_date_finder = {}
    city_regex = r"/[0-9]{4}-[0-9]{2}-[0-9]{2}_([a-zA-Z]+)_8ch"
    date_regex = r"([0-9]{4}-[0-9]{2}-[0-9]{2})_"
    len_tarballs = 0
    len_extracted = 0
    if check_tars:
        contents_of_tarballs = {}
        tarballs = list(glob.glob(f"{dynamic_data_path}/*.tar"))
        for tarname in tarballs:
            print(tarname)
            with tarfile.open(tarname, "r") as tar:
                for tarinfo in tar:
                    if tarinfo.name.endswith(".h5"):
                        contents_of_tarballs[tarinfo.name] = tarname
        logging.info(f"{len(contents_of_tarballs)} .h5 files in {len(tarballs)} tar balls.")
        city_date_finder_tarballs = {
            (re.search(city_regex, h5).group(1).upper(), re.search(date_regex, h5).group(1)): (h5, tarball) for h5, tarball in contents_of_tarballs.items()
        }
        len_tarballs = len(city_date_finder_tarballs)
        assert len_tarballs == len(contents_of_tarballs)
        city_date_finder.update(city_date_finder_tarballs)
    if check_dirs:
        ch8_files = glob.glob(f"{dynamic_data_path}/**/*8ch.h5", recursive=True)
        logging.info(f"{len(ch8_files)} .h5 files in folders (untarred).")
        city_date_finder_extracted = {(re.search(city_regex, f).group(1).upper(), re.search(date_regex, f).group(1)): (f, None) for f in ch8_files}
        len_extracted = len(city_date_finder_extracted)
        assert len_extracted == len(ch8_files)
        city_date_finder.update(city_date_finder_extracted)
        assert len(city_date_finder) == len_extracted
    if check_tars and check_dirs:
        assert len_tarballs == len_extracted
    if check_tars:
        assert len(city_date_finder) == len_tarballs
    print(f"... found {len(city_date_finder)} input files.")
    return city_date_finder


def _extract_h5_for_city_and_date(city: str, city_date_finder, date: str, output_directory: str):
    h5, tarball = city_date_finder.get((city, date), (None, None))
    if h5 is None:
        return None
    if tarball is None:
        target_filename = os.path.join(output_directory, os.path.basename(h5))
        _check_exists_dir(output_directory)
        shutil.copyfile(h5, target_filename)
        return target_filename
    with tarfile.open(tarball, "r") as tar:
        target_filename = h5.split("/")[-1]
        _check_exists_dir(output_directory)
        target_filename = f"{output_directory}/{target_filename}"
        with open(target_filename, "wb") as out:
            out.write(tar.extractfile(h5).read())
            return target_filename


def _prepare_test_slots_city(
    arg, city_date_finder, competition, expected_number_of_slots, output_directory, output_directory_ground_truth, tempdir,
):
    city, test_slots = arg
    expanded_config = []

    logging.debug("%s %s", city, test_slots)
    # continue
    m = expected_number_of_slots[city]
    test_data = np.zeros(shape=(m, 12, 495, 436, 8), dtype=np.uint8)
    ground_truth_data = np.zeros(shape=(m, 6, 495, 436, 8), dtype=np.uint8)
    test_additional_data = np.zeros(shape=(m, 2), dtype=np.uint8)
    i = 0
    for k, (date, slots) in enumerate(test_slots.items()):
        gc.collect(generation=2)
        logging.debug("%s %s [%s/%s], RAM %s%%", city, date, k, len(test_slots), psutil.virtual_memory()[2])
        weekday = weekday_parser(date)
        h5 = _extract_h5_for_city_and_date(city=city, city_date_finder=city_date_finder, date=date, output_directory=tempdir)
        data = load_h5_file(h5)

        for slot in slots:
            test_input, test_ground_truth_output = prepare_test(data=data, offset=slot)
            test_data[i] = test_input
            ground_truth_data[i] = test_ground_truth_output
            test_additional_data[i, 0] = weekday
            test_additional_data[i, 1] = slot
            expanded_config.append({"index": i, "date": date, "slot": slot})
            i += 1
    assert i == m, f"{i} != {m}"
    competition_dir_city = f"{output_directory}/{city}/"
    ground_truth_dir_city = f"{output_directory_ground_truth}/{competition}/{city}/"
    _check_exists_dir(competition_dir_city)
    _check_exists_dir(ground_truth_dir_city)
    logging.debug(
        "Writing participant test h5 files for %s %s %s (RAM %s %%, disk %s)",
        city,
        date,
        competition,
        psutil.virtual_memory()[2],
        disk_usage_human_readable(competition_dir_city),
    )

    write_data_to_h5(
        data=test_data, filename=f"{competition_dir_city}/{city}_test_{competition}.h5",
    )
    logging.debug(
        "Writing participant additional data h5 files for %s %s %s (RAM %s %%, disk %s)",
        city,
        date,
        competition,
        psutil.virtual_memory()[2],
        disk_usage_human_readable(competition_dir_city),
    )
    write_data_to_h5(data=test_additional_data, filename=f"{competition_dir_city}/{city}_test_additional_{competition}.h5")
    logging.debug(
        "Writing ground truth h5 files for %s %s %s(RAM %s %%, disk %s)",
        city,
        date,
        competition,
        psutil.virtual_memory()[2],
        disk_usage_human_readable(competition_dir_city),
    )
    write_data_to_h5(data=ground_truth_data, filename=f"{ground_truth_dir_city}/{city}_test_{competition}.h5")
    logging.debug(
        "Done writing h5 files for %s %s (RAM %s %%, disk %s)", city, date, psutil.virtual_memory()[2], disk_usage_human_readable(competition_dir_city)
    )
    return expanded_config


def _extract_and_slice_h5_files_for_testing(
    city_date_finder, output_directory, output_directory_ground_truth, test_slots_dir, expected_number_of_slots_per_city
):
    # no pool because of memory consumption!
    expanded_test_slots = {}
    with tempfile.TemporaryDirectory() as tempdir:
        for competition in ["temporal", "spatiotemporal"]:
            test_slots_file = f"{test_slots_dir}/test_slots_{competition}.json"
            logging.info(test_slots_file)
            with open(test_slots_file) as json_file:
                test_slots_data = json.load(json_file)

            expanded_test_slots[competition] = {}
            for arg in test_slots_data.items():
                city, _ = arg
                expanded_test_slots[competition][city] = _prepare_test_slots_city(
                    arg=arg,
                    city_date_finder=city_date_finder,
                    competition=competition,
                    expected_number_of_slots=expected_number_of_slots_per_city,
                    output_directory=output_directory,
                    output_directory_ground_truth=output_directory_ground_truth,
                    tempdir=tempdir,
                )
    return expanded_test_slots


def _check_expanded_test_slots(expanded_test_slots, city_date_finder, output_directory, output_directory_ground_truth):
    for competition, cities in expanded_test_slots.items():
        for city, slots in cities.items():
            print(f"  checking slots for {city}")
            for test in slots:
                slot = test["slot"]
                h5, _ = city_date_finder[(city, test["date"])]
                expected_data = load_h5_file(h5)
                actual_data_participants = load_h5_file(os.path.join(output_directory, city, f"{city}_test_{competition}.h5"))
                actual_data_ground_truth = load_h5_file(os.path.join(output_directory_ground_truth, competition, city, f"{city}_test_{competition}.h5"))
                test_index = test["index"]
                expected_data_ub = slot + 12
                assert (expected_data[slot:expected_data_ub] == actual_data_participants[test_index]).all()
                # 5,10,15,30,45 and 60 into the future
                assert (expected_data[slot + 11 + (5 // 5)] == actual_data_ground_truth[test_index, 0]).all()
                assert (expected_data[slot + 11 + (10 // 5)] == actual_data_ground_truth[test_index, 1]).all()
                assert (expected_data[slot + 11 + (15 // 5)] == actual_data_ground_truth[test_index, 2]).all()
                assert (expected_data[slot + 11 + (30 // 5)] == actual_data_ground_truth[test_index, 3]).all()
                assert (expected_data[slot + 11 + (45 // 5)] == actual_data_ground_truth[test_index, 4]).all()
                assert (expected_data[slot + 11 + (60 // 5)] == actual_data_ground_truth[test_index, 5]).all()


def create_parser() -> argparse.ArgumentParser:
    """Creates the argument parser for this program."""
    parser = argparse.ArgumentParser(description=("This programs generates the competition manifest files."))
    # data arguments
    parser.add_argument(
        "-tsd", "--test_slot_directory", type=str, default=".", help="Folder to write/read test_slots_<temporal|spatiotemporal>.json.", required=False
    )
    parser.add_argument(
        "-ddp",
        "--dynamic_data_path",
        type=str,
        default="../../data/raw",
        help="Folder with dynamic probe movies in h5 format (usually ./data/raw in project root), see data/README.md",
        required=False,
    )
    parser.add_argument(
        "-od",
        "--output_directory",
        type=str,
        default="./test_set",
        help="Folder to put test set output to, per city with subfolders, see data/README.md",
        required=False,
    )
    parser.add_argument(
        "-odgt",
        "--output_directory_ground_truth",
        type=str,
        default="./ground_truth",
        help="Folder to put generated ground truth to, per city with subfolders, see data/README.md",
        required=False,
    )
    parser.add_argument("-cf", "--config_file", type=str, default="test_slots_config.json", help='Path of "test_slots_config.json"', required=False)
    parser.add_argument("-utm", "--update_test_manifest", action="store_true")
    return parser


def main(params):
    # Create test manifests if needed
    with open(params.config_file, "r") as fp:
        config = json.load(fp)
        if params.update_test_manifest:
            _create_test_manifest(config, params.test_slot_directory)
            print(f"Success creating test manifest from {params.config_file} into {params.test_slot_directory}.")
        if not _check_test_manifest(config, params.test_slot_directory):
            exit(1)
        # Compute the expected number of slots for checking the output later
        expected_number_of_slots_per_city = {}
        for _, competition_items in config.items():
            for city, city_items in competition_items.items():
                assert city not in expected_number_of_slots_per_city
                expected_number_of_slots_per_city[city] = sum(int(config["n"]) for config in city_items)

    print(f"Using test_slots_<temporal|spatiotemporal>.json from {params.test_slot_directory}")

    # Create _test.h5 into output directory
    city_date_finder = _get_contents_from_tarballs_and_directories(params.dynamic_data_path, check_tars=False)
    expanded_test_slots = _extract_and_slice_h5_files_for_testing(
        city_date_finder, params.output_directory, params.output_directory_ground_truth, params.test_slot_directory, expected_number_of_slots_per_city
    )
    # Pedantry is strength: check our bookkeeping reflects the original data
    print("Checking expanded test slots ...")
    _check_expanded_test_slots(expanded_test_slots, city_date_finder, params.output_directory, params.output_directory_ground_truth)
    # Save the expanded test slots
    with open(params.config_file.replace("_config", "_expanded"), "w") as f:
        f.write(json.dumps(expanded_test_slots))

    # Copy static files to the golden dir
    print("Copy the static files to the ground truth dir ...")
    for competition, competition_items in config.items():
        for city, _ in competition_items.items():
            shutil.copyfile(
                f"{params.dynamic_data_path}/{city}/{city}_static.h5", f"{params.output_directory_ground_truth}/{competition}/{city}/{city}_static.h5"
            )

    # 6. create tar ball / zips for ground truth
    print("Create zips for ground truth")
    for competition, _ in config.items():
        with zipfile.ZipFile(f"{params.output_directory_ground_truth}/../ground_truth_{competition}.zip", "w") as ground_truth_f:
            print(f"{params.output_directory_ground_truth}/{competition}")
            for f in glob.glob(f"{params.output_directory_ground_truth}/{competition}/**/*.h5", recursive=True):
                print(f)
                ground_truth_f.write(f, arcname=os.path.join(*f.split(os.sep)[-2:]))


if __name__ == "__main__":
    parser = create_parser()
    try:
        params = parser.parse_args()
    except Exception as e:
        print(f"There was an error during execution, please review: {e}")
        parser.print_help()
        exit(1)

    t4c_apply_basic_logging_config("DEBUG")

    main(params)

    print("... success!")
