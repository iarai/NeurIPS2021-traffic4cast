#!/usr/bin/python3
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
import glob
import json
import logging
import os
import re
import sys
import tempfile
import time
import zipfile
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Union

import h5py
import numpy as np
import psutil

EXPECTED_SHAPE = (100, 6, 495, 436, 8)
MAXSIZE = 800 * 1024 * 1024 * 8
VOL_CHANNELS = [0, 2, 4, 6]
SPEED_CHANNELS = [1, 3, 5, 7]
# naming convention: name what we keep, what we do not "integrate out"
#   slots -> axis 0
#   bins  -> axis 1
#   channels -> axis 4
#   volumes ->  filter on 0,2,4,6 of axis 4, but do not keep
#   speeds  ->  filter on 1,3,5,7 of axis 4, but do not keep
OVERALLONLY_CONFIG = [
    (
        None,
        [
            # overall mse
            (None, ""),
        ],
    ),
    (
        VOL_CHANNELS,
        [
            # volumes
            (None, "_volumes"),
        ],
    ),
    (
        SPEED_CHANNELS,
        [
            # speeds
            (None, "_speeds"),
        ],
    ),
]

CONFIG = [
    (
        None,
        [
            # overall mse
            (None, ""),
            # per-slot mse
            ((1, 2, 3, 4), "_slots"),
            # bins
            ((0, 2, 3, 4), "_bins"),
            ((2, 3, 4), "_slots_bins"),
            # all channels
            ((0, 1, 2, 3), "_channels"),
            ((1, 2, 3), "_slots_channels"),
        ],
    ),
    (
        VOL_CHANNELS,
        [
            # volumes
            (None, "_volumes"),
            ((1, 2, 3, 4), "_slots_volumes"),
        ],
    ),
    (
        SPEED_CHANNELS,
        [
            # speeds
            (None, "_speeds"),
            ((1, 2, 3, 4), "_slots_speeds"),
        ],
    ),
]


def sanitize(a):
    if hasattr(a, "shape"):
        return a.tolist()
    return a


def create_parser() -> argparse.ArgumentParser:
    """Create test files and copy static and dynamic h5 files to the same place
    and tar them."""
    parser = argparse.ArgumentParser(
        description=(
            "This script takes either the path for an individual T4c 2021 submission zip file and evaluates the total "
            "score or it scans through the submission directory to compute scores for all files missing a score."
        )
    )
    # data arguments
    parser.add_argument(
        "-g", "--ground_truth_archive", type=str, help="zip file containing the ground truth", required=True,
    )
    parser.add_argument(
        "-i", "--input_archive", type=str, help="single participant submission zip archive", required=False,
    )
    parser.add_argument(
        "-s", "--submissions_folder", type=str, help="folder containing participant submissions", required=False,
    )
    parser.add_argument("-j", "--jobs", type=int, help="Number of jobs to run in parallel", required=False, default=1)
    parser.add_argument("-n", "--no_json", help="To speed-up scoring, do not generate .score.json", required=False, default=False, action="store_true")

    return parser


def load_h5_file(file_path) -> np.ndarray:
    """Given a file path to an h5 file assumed to house a tensor, load that
    tensor into memory and return a pointer.

    Duplicated here so we can copy this file standalone.
    """
    # load
    with h5py.File(file_path, "r") as fr:
        data = fr.get("array")
        data = np.array(data)
        return data


def main(args):  # noqa C901
    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO"), format="[%(asctime)s][%(levelname)s][%(process)d][%(filename)s:%(funcName)s:%(lineno)d] %(message)s"
    )
    parser = create_parser()
    try:
        params = parser.parse_args(args)
        params = vars(params)
        ground_truth_archive = params["ground_truth_archive"]
        jobs = params["jobs"]
        no_json = params["no_json"]
        if no_json:
            # keep only first for overall, volumes and speeds
            global CONFIG
            CONFIG = OVERALLONLY_CONFIG
        if params["input_archive"] is not None:
            try:
                score_participant(input_archive=params["input_archive"], ground_truth_archive=ground_truth_archive)
            except Exception:
                # exceptions are logged to participants and full log. Should we remove score file in case of runtime exception (OOM)?
                pass
        elif params["submissions_folder"] is not None:
            try:
                score_unscored_participants(ground_truth_archive=ground_truth_archive, jobs=jobs, submissions_folder=params["submissions_folder"])
            except Exception:
                # exceptions are logged to participants and full log. Should we remove score file in case of runtime exception (OOM)?
                pass
        else:
            raise Exception("Either input archive or submissions folder must be given")
    except Exception as e:
        logging.exception(f"Could not parse args.", exc_info=e)
        parser.print_help()
        exit(1)


def score_unscored_participants(ground_truth_archive, jobs, submissions_folder):
    all_submissions = [z.replace(".zip", "") for z in glob.glob(f"{submissions_folder}/*.zip")]
    unscored = [s for s in all_submissions if not os.path.exists(os.path.join(submissions_folder, f"{s}.score"))]
    unscored_zips = [os.path.join(submissions_folder, f"{s}.zip") for s in unscored]
    if jobs == 0:
        for u in unscored_zips:
            score_participant(u, ground_truth_archive=ground_truth_archive)
    else:
        with Pool(processes=jobs) as pool:
            _ = list(pool.imap_unordered(partial(score_participant, ground_truth_archive=ground_truth_archive), unscored_zips))


def score_participant(input_archive: str, ground_truth_archive: str):
    submission_id = os.path.basename(input_archive).replace(".zip", "")

    full_handler = logging.FileHandler(input_archive.replace(".zip", "-full.log"))
    json_score_file = input_archive.replace(".zip", ".score.json")
    full_handler.setLevel(logging.INFO)
    full_handler.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s]%(message)s"))
    full_logger = logging.getLogger()
    full_logger.addHandler(full_handler)

    # create the log and score files with bad score in case an exception happens
    participants_handler = logging.FileHandler(input_archive.replace(".zip", ".log"))
    participants_handler.setLevel(logging.INFO)
    participants_handler.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s]%(message)s"))
    participants_logger_name = f"participants-{submission_id}"
    participants_logger = logging.getLogger(participants_logger_name)
    participants_logger.addHandler(participants_handler)
    input_archive_basename = os.path.basename(input_archive)
    participants_logger.info(f"start scoring of {input_archive_basename}")
    participants_logger.addHandler(full_handler)

    score_file_extensions = {
        ".score": "mse",
        ".score2": "mse_wiedemann",
    }
    for score_file_ext in score_file_extensions:
        score_file = input_archive.replace(".zip", score_file_ext)
        with open(score_file, "w") as f:
            f.write("999")
    try:
        # do scoring and update score file
        vanilla_score, scores_dict = do_score(
            input_archive=input_archive, ground_truth_archive=ground_truth_archive, participants_logger_name=participants_logger_name
        )
        with open(json_score_file, "w") as f:
            json.dump(scores_dict, f)
        for score_file_ext, score_key in score_file_extensions.items():
            score_file = input_archive.replace(".zip", score_file_ext)
            score = scores_dict["all"][score_key]
            with open(score_file, "w") as f:
                f.write(str(score))
        participants_logger.info(f"Evaluation completed ok with score {vanilla_score} for {input_archive_basename}")
        participants_handler.flush()

    except Exception as e:
        logging.exception(f"There was an error during execution, please review", exc_info=e)
        participants_logger.error(f"Evaluation errors for {input_archive_basename}, contact us for details.")


def do_score(ground_truth_archive: str, input_archive: str, participants_logger_name) -> Tuple[float,Dict]:  # noqa
    start_time = time.time()
    participants_logger = logging.getLogger(participants_logger_name)

    archive_size = os.path.getsize(input_archive)
    participants_logger.info(f"{os.path.basename(input_archive)} has size {archive_size / (1024 * 1024)}MB")
    if archive_size > MAXSIZE:
        msg = (
            f"Your submission archive is too large (> {MAXSIZE / (1024 * 1024):.2f}MB). "
            f"Have you activated HDF5 compression? Please adapt your files as necessary and resubmit."
        )
        participants_logger.error(msg)
        raise Exception(msg)
    with zipfile.ZipFile(input_archive) as prediction_f:
        prediction_file_list = [f for f in prediction_f.namelist() if "test" in f and f.endswith(".h5")]
    with zipfile.ZipFile(ground_truth_archive) as ground_truth_f:
        ground_truth_archive_list = [f for f in ground_truth_f.namelist() if "test" in f and "mask" not in f and f.endswith(".h5")]
        static_file_set = [f for f in ground_truth_f.namelist() if "static" in f and f.endswith(".h5")]
        mask_file_set = [f for f in ground_truth_f.namelist() if "mask" in f and f.endswith(".h5")]
    if set(prediction_file_list) != set(ground_truth_archive_list):
        msg = (
            f"Your submission differs from the ground truth file list. Please adapt the submitted archive as necessary and resubmit. "
            f"Missing files: {set(ground_truth_archive_list).difference(prediction_file_list)}. "
            f"Unexpected files: {set(prediction_file_list).difference(ground_truth_archive_list)}."
        )
        participants_logger.error(msg)
        raise Exception(msg)

    score = 0.0
    scores_dict = {"all": {}}
    count = 0

    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(ground_truth_archive) as ground_truth_f:
            with zipfile.ZipFile(input_archive) as prediction_f:

                for f in ground_truth_archive_list:
                    ground_truth_h5 = ground_truth_f.extract(f, path=temp_dir)
                    ground_truth = load_h5_file(ground_truth_h5)
                    prediction_h5 = prediction_f.extract(f, path=temp_dir)
                    prediction = load_h5_file(prediction_h5)
                    if prediction.shape != EXPECTED_SHAPE:
                        msg = (
                            f"At least one of your submission files ({f}) differs from the reference shape {EXPECTED_SHAPE}. Found {prediction.shape}."
                            f"Use the h5shape.py script to validate your files.; Please adapt your files as necessary and resubmit."
                        )
                        participants_logger.error(msg)
                        raise Exception(msg)
                    assert ground_truth.shape == EXPECTED_SHAPE, f"Ground truth corrupt for {f}. Found {ground_truth.shape}; expected {EXPECTED_SHAPE}."

                    # Check and convert the dtype to uint8. Ground truth is also uint8.
                    if prediction.dtype != np.dtype("uint8"):
                        logging.warning(f"Found input data with {prediction.dtype}, expected dtype('uint8'). Converting data to match expected dtype.")
                        prediction = prediction.astype("uint8")

                    # Try to load the mask
                    static_mask_filename = re.sub("_test_.*\\.h5", "_static.h5", f)
                    full_mask_filename = re.sub("\\.h5", "_mask.h5", f)
                    if static_mask_filename in static_file_set:
                        logging.info(f"Using static mask {static_mask_filename}")
                        static_h5 = ground_truth_f.extract(static_mask_filename, path=temp_dir)
                        static_mask = create_static_mask(load_h5_file(static_h5))
                        full_mask = np.broadcast_to(static_mask, EXPECTED_SHAPE)
                    elif full_mask_filename in mask_file_set:
                        logging.info(f"Using full mask {full_mask_filename}")
                        full_mask_h5 = ground_truth_f.extract(full_mask_filename, path=temp_dir)
                        full_mask = load_h5_file(full_mask_h5)
                    else:
                        raise Exception(f"Neither full mask nor static file to create mask for {ground_truth_archive}")

                    full_mask = full_mask > 0
                    city_name = f.split("/")[0]
                    compute_mse(prediction, ground_truth, city_name, full_mask, scores_dict)

                    logging.info(f"City scores {city_name}")
                    count += 1

    # N.B. we give the average of the per-city-normalized masked mse!
    # TODO add np.mean over all cities for numpy fields and sanitze only after doing this?
    for k in scores_dict["all"].keys():
        scores_dict["all"][k] /= count
    for _, d in scores_dict.items():
        for ki, v in d.items():
            d[ki] = sanitize(v)
    score = scores_dict["all"]["mse"]
    elapsed_seconds = time.time() - start_time
    logging.info(f"scoring {os.path.basename(input_archive)} took {elapsed_seconds :.1f}s")
    logging.info(f"Scores {scores_dict}")
    return score, scores_dict


# Same logic as in metrics/masking.py, duplicated here for easier portability.
def create_static_mask(static_roads, num_slots=0):
    static_mask_once = np.where(static_roads[0] > 0, 1, 0)
    assert static_mask_once.shape == EXPECTED_SHAPE[-3:-1], f"{static_mask_once.shape} != {EXPECTED_SHAPE[-3:-1]}"
    static_mask = np.repeat(static_mask_once, EXPECTED_SHAPE[-1]).reshape(EXPECTED_SHAPE[-3:])
    assert static_mask.shape == EXPECTED_SHAPE[-3:], f"{static_mask.shape}"
    assert (static_mask[:, :, 0] == static_mask_once).all()
    return static_mask


# Simliar logic as in metrics/mse.py, duplicated here for easier portability and with dict implementation and numpy only.
def compute_mse(actual, expected, city_name="", full_mask=None, scores_dict=None, config=None):
    if scores_dict is None:
        scores_dict = {"all": {}}
    if config is None:
        config = CONFIG

    scores_dict[city_name] = {}
    expected = expected.astype(np.int64)
    actual = actual.astype(np.int64)
    logging.debug(f"/ Start mse {city_name}")
    wiedemann_mask = create_wiedmann_mask(expected)
    for channels, items in config:
        logging.debug(f"    / Start mse {channels} {psutil.virtual_memory()}")
        ground_truth_channels = expected[..., channels]
        prediction_channels = actual[..., channels]
        wiedemann_mask_channels = wiedemann_mask[..., channels]
        se_channels = (ground_truth_channels - prediction_channels) ** 2

        for axis, label in items:
            logging.debug(f"      / Start mse {axis} {channels} {label} {psutil.virtual_memory()}")
            mse_channels = np.mean(se_channels, axis=axis)
            assert mse_channels.dtype == np.float64
            scores_dict[city_name][f"mse{label}"] = mse_channels
            scores_dict["all"][f"mse{label}"] = scores_dict["all"].get(f"mse{label}", np.zeros_like(mse_channels)) + mse_channels
            # TODO wiedemann_ratio
            mse_wiedemann_channels = np.mean(se_channels, axis=axis, where=wiedemann_mask_channels)
            scores_dict[city_name][f"mse_wiedemann{label}"] = mse_wiedemann_channels
            scores_dict[city_name][f"wiedemann_ratio{label}"] = (
                np.count_nonzero(wiedemann_mask_channels, axis=axis) / np.prod([wiedemann_mask_channels.shape[d] for d in axis])
                if axis is not None
                else wiedemann_mask_channels.size
            )
            scores_dict["all"][f"mse_wiedemann{label}"] = (
                scores_dict["all"].get(f"mse_wiedemann{label}", np.zeros_like(mse_wiedemann_channels)) + mse_wiedemann_channels
            )
            logging.debug(f"      \\ End mse {axis} {channels} {label} {psutil.virtual_memory()}")
            if full_mask is not None:
                # convert to boolean mask
                full_mask_channels = full_mask[..., channels] > 0
                full_and_wiedemann_mask_channels = full_mask_channels * wiedemann_mask_channels
                for m, s in [(full_mask_channels, ""), (full_and_wiedemann_mask_channels, "_wiedemann")]:
                    logging.debug(f"      / Start mse{s} masked {axis} {channels} {label} {psutil.virtual_memory()}")
                    # this is normalized to the mask!
                    scores_dict[city_name][f"mse{s}_masked{label}"] = np.mean(se_channels, axis=axis, where=m)
                    scores_dict[city_name][f"mask_nonzero{s}_masked{label}"] = np.count_nonzero(m, axis=axis)
                    logging.debug(f"      \\ End mse{s} masked {axis} {channels} {label} {psutil.virtual_memory()}")
        logging.debug(f"    \\ End mse {channels} {psutil.virtual_memory()}")
    logging.debug(f"\\ End mse {city_name}")
    scores_dict[city_name] = dict(scores_dict[city_name])
    return scores_dict


def create_wiedmann_mask(ground_truth: np.ndarray):
    mask = np.zeros(shape=ground_truth.shape, dtype=bool)
    mask[..., SPEED_CHANNELS] = ground_truth[..., VOL_CHANNELS] != 0
    mask[..., VOL_CHANNELS] = True
    return mask


def verify_submission(input_archive: Union[str, Path], competition: str):
    ground_truth_archive_list = {
        "temporal": [
            "CHICAGO/CHICAGO_test_temporal.h5",
            "ISTANBUL/ISTANBUL_test_temporal.h5",
            "BERLIN/BERLIN_test_temporal.h5",
            "MELBOURNE/MELBOURNE_test_temporal.h5",
        ],
        "spatiotemporal": ["NEWYORK/NEWYORK_test_spatiotemporal.h5", "VIENNA/VIENNA_test_spatiotemporal.h5"],
    }[competition]
    archive_size = os.path.getsize(input_archive)
    logging.info(f"{os.path.basename(input_archive)} has size {archive_size / (1024 * 1024)}MB")
    if archive_size > MAXSIZE:
        msg = (
            f"Your submission archive is too large (> {MAXSIZE / (1024 * 1024):.2f}MB). "
            f"Have you activated HDF5 compression? Please adapt your files as necessary and resubmit."
        )
        raise Exception(msg)
    with zipfile.ZipFile(input_archive) as prediction_f:
        prediction_file_list = [f for f in prediction_f.namelist() if "test" in f and f.endswith(".h5")]

    if set(prediction_file_list) != set(ground_truth_archive_list):
        msg = (
            f"Your submission differs from the ground truth file list. Please adapt the submitted archive as necessary and resubmit. "
            f"Missing files: {set(ground_truth_archive_list).difference(prediction_file_list)}."
            f"Unexpected files: {set(prediction_file_list).difference(ground_truth_archive_list)}."
        )
        raise Exception(msg)

    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(input_archive) as prediction_f:
            for f in ground_truth_archive_list:
                prediction_h5 = prediction_f.extract(f, path=temp_dir)
                prediction = load_h5_file(prediction_h5)
                if prediction.shape != EXPECTED_SHAPE:
                    msg = (
                        f"At least one of your submission files ({f}) differs from the reference shape {EXPECTED_SHAPE}. Found {prediction.shape}."
                        f"Use the h5shape.py script to validate your files.; Please adapt your files as necessary and resubmit."
                    )
                    raise Exception(msg)


if __name__ == "__main__":
    main(sys.argv[1:])
