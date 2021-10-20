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
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Optional
from typing import Union

import h5py
import numpy as np
import torch
from torch.nn.functional import mse_loss as torch_mse

EXPECTED_SHAPE = (100, 6, 495, 436, 8)
MAXSIZE = 800 * 1024 * 1024 * 8
VOL_CHANNELS = [0, 2, 4, 6]
SPEED_CHANNELS = [1, 3, 5, 7]


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
    parser.add_argument("-b", "--batch_size", type=int, help="Batch size for scoring", required=False, default=10)

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


def main(args):
    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO"), format="[%(asctime)s][%(levelname)s][%(process)d][%(filename)s:%(funcName)s:%(lineno)d] %(message)s"
    )
    parser = create_parser()
    try:
        params = parser.parse_args(args)
        params = vars(params)
        ground_truth_archive = params["ground_truth_archive"]
        jobs = params["jobs"]
        if params["input_archive"] is not None:
            try:
                score_participant(input_archive=params["input_archive"], ground_truth_archive=ground_truth_archive, batch_size=params["batch_size"])
            except Exception:
                # exceptions are logged to participants and full log. Should we remove score file in case of runtime exception (OOM)?
                pass
        elif params["submissions_folder"] is not None:
            try:
                score_unscored_participants(
                    ground_truth_archive=ground_truth_archive, jobs=jobs, submissions_folder=params["submissions_folder"], batch_size=params["batch_size"]
                )
            except Exception:
                # exceptions are logged to participants and full log. Should we remove score file in case of runtime exception (OOM)?
                pass
        else:
            raise Exception("Either input archive or submissions folder must be given")
    except Exception as e:
        logging.exception(f"Could not parse args.", exc_info=e)
        parser.print_help()
        exit(1)


def score_unscored_participants(ground_truth_archive, jobs, submissions_folder, batch_size: int = 10):
    all_submissions = [z.replace(".zip", "") for z in glob.glob(f"{submissions_folder}/*.zip")]
    unscored = [s for s in all_submissions if not os.path.exists(os.path.join(submissions_folder, f"{s}.score"))]
    unscored_zips = [os.path.join(submissions_folder, f"{s}.zip") for s in unscored]
    if jobs == 1:
        for u in unscored_zips:
            score_participant(u, ground_truth_archive=ground_truth_archive, batch_size=batch_size)
    else:
        with Pool(processes=jobs) as pool:
            _ = list(pool.imap_unordered(partial(score_participant, ground_truth_archive=ground_truth_archive, batch_size=batch_size), unscored_zips))


def score_participant(input_archive: str, ground_truth_archive: str, batch_size: int = 10):
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

    score_file = input_archive.replace(".zip", ".score")
    with open(score_file, "w") as f:
        f.write("999")
    try:
        # do scoring and update score file
        score, scores_dict = do_score(
            input_archive=input_archive, ground_truth_archive=ground_truth_archive, participants_logger_name=participants_logger_name, batch_size=batch_size
        )
        with open(score_file, "w") as f:
            f.write(str(score))
        with open(json_score_file, "w") as f:
            json.dump(scores_dict, f)
        participants_logger.info(f"Evaluation completed ok with score {score} for {input_archive_basename}")
        participants_handler.flush()

    except Exception as e:
        logging.exception(f"There was an error during execution, please review", exc_info=e)
        participants_logger.error(f"Evaluation errors for {input_archive_basename}, contact us for details.")


# Same logic as in metrics/masking.py, duplicated here for easier portability.
def create_static_mask(static_roads, num_slots=0):
    static_mask_once = np.where(static_roads[0] > 0, 1, 0)
    assert static_mask_once.shape == EXPECTED_SHAPE[-3:-1], f"{static_mask_once.shape} != {EXPECTED_SHAPE[-3:-1]}"
    static_mask = np.repeat(static_mask_once, EXPECTED_SHAPE[-1]).reshape(EXPECTED_SHAPE[-3:])
    assert static_mask.shape == EXPECTED_SHAPE[-3:], f"{static_mask.shape}"
    assert (static_mask[:, :, 0] == static_mask_once).all()
    return static_mask


# Similar logic as in metrics/mse.py, duplicated here for easier portability.
def compute_mse(actual: np.ndarray, expected: np.ndarray, mask: Optional[np.ndarray] = None):
    scores = {}
    actual = torch.from_numpy(actual[:]).float()
    expected = torch.from_numpy(expected[:]).float()
    scores["mse"] = torch_mse(expected, actual).numpy().item()
    actual_volumes = actual[..., VOL_CHANNELS]
    actual_speeds = actual[..., SPEED_CHANNELS]
    expected_volumes = expected[..., VOL_CHANNELS]
    expected_speeds = expected[..., SPEED_CHANNELS]
    scores["mse_volumes"] = torch_mse(expected_volumes, actual_volumes).numpy().item()
    scores["mse_speeds"] = torch_mse(expected_speeds, actual_speeds).numpy().item()

    if mask is not None:
        mask = torch.from_numpy(mask[:]).float()

        # ensure there is enough memory! Should we remove score file again in this case?
        actual = actual * mask
        expected = expected * mask
        actual_volumes = actual[..., VOL_CHANNELS]
        actual_speeds = actual[..., SPEED_CHANNELS]
        expected_volumes = expected[..., VOL_CHANNELS]
        expected_speeds = expected[..., SPEED_CHANNELS]

        mask_ratio = np.count_nonzero(mask) / np.prod(mask.size())
        scores["mask_ratio"] = mask_ratio
        mv = mask[..., VOL_CHANNELS]
        mask_ratio_volumes = np.count_nonzero(mv) / np.prod(mv.size())
        scores["mask_ratio_volumes"] = mask_ratio_volumes
        ms = mask[..., SPEED_CHANNELS]
        mask_ratio_speeds = np.count_nonzero(ms) / np.prod(ms.size())
        scores["mask_ratio_speeds"] = mask_ratio_speeds
        mse_masked_base = torch_mse(expected * mask, actual * mask).numpy().item()

        scores["mse_masked_base"] = mse_masked_base
        mse_masked_volumes_base = torch_mse(expected_volumes, actual_volumes).numpy().item()
        scores["mse_masked_volumes_base"] = mse_masked_volumes_base
        mse_masked_speeds_base = torch_mse(expected_speeds, actual_speeds).numpy().item()
        scores["mse_masked_speeds_base"] = mse_masked_speeds_base

        scores["mse_masked"] = mse_masked_base / mask_ratio if mask_ratio > 0 else 0
        scores["mse_masked_volumes"] = mse_masked_volumes_base / mask_ratio_volumes if mask_ratio_volumes > 0 else 0
        scores["mse_masked_speeds"] = mse_masked_speeds_base / mask_ratio_speeds if mask_ratio_speeds > 0 else 0

    return scores


def do_score(ground_truth_archive: str, input_archive: str, participants_logger_name, batch_size: int = 10) -> float:  # noqa
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
    scores_dict = {"all": defaultdict(float)}
    count = 0

    assert EXPECTED_SHAPE[0] % batch_size == 0, f"{EXPECTED_SHAPE[0]} % {batch_size} != 0"
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
                    assert ground_truth.shape == EXPECTED_SHAPE, f"Ground truth corrupt for {f}. Found {ground_truth_archive.shape}; expected {EXPECTED_SHAPE}."

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
                        mask = np.broadcast_to(static_mask, EXPECTED_SHAPE)
                    elif full_mask_filename in mask_file_set:
                        logging.info(f"Using full mask {full_mask_filename}")
                        full_mask_h5 = ground_truth_f.extract(full_mask_filename, path=temp_dir)
                        mask = load_h5_file(full_mask_h5)
                    else:
                        raise Exception(f"Neither full mask nor static file to create mask for {ground_truth_archive}")

                    city_name = f.split("/")[0]
                    scores_dict[city_name] = {}
                    # Compute the mse in batches to save memory
                    for i in range(len(ground_truth) // batch_size):
                        batch_start = i * batch_size
                        batch_end = (i + 1) * batch_size
                        # keep for now to regression test behaviour
                        if static_mask is not None:
                            batch_scores = compute_mse(ground_truth[batch_start:batch_end], prediction[batch_start:batch_end], static_mask)
                        else:
                            batch_scores = compute_mse(ground_truth[batch_start:batch_end], prediction[batch_start:batch_end], mask[batch_start:batch_end])
                        for k, v in batch_scores.items():
                            scores_dict["all"][k] += v
                            scores_dict[city_name].setdefault(k, []).append(v)
                        count += 1
                    # Save the per prediction (city) score
                    mse_masked_items = []
                    for k in scores_dict[city_name].keys():
                        if k.startswith("mse_masked") and not k.endswith("base"):
                            mse_masked_items.append(k)
                        scores_dict[city_name][k] = np.mean(scores_dict[city_name][k])

                    # we need to divide by average ratio to get masked mse (ratio is proportional to non-zero count)
                    for k in mse_masked_items:
                        ratio = scores_dict[city_name][k.replace("masked", "ratio").replace("mse", "mask")]
                        scores_dict[city_name][k] = scores_dict[city_name][f"{k}_base"] / ratio if ratio > 0 else 0

                    ground_truth = ground_truth.astype(np.float64)
                    prediction = prediction.astype(np.float64)
                    # naming convention: name what we keep, what we do not "integrate out"
                    #   slots -> axis 0
                    #   bins  -> axis 1
                    #   channels -> axis 4
                    #   volumes ->  filter on 0,2,4,6 of axis 4, but do not keep
                    #   speeds  ->  filter on 1,3,5,7 of axis 4, but do not keep
                    config = [
                        # overall mse
                        (None, None, ""),
                        # per-slot mse
                        ((1, 2, 3, 4), None, "_slots"),
                        # bins
                        ((0, 2, 3, 4), None, "_bins"),
                        ((2, 3, 4), None, "_slots_bins"),
                        # all channels
                        ((0, 1, 2, 3), None, "_channels"),
                        ((1, 2, 3), None, "_slots_channels"),
                        # volumes and speeds
                        (None, VOL_CHANNELS, "_volumes"),
                        (None, SPEED_CHANNELS, "_speeds"),
                        ((1, 2, 3, 4), VOL_CHANNELS, "_slots_volumes"),
                        ((1, 2, 3, 4), SPEED_CHANNELS, "_slots_speeds"),
                    ]
                    for axis, channels, label in config:
                        if channels is not None:
                            mse_numpy = np.mean((ground_truth[..., channels] - prediction[..., channels]) ** 2, axis=axis)
                        else:
                            mse_numpy = np.mean((ground_truth - prediction) ** 2, axis=axis)
                        scores_dict[city_name][f"mse{label}_numpy"] = mse_numpy.tolist()

                    if mask is not None:
                        prediction = prediction * mask
                        ground_truth = ground_truth * mask
                        assert prediction.dtype == np.float64
                        assert ground_truth.dtype == np.float64

                        for axis, channels, label in config:
                            if channels is not None:
                                mse_masked_base = np.mean((ground_truth[..., channels] - prediction[..., channels]) ** 2, axis=axis)
                                mask_ratio = np.count_nonzero(mask[..., channels], axis=axis)
                            else:
                                mse_masked_base = np.mean((ground_truth - prediction) ** 2, axis=axis)
                                mask_ratio = np.count_nonzero(mask, axis=axis)
                            size = np.prod([mask.shape[d] for d in axis]) if axis is not None else mask.size

                            def sanitize(a):
                                if hasattr(mask_ratio, "shape"):
                                    return a.tolist()
                                return a

                            scores_dict[city_name][f"mask_ratio{label}_numpy"] = sanitize(mask_ratio / size)
                            scores_dict[city_name][f"mse_masked{label}_base_numpy"] = sanitize(mse_masked_base)
                            # if mask_ratio is zero, then there are no ones there and mse is 0 in the masked out data.
                            mse_masked = np.divide(mse_masked_base, mask_ratio, where=mask_ratio > 0)
                            scores_dict[city_name][f"mse_masked{label}_numpy"] = sanitize(mse_masked)

                    # regression tests for *_numpy against batched torch implementation
                    # TODO remove batching?
                    # TODO remove _numpy extensions

                    for k in scores_dict[city_name]:
                        if "numpy" not in k:
                            if "mask" in k and mask is None:
                                continue
                            np.allclose(scores_dict[city_name][k], scores_dict[city_name][f"{k}_numpy"]), k
                            logging.info(f"Checked numpy for {k}")
                    assert np.array(scores_dict[city_name]["mse_slots_numpy"]).shape == (len(ground_truth),), str(
                        np.array(scores_dict[city_name]["mse_slots_numpy"]).shape
                    )
                    assert np.array(scores_dict[city_name]["mse_slots_bins_numpy"]).shape == (len(ground_truth), 6), str(
                        np.array(scores_dict[city_name]["mse_slots_bins_numpy"]).shape
                    )
                    assert np.array(scores_dict[city_name]["mse_bins_numpy"]).shape == (6,), str(np.array(scores_dict[city_name]["mse_bins_numpy"]).shape)
                    assert np.array(scores_dict[city_name]["mse_slots_channels_numpy"]).shape == (len(ground_truth), 8), str(
                        np.array(scores_dict[city_name]["mse_slots_channels_numpy"]).shape
                    )
                    assert np.array(scores_dict[city_name]["mse_channels_numpy"]).shape == (8,), str(
                        np.array(scores_dict[city_name]["mse_channels_numpy"]).shape
                    )
                    assert np.array(scores_dict[city_name]["mse_slots_volumes_numpy"]).shape == (len(ground_truth),), str(
                        np.array(scores_dict[city_name]["mse_slots_volumes_numpy"]).shape
                    )
                    assert np.array(scores_dict[city_name]["mse_slots_speeds_numpy"]).shape == (len(ground_truth),), str(
                        np.array(scores_dict[city_name]["mse_slots_speeds_numpy"]).shape
                    )

                    # TODO plausi: average of slots/bins/channel should again be the same as torch imple

                    scores_dict[city_name] = dict(scores_dict[city_name])

                    logging.info(f"City scores {city_name}")

    # N.B. we give the average of the per-city-normalized masked mse!
    # TODO add np.mean over all cities for numpy fields and sanitze only after doing this?
    for k in scores_dict["all"].keys():
        scores_dict["all"][k] /= count
    scores_dict["all"] = dict(scores_dict["all"])
    score = scores_dict["all"]["mse"]
    elapsed_seconds = time.time() - start_time
    logging.info(f"scoring {os.path.basename(input_archive)} took {elapsed_seconds :.1f}s")
    logging.info(f"Scores {scores_dict}")
    return score, scores_dict


def verify_submission(input_archive: Union[str, Path], competition: str, batch_size: int = 10):
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

    assert EXPECTED_SHAPE[0] % batch_size == 0
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
