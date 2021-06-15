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
import logging
import os
import tempfile
import timeit
import zipfile

import numpy as np
import pytest
from scorecomp import EXPECTED_SHAPE
from scorecomp import main

from util.h5_util import write_data_to_h5


def test_scorecomp_write_log_and_score_despite_exception():
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            main(["-g", os.path.join(temp_dir, "nix.zip"), "-i", os.path.join(temp_dir, "nix.zip")])
        except:  # noqa
            pass
        log_file = os.path.join(temp_dir, "nix.log")
        assert os.path.exists(log_file)
        with open(log_file, "r") as f:
            content = f.read()
            logging.info(content)
            assert "contact us for details" in content
        score_file = os.path.join(temp_dir, "nix.score")
        assert os.path.exists(score_file)
        with open(score_file, "r") as f:
            content = f.read()
            logging.info(content)
            assert content == "999"


def test_scorecomp_scoring(caplog):
    # N.B. prevent pytest from swallowing all logging https://docs.pytest.org/en/6.2.x/logging.html#caplog-fixture
    caplog.set_level(logging.INFO, logger="participants-prediction")
    caplog.set_level(logging.INFO, logger="full-prediction")

    ground_truth = np.full(shape=EXPECTED_SHAPE, fill_value=1, dtype=np.uint8)
    prediction = np.full(shape=EXPECTED_SHAPE, fill_value=3, dtype=np.uint8)

    with tempfile.TemporaryDirectory() as temp_dir:
        ground_truth_h5 = os.path.join(temp_dir, "ground_truth.h5")
        write_data_to_h5(ground_truth, ground_truth_h5, compression="lzf", compression_level=None)
        prediction_h5 = os.path.join(temp_dir, "prediction.h5")
        write_data_to_h5(prediction, prediction_h5, compression="lzf", compression_level=None)

        ground_truth_zip = os.path.join(temp_dir, "ground_truth.zip")
        with zipfile.ZipFile(ground_truth_zip, "w") as ground_truth_f:
            ground_truth_f.write(ground_truth_h5, "test.h5")
        prediction_zip = os.path.join(temp_dir, "prediction.zip")
        with zipfile.ZipFile(prediction_zip, "w") as prediction_f:
            prediction_f.write(prediction_h5, "test.h5")
        main(["-g", ground_truth_zip, "-i", prediction_zip])

        log_file = os.path.join(temp_dir, "prediction.log")
        assert os.path.exists(log_file)
        full_log_file = os.path.join(temp_dir, "prediction-full.log")
        assert os.path.exists(full_log_file)
        with open(full_log_file, "r") as f:
            content = f.read()
            print(full_log_file)
            print(content)
            logging.info(content)
            assert "completed ok" in content
        with open(log_file, "r") as f:
            content = f.read()
            logging.info(content)
            assert "completed ok" in content
        score_file = os.path.join(temp_dir, "prediction.score")
        assert os.path.exists(score_file)
        with open(score_file, "r") as f:
            content = f.read()
            logging.info(content)
            assert np.isclose(float(content), 4.0)


@pytest.mark.skipif(os.getenv("CI") is not None, reason="Not enough resources in ci.")
@pytest.mark.parametrize("jobs,submissions,scored", [(1, 10, 2), (2, 10, 2), (4, 10, 2), (8, 10, 2)])
def test_unscored_from_folder(caplog, jobs, submissions, scored):
    # N.B. prevent pytest from swallowing all logging https://docs.pytest.org/en/6.2.x/logging.html#caplog-fixture
    submissions = list(range(submissions))
    scored_submissions = np.random.choice(submissions, size=scored, replace=False)

    for i in submissions:
        caplog.set_level(logging.INFO, logger=f"participants-submission-{i}")
        caplog.set_level(logging.INFO)

    ground_truth = np.full(shape=EXPECTED_SHAPE, fill_value=1, dtype=np.uint8)
    prediction = np.full(shape=EXPECTED_SHAPE, fill_value=3, dtype=np.uint8)

    with tempfile.TemporaryDirectory() as temp_dir:
        ground_truth_h5 = os.path.join(temp_dir, "ground_truth.h5")
        write_data_to_h5(ground_truth, ground_truth_h5, compression="lzf", compression_level=None)
        prediction_h5 = os.path.join(temp_dir, "prediction.h5")
        write_data_to_h5(prediction, prediction_h5, compression="lzf", compression_level=None)

        ground_truth_zip = os.path.join(temp_dir, "ground_truth.zip")
        with zipfile.ZipFile(ground_truth_zip, "w") as ground_truth_f:
            ground_truth_f.write(ground_truth_h5, "test.h5")

        submissions_dir = os.path.join(temp_dir, "submissions")
        os.makedirs(submissions_dir)
        for i in submissions:
            prediction_zip = os.path.join(submissions_dir, f"submission-{i}.zip")
            with zipfile.ZipFile(prediction_zip, "w") as prediction_f:
                prediction_f.write(prediction_h5, "test.h5")

        for i in scored_submissions:
            with open(os.path.join(submissions_dir, f"submission-{i}.score"), "w") as f:
                f.write("123.5")
            with open(os.path.join(submissions_dir, f"submission-{i}.log"), "w") as f:
                f.write("dummy")
            with open(os.path.join(submissions_dir, f"submission-{i}-full.log"), "w") as f:
                f.write("dummy full")

        scoring_time = timeit.timeit(lambda: main(["-g", ground_truth_zip, "-s", submissions_dir, "-j", str(jobs)]), number=1)
        print(f"scoring took {scoring_time / 1000:.2f}s")

        unscored_submissions = [i for i in submissions if i not in scored_submissions]
        for i in unscored_submissions:
            score_file = os.path.join(submissions_dir, f"submission-{i}.score")
            print(score_file)
            assert os.path.exists(score_file)
            with open(score_file, "r") as f:
                content = f.read()
                logging.info(content)
                assert np.isclose(float(content), 4.0)
            log_file = os.path.join(submissions_dir, f"submission-{i}.log")
            assert os.path.exists(log_file)
            with open(log_file, "r") as f:
                content = f.read()
                logging.info(content)
                assert "completed ok" in content
            full_log_file = os.path.join(submissions_dir, f"submission-{i}-full.log")
            assert os.path.exists(full_log_file)
            with open(full_log_file, "r") as f:
                content = f.read()
                logging.info(content)
                assert "completed ok" in content
