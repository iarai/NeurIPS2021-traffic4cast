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
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pytest

from baselines.baselines_cli import main
from competition.competition_constants import MAX_TEST_SLOT_INDEX
from competition.scorecomp import scorecomp
from util.h5_util import write_data_to_h5


@pytest.mark.skipif(os.getenv("CI") is not None, reason="Not enough resources in ci.")
@pytest.mark.parametrize(
    "model_str", ["naive_average", "unet", "gcn"],
)
def test_baselines_cli_run_through(model_str):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        data_raw_path = temp_dir_path / "raw"
        dynamic_file = data_raw_path / "DOWNTOWN" / "training" / "1970-01-01_DOWNTOWN_8ch.h5"
        test_temporal = data_raw_path / "DOWNTOWN" / "DOWNTOWN_test_temporal.h5"
        static_file = data_raw_path / "DOWNTOWN" / "DOWNTOWN_static.h5"
        additional_test_temporal = data_raw_path / "DOWNTOWN" / "DOWNTOWN_test_additional_temporal.h5"
        test_spatiotemporal = data_raw_path / "DOWNTOWN" / "DOWNTOWN_test_spatiotemporal.h5"
        additional_test_spatiotemporal = data_raw_path / "DOWNTOWN" / "DOWNTOWN_test_additional_spatiotemporal.h5"
        submission_output_dir = temp_dir_path / "submissions"

        dynamic_file.parent.mkdir(exist_ok=True, parents=True)

        data = np.random.randint(256, size=(288, 495, 436, 8), dtype=np.uint8)
        write_data_to_h5(data=data, filename=dynamic_file, compression="lzf", compression_level=None)
        data = np.random.randint(2, size=(9, 495, 436), dtype=np.uint8) * 255
        data[:, 0, :] = 0
        data[:, 494, :] = 0
        data[:, :, 0] = 0
        data[:, :, 435] = 0
        write_data_to_h5(data=data, filename=static_file, compression="lzf", compression_level=None)

        num_tests_per_file = 4
        data = np.random.randint(256, size=(num_tests_per_file, 12, 495, 436, 8), dtype=np.uint8)
        write_data_to_h5(data=data, filename=str(test_temporal), compression="lzf", compression_level=None)
        data = np.random.randint(MAX_TEST_SLOT_INDEX, size=(num_tests_per_file, 2), dtype=np.uint8)
        write_data_to_h5(data=data, filename=additional_test_temporal, compression="lzf", compression_level=None)

        data = np.random.randint(256, size=(num_tests_per_file, 12, 495, 436, 8), dtype=np.uint8)
        write_data_to_h5(data=data, filename=test_spatiotemporal, compression="lzf", compression_level=None)
        data = np.random.randint(MAX_TEST_SLOT_INDEX, size=(num_tests_per_file, 495, 436, 8), dtype=np.uint8)
        write_data_to_h5(data=data, filename=additional_test_spatiotemporal, compression="lzf", compression_level=None)

        ground_truth_dir = temp_dir_path / "ground_truth"
        ground_truth_dir.mkdir()

        for competition in ["temporal", "spatiotemporal"]:
            data = np.random.randint(256, size=(num_tests_per_file, 6, 495, 436, 8), dtype=np.uint8)
            ground_truth_h5 = ground_truth_dir / f"DOWNTOWN_test_{competition}.h5"
            write_data_to_h5(data, ground_truth_h5, compression="lzf", compression_level=None)
            with zipfile.ZipFile(ground_truth_dir / f"ground_truth_{competition}.zip", "w") as ground_truth_f:
                ground_truth_f.write(ground_truth_h5, arcname=f"DOWNTOWN/DOWNTOWN_test_{competition}.h5")
        scorecomp.EXPECTED_SHAPE = (num_tests_per_file, 6, 495, 436, 8)
        scorecomp.BATCH_SIZE = 2
        main(
            [
                "--model_str",
                model_str,
                "--limit",
                "4",
                "--epochs",
                "1",
                "--data_raw_path",
                str(data_raw_path),
                "--num_workers",
                "1",
                "--ground_truth_dir",
                str(ground_truth_dir),
                "--submission_output_dir",
                str(submission_output_dir),
                "--batch_size",
                "2",
                "--num_tests_per_file",
                str(num_tests_per_file),
                "--device",
                "cpu",
            ]
        )
        logs = list(Path(submission_output_dir).rglob("submission*.log"))
        assert len(logs) == 4, f"found {len(logs)}, expected 4. {logs}"
        for log in logs:
            content = log.read_text()
            assert "ERROR" not in content
            assert "completed ok with score" in content
        submissions = list(submission_output_dir.rglob("submission*.zip"))
        assert len(submissions) == 2
