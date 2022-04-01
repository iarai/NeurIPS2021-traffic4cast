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
import datetime
import glob
import importlib
import logging
import os
import shutil
import tempfile
import traceback
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import psutil
import torch
from torch.nn import DataParallel

from t4c21.util.h5_util import load_h5_file
from t4c21.util.h5_util import write_data_to_h5
from t4c21.util.logging import t4c_apply_basic_logging_config


def load_torch_model_from_checkpoint(checkpoint: Union[str, Path], model: torch.nn.Module, map_location: str = None) -> torch.nn.Module:
    if not torch.cuda.is_available():
        map_location = "cpu"
    state_dict = torch.load(checkpoint, map_location=map_location)
    if isinstance(state_dict, DataParallel):
        logging.debug("     [torch-DataParallel]:")
        state_dict = state_dict.state_dict()
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k[:7] == "module.":
                k = k[7:]  # remove `module.` if trained with data parallelism
            new_state_dict[k] = v
        state_dict = new_state_dict
    elif isinstance(state_dict, dict) and "model" in state_dict:
        # Is that what ignite does?
        logging.debug("     [torch-model-attr]:")
        state_dict = state_dict["model"]
    elif isinstance(state_dict, dict) and "state_dict" in state_dict:
        # That's what lightning does.
        logging.debug("     [torch-state_dict-attr]:")
        state_dict = state_dict["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k[:6] == "model.":
                k = k[6:]  # remove `module.` if trained with data parallelism
            new_state_dict[k] = v
        state_dict = new_state_dict
    model.load_state_dict(state_dict)
    return model


def package_submission(
    data_raw_path: str,
    competition: str,
    model_str: str,
    model: Union[torch.nn.Module, Dict[str, torch.nn.Module]],
    device: str,
    submission_output_dir: Path,
    batch_size=10,
    num_tests_per_file=100,
    h5_compression_params: dict = None,
) -> Path:
    tstamp = datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d%H%M")

    if h5_compression_params is None:
        h5_compression_params = {}

    if submission_output_dir is None:
        submission_output_dir = Path(".")
    submission_output_dir.mkdir(exist_ok=True, parents=True)
    submission = submission_output_dir / f"submission_{model_str}_{competition}_{tstamp}.zip"
    logging.info(submission)

    competition_files = glob.glob(f"{data_raw_path}/**/*test_{competition}.h5", recursive=True)

    assert len(competition_files) > 0

    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(submission, "w") as z:
            for competition_file in competition_files:
                logging.info(f"  running model on {competition_file} (RAM {psutil.virtual_memory()[2]}%)")

                prediction = forward_torch_model_from_h5(
                    model=model,
                    num_tests_per_file=num_tests_per_file,
                    batch_size=batch_size,
                    competition_file=competition_file,
                    device=device,
                )
                unique_values = np.unique(prediction)
                logging.info(f"  {len(unique_values)} unique values in prediction in the range [{np.min(prediction)}, {np.max(prediction)}]")
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    logging.debug(str(np.unique(prediction)))
                temp_h5 = os.path.join(temp_dir, os.path.basename(competition_file))
                arcname = os.path.join(*competition_file.split(os.sep)[-2:])
                logging.info(f"  writing h5 file {temp_h5} (RAM {psutil.virtual_memory()[2]}%)")
                write_data_to_h5(prediction, temp_h5, **h5_compression_params)
                logging.info(f"  adding {temp_h5} as {arcname} (RAM {psutil.virtual_memory()[2]}%)")
                z.write(temp_h5, arcname=arcname)
            logging.info(z.namelist())
    submission_mb_size = os.path.getsize(submission) / (1024 * 1024)
    logging.info(f"Submission {submission} with {submission_mb_size:.2f}MB")
    logging.info(f"RAM {psutil.virtual_memory()[2]}%, disk usage {(shutil.disk_usage('.')[0] - shutil.disk_usage('.')[1]) / (1024 * 1024 * 1024):.2f}GB left")
    return submission


# TODO forward_torch_model_in_memory
def forward_torch_model_from_h5(
    model: torch.nn.Module,
    num_tests_per_file: int,
    batch_size: int,
    competition_file: str,
    device: str,
):
    model = model.to(device)
    model.eval()
    assert num_tests_per_file % batch_size == 0, f"num_tests_per_file={num_tests_per_file} must be a multiple of batch_size={batch_size}"
    num_batches = num_tests_per_file // batch_size
    prediction = np.zeros(shape=(num_tests_per_file, 6, 495, 436, 8), dtype=np.uint8)
    with torch.no_grad():
        for i in range(num_batches):
            batch_start = i * batch_size
            batch_end: np.ndarray = batch_start + batch_size
            test_data: np.ndarray = load_h5_file(competition_file, sl=slice(batch_start, batch_end), to_torch=False)

            test_data = torch.from_numpy(test_data)
            test_data = test_data.to(dtype=torch.float)
            test_data = test_data.to(device)
            # TODO inject static data appropriate to the model... this is UNet specific (new implementation)
            batch_prediction = model.forward((test_data, torch.zeros(size=([test_data.shape[0]] + list(test_data.shape[2:-1]) + [0]))))

            batch_prediction = batch_prediction.cpu().detach().numpy()
            batch_prediction = np.clip(batch_prediction, 0, 255)
            # clipping is important as assigning float array to uint8 array has not the intended effect.... (see `test_submission.test_assign_reload_floats)
            prediction[batch_start:batch_end] = batch_prediction
    return prediction


def find_module(using: str) -> Any:
    module = ".".join(using.split(".")[:-1])
    module = importlib.import_module(module)
    cls = using.split(".")[-1]
    cls = module.__getattribute__(cls)
    return cls


# TODO pass dict with model params
# TODO inject additional_data or static_data etc.
def create_parser() -> argparse.ArgumentParser:
    """Creates the argument parser for this program."""
    parser = argparse.ArgumentParser(description=("This programs creates a submission."))
    parser.add_argument("--checkpoint", type=str, help="Torch checkpoint file", required=True, default=None)
    parser.add_argument("--model_str", type=str, help="Fully qualified class name.", required=False, default="t4c21.vanilla_unet.UNet")
    parser.add_argument("--name", type=str, help="submission name", required=False, default="vanilla_unet")
    parser.add_argument("--data_raw_path", type=str, help="Path of raw data", required=False, default="./data/raw")
    parser.add_argument("--batch_size", type=int, help="Batch size for evaluation", required=False, default=10)
    parser.add_argument("--device", type=str, help="Device", required=False, default="cpu")
    parser.add_argument(
        "--submission_output_dir",
        type=str,
        default=None,
        required=False,
        help="If given, submission is stored to this directory instead of current.",
    )
    parser.add_argument("-c", "--competitions", nargs="+", help="<Required> Set flag", required=True, default=["temporal", "spatiotemporal"])
    return parser


def main(
    model_str: str,
    checkpoint: str,
    batch_size: int,
    device: str,
    data_raw_path: str,
    name: str,
    competitions: List[str],
    submission_output_dir: Optional[str] = None,
):
    t4c_apply_basic_logging_config()

    model = find_module(model_str)()

    load_torch_model_from_checkpoint(checkpoint=checkpoint, model=model)

    for competition in competitions:
        package_submission(
            data_raw_path=data_raw_path,
            competition=competition,
            model=model,
            model_str=name,
            batch_size=batch_size,
            device=device,
            h5_compression_params={"compression_level": 6},
            submission_output_dir=Path(submission_output_dir if submission_output_dir is not None else "."),
        )


if __name__ == "__main__":
    parser = create_parser()
    try:
        params = parser.parse_args()
        main(**vars(params))
    except Exception as e:
        print(f"There was an error during execution, please review: {e}")
        traceback.print_exc()
        parser.print_help()
        exit(1)
