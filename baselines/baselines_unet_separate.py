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
import sys
from collections import OrderedDict
from pathlib import Path

import torch

from baselines.baselines_cli import create_parser
from baselines.baselines_cli import run_model
from baselines.baselines_configs import configs
from competition.scorecomp import scorecomp
from competition.submission.submission import package_submission
from data.dataset.dataset import T4CDataset
from util.logging import t4c_apply_basic_logging_config
from util.monitoring import system_status


def load_torch_model_from_checkpoint(checkpoint: str, model: torch.nn.Module) -> torch.nn.Module:
    map_location = None
    if not torch.cuda.is_available():
        map_location = "cpu"

    state_dict = torch.load(checkpoint, map_location=map_location).state_dict()

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == "module.":
            k = k[7:]  # remove `module.` if trained with data parallelism
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    return model


def train_city(
    city: str,
    data_raw_path: str,
    experiment_name="unet_temporal",
    retrain_experiment_name: str = None,
    retrain_params_suffix: str = None,
    retrain_model_file: str = None,
    file_filter=None,
    model_str: str = "unet",
    limit: int = None,
    device: str = None,
    **kwargs,
):
    logging.info("===================================================================================")
    logging.info("===================================================================================")
    logging.info(f"               {city}                                                             ")
    logging.info("===================================================================================")
    logging.info("===================================================================================")

    model_file = f"{experiment_name}_{city}"
    model_file = f"t4c21_{model_file}.pt"

    if retrain_model_file is None:
        retrain_model_file = f"{retrain_experiment_name}_{city}_{retrain_params_suffix}"
        retrain_model_file = f"t4c21_{retrain_model_file}.pt"

    dataset_config = configs[model_str].get("dataset_config", {})
    dataset = T4CDataset(root_dir=data_raw_path, file_filter=file_filter, limit=limit, **dataset_config)
    logging.info("Dataset has size %s", len(dataset))
    assert len(dataset) > 0

    # Model
    logging.info("Create train_model.")
    model_class = configs[model_str]["model_class"]
    model_config = configs[model_str].get("model_config", {})
    model = model_class(**model_config)
    dataloader_config = configs[model_str].get("dataloader_config", {})
    optimizer_config = configs[model_str].get("optimizer_config", {})

    if os.path.exists(retrain_model_file):
        logging.info(f"Loading pretrained model {retrain_model_file}")
        load_torch_model_from_checkpoint(checkpoint=retrain_model_file, model=model)

    logging.info("Going to run train_model.")
    logging.info(system_status())
    _, device = run_model(train_model=model, dataset=dataset, dataloader_config=dataloader_config, optimizer_config=optimizer_config, device=device, **kwargs)

    torch.save(model, model_file)
    logging.info(f"saved {model_file}")
    return model_file


def main(args):
    parser = create_parser()
    args = parser.parse_args(args)

    t4c_apply_basic_logging_config()

    model_str = args.model_str
    resume_checkpoint = args.resume_checkpoint
    assert resume_checkpoint is None, f"not applicable for separate models"
    file_filter = args.file_filter
    assert file_filter is None, f"not applicable for separate models"

    device = args.device
    data_raw_path = args.data_raw_path
    ground_truth_dir = args.ground_truth_dir
    batch_size = args.batch_size
    num_tests_per_file = args.num_tests_per_file
    submission_output_dir = args.submission_output_dir if args.submission_output_dir is not None else "."

    args = vars(args)
    args.pop("file_filter")
    args.pop("resume_checkpoint")

    temporal_cities = ["BERLIN", "MELBOURNE", "ISTANBUL", "CHICAGO"]
    spatiotemporal_cities = ["VIENNA", "NEWYORK"]
    model_files = {}

    # For the temporal challenge we use the basic Unet configuration and train it separately for all 4 cities using the training data from 2019.
    for city in temporal_cities:
        model_files[city] = train_city(city=city, file_filter=f"**/*{city}*8ch.h5", **args)
    # For the spatio-temporal challenge we use the pre-trained Unet for Berlin and train a couple more epochs with data sampled from all training cities.
    fine_tuned = train_city(city=city, file_filter=f"**/*8ch.h5", retrain_model_file=model_files["BERLIN"], **args)
    for city in spatiotemporal_cities:
        model_files[city] = fine_tuned
    models = {
        city: load_torch_model_from_checkpoint(f, configs[model_str]["model_class"](**configs[model_str].get("model_config", {})))
        for city, f in model_files.items()
    }

    logging.info(model_files)
    competitions = ["temporal", "spatiotemporal"]

    for competition in competitions:

        submission = package_submission(
            data_raw_path=data_raw_path,
            competition=competition,
            model=models,
            model_str=model_str,
            device=device,
            h5_compression_params={"compression_level": 6},
            submission_output_dir=Path(submission_output_dir),
            batch_size=batch_size,
            num_tests_per_file=num_tests_per_file,
        )

        if ground_truth_dir is not None:
            ground_truth_dir = Path(ground_truth_dir)
            scorecomp.score_participant(ground_truth_archive=str(ground_truth_dir / f"ground_truth_{competition}.zip"), input_archive=str(submission))
        else:
            scorecomp.verify_submission(input_archive=submission, competition=competition)


if __name__ == "__main__":
    main(sys.argv[1:])
