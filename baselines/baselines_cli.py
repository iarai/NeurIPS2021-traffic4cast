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
import binascii
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F  # noqa
import torch.optim as optim
import torch_geometric
import tqdm
from ignite.contrib.handlers import TensorboardLogger
from ignite.contrib.handlers.tensorboard_logger import GradsHistHandler
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import create_supervised_evaluator
from ignite.engine import create_supervised_trainer
from ignite.engine import Engine
from ignite.engine import Events
from ignite.handlers import Checkpoint
from ignite.handlers import DiskSaver
from ignite.handlers import global_step_from_engine
from ignite.metrics import Loss
from ignite.metrics import RunningAverage
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler

from baselines.baselines_configs import configs
from baselines.checkpointing import load_torch_model_from_checkpoint
from baselines.checkpointing import save_torch_model_to_checkpoint
from competition.scorecomp import scorecomp
from competition.submission.submission import package_submission
from data.dataset.dataset import T4CDataset
from data.dataset.dataset_geometric import GraphTransformer
from data.dataset.dataset_geometric import T4CGeometricDataset
from util.logging import t4c_apply_basic_logging_config
from util.monitoring import system_status
from util.tar_util import untar_files


def run_model(
    train_model: torch.nn.Module,
    dataset: T4CDataset,
    random_seed: int,
    train_fraction: float,
    val_fraction: float,
    batch_size: int,
    num_workers: int,
    epochs: int,
    dataloader_config: dict,
    optimizer_config: dict,
    device: str = None,
    geometric: bool = False,
    limit: Optional[int] = None,
    data_parallel=False,
    device_ids=None,
    **kwargs,
):  # noqa

    logging.info("dataset has size %s", len(dataset))

    # Train / Dev / Test set splits
    logging.info("train/dev split")
    full_dataset_size = len(dataset)

    effective_dataset_size = full_dataset_size
    if limit is not None:
        effective_dataset_size = min(full_dataset_size, limit)
    indices = list(range(full_dataset_size))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    assert np.isclose(train_fraction + val_fraction, 1.0)
    num_train_items = max(int(np.floor(train_fraction * effective_dataset_size)), batch_size)
    num_val_items = max(int(np.floor(val_fraction * effective_dataset_size)), batch_size)
    logging.info(
        "Taking %s from dataset of length %s, splitting into %s train items and %s val items",
        effective_dataset_size,
        full_dataset_size,
        num_train_items,
        num_val_items,
    )

    # Data loaders
    if geometric:
        train_set, val_set, _ = torch.utils.data.random_split(dataset, [num_train_items, num_val_items, full_dataset_size - num_train_items - num_val_items])

        train_loader = torch_geometric.data.DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, **dataloader_config)
        val_loader = torch_geometric.data.DataLoader(val_set, batch_size=batch_size, num_workers=num_workers, **dataloader_config)
    else:
        train_indices, dev_indices = indices[:num_train_items], indices[num_train_items : num_train_items + num_val_items]

        train_sampler = SubsetRandomSampler(train_indices)
        dev_sampler = SubsetRandomSampler(dev_indices)

        train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=train_sampler, **dataloader_config)
        val_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=dev_sampler, **dataloader_config)

    # Optimizer
    if "lr" not in optimizer_config:
        optimizer_config["lr"] = 1e-4

    if device is None:
        logging.warning("device not set, torturing CPU.")
        device = "cpu"
        # TODO data parallelism and whitelist

    if torch.cuda.is_available() and data_parallel:
        # https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
        if torch.cuda.device_count() > 1:
            # https://stackoverflow.com/questions/59249563/runtimeerror-module-must-have-its-parameters-and-buffers-on-device-cuda1-devi
            train_model = torch.nn.DataParallel(train_model, device_ids=device_ids)
            logging.info(f"Let's use {len(train_model.device_ids)} GPUs: {train_model.device_ids}!")
            device = f"cuda:{train_model.device_ids[0]}"

    optimizer = optim.Adam(train_model.parameters(), **optimizer_config)

    train_model = train_model.to(device)

    # Loss
    loss = F.mse_loss
    if geometric:
        train_pure_torch(device, epochs, optimizer, train_loader, val_loader, train_model)
    else:
        train_ignite(device, epochs, loss, optimizer, train_loader, val_loader, train_model)
    logging.info("End training of train_model %s on %s for %s epochs", train_model, device, epochs)
    return train_model, device


def train_pure_torch(device, epochs, optimizer, train_loader, val_loader, train_model):
    best_acc = 10000
    for epoch in range(epochs):
        _train_epoch_pure_torch(train_loader, device, train_model, optimizer)
        acc = _val_pure_torch(val_loader, device, train_model)
        if acc > best_acc:
            best_acc = acc
        log = "Epoch: {:03d}, Test: {:.4f}"
        logging.info(log.format(epoch, acc))
        save_torch_model_to_checkpoint(model=train_model, model_str="gcn", epoch=epoch)


def _train_epoch_pure_torch(loader, device, model, optimizer):
    loss_to_print = 0
    for i, input_data in enumerate(tqdm.tqdm(loader, desc="train")):
        if isinstance(input_data, torch_geometric.data.Data):
            input_data = input_data.to(device)
            ground_truth = input_data.y
        else:
            input_data, ground_truth = input_data
            input_data = input_data.to(device)
            ground_truth = ground_truth.to(ground_truth)

        model.train()
        optimizer.zero_grad()
        criterion = torch.nn.MSELoss()
        output = model(input_data)
        loss = criterion(output, ground_truth)
        loss.backward()
        optimizer.step()

        loss_to_print += float(loss)
        if i % 1000 == 0 and i > 0:
            logging.info("train_loss %s", loss_to_print / 1000)
            loss_to_print = 0


@torch.no_grad()
def _val_pure_torch(loader, device, model):
    running_loss = 0
    for input_data in tqdm.tqdm(loader, desc="val"):
        if isinstance(input_data, torch_geometric.data.Data):
            input_data = input_data.to(device)
            ground_truth = input_data.y
        else:
            input_data, ground_truth = input_data
        model.eval()
        criterion = torch.nn.MSELoss()
        output = model(input_data)
        loss = criterion(output, ground_truth)
        running_loss = running_loss + float(loss)
    return running_loss / len(loader) if len(loader) > 0 else running_loss


def train_ignite(device, epochs, loss, optimizer, train_loader, val_loader, train_model):
    # Validator
    validation_evaluator = create_supervised_evaluator(train_model, metrics={"val_loss": Loss(loss)}, device=device)
    # Trainer
    trainer = create_supervised_trainer(train_model, optimizer, loss, device=device)
    train_evaluator = create_supervised_evaluator(train_model, metrics={"loss": Loss(loss)}, device=device)
    run_id = binascii.hexlify(os.urandom(15)).decode("utf-8")
    artifacts_path = os.path.join(os.path.curdir, f"artifacts/{run_id}")
    logs_path = os.path.join(artifacts_path, "tensorboard")
    checkpoints_dir = os.path.join(os.path.curdir, "checkpoints")
    RunningAverage(output_transform=lambda x: x).attach(trainer, name="loss")
    pbar = ProgressBar(persist=True, bar_format="{desc}[{n_fmt}/{total_fmt}] {percentage:3.0f}%|{bar}{postfix} [{elapsed}<{remaining}]{rate_fmt}")
    pbar.attach(trainer, metric_names="all")

    @trainer.on(Events.EPOCH_STARTED)  # noqa
    def log_epoch_start(engine: Engine):
        logging.info(f"Started epoch {engine.state.epoch}")
        logging.info(system_status())

    @trainer.on(Events.EPOCH_COMPLETED)  # noqa
    def log_epoch_summary(engine: Engine):
        # Training
        train_evaluator.run(train_loader)
        metrics = train_evaluator.state.metrics
        train_avg_loss = metrics["loss"]

        # Validation
        validation_evaluator.run(val_loader)
        metrics = validation_evaluator.state.metrics
        val_avg_loss = metrics["val_loss"]

        msg = f"Epoch summary for epoch {engine.state.epoch}: loss: {train_avg_loss:.4f}, val_loss: {val_avg_loss:.4f}\n"
        pbar.log_message(msg)
        logging.info(msg)
        logging.info(system_status())

    tb_logger = TensorboardLogger(log_dir=logs_path)
    tb_logger.attach(trainer, log_handler=GradsHistHandler(train_model), event_name=Events.ITERATION_COMPLETED)
    tb_logger.attach_output_handler(
        train_evaluator, event_name=Events.EPOCH_COMPLETED, tag="train", metric_names=["loss"], global_step_transform=global_step_from_engine(trainer)
    )
    tb_logger.attach_output_handler(
        validation_evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag="validation",
        metric_names=["val_loss"],
        global_step_transform=global_step_from_engine(trainer),
    )
    to_save = {"train_model": train_model, "optimizer": optimizer}
    checkpoint_handler = Checkpoint(to_save, DiskSaver(checkpoints_dir, create_dir=True, require_empty=False), n_saved=1)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler)
    # Run Training
    logging.info("Start training of train_model %s on %s for %s epochs", train_model, device, epochs)
    logging.info(f"tensorboard --logdir={artifacts_path}")
    trainer.run(train_loader, max_epochs=epochs)
    pbar.close()


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_str", type=str, help="One of configurations, e.g. 'unet'.", default="unet", required=False)
    parser.add_argument("--resume_checkpoint", type=str, help="torch pt file to be re-loaded.", default=None, required=False)
    parser.add_argument("--data_raw_path", type=str, help="Base dir of raw data", default="./data/raw")
    parser.add_argument(
        "--data_compressed_path",
        type=str,
        help="If given, data is extracted from this location if no data at  data_raw_path  Standard layout: ./data/compressed",
        default=None,
    )
    parser.add_argument("-log", "--loglevel", default="info", help="Provide logging level. Example --loglevel debug, default=warning")
    parser.add_argument("--random_seed", type=int, default=123, required=False, help="Seed for shuffling the dataset.")
    parser.add_argument("--train_fraction", type=float, default=0.9, required=False, help="Fraction of the data set for training.")
    parser.add_argument("--val_fraction", type=float, default=0.1, required=False, help="Fraction of the data set for validation.")
    parser.add_argument("--batch_size", type=int, default=5, required=False, help="Batch Size for training and validation.")
    parser.add_argument("--num_workers", type=int, default=10, required=False, help="Number of workers for data loader.")
    parser.add_argument("--epochs", type=int, default=20, required=False, help="Number of epochs to train.")
    parser.add_argument("--file_filter", type=str, default=None, required=False, help='Filter files in the dataset. Defaults to "**/*8ch.h5"')
    parser.add_argument("--limit", type=int, default=None, required=False, help="Cap dataset size at this limit.")
    parser.add_argument("--device", type=str, default=None, required=False, help="Force usage of device.")
    parser.add_argument("--device_ids", nargs="*", default=None, required=False, help="Whitelist of device ids. If not given, all device ids are taken.")
    parser.add_argument("--data_parallel", default=False, required=False, help="Use DataParallel.", action="store_true")
    parser.add_argument("--num_tests_per_file", default=100, type=int, required=False, help="Number of test slots per file")
    parser.add_argument(
        "--ground_truth_dir",
        type=str,
        default=None,
        required=False,
        help='If given, submission is evaluated from ground truth zips "ground_truth_[spatio]tempmoral.zip" from this directory.',
    )
    parser.add_argument(
        "--submission_output_dir", type=str, default=None, required=False, help="If given, submission is stored to this directory instead of current.",
    )
    parser.add_argument("-c", "--competitions", nargs="+", help="<Required> Set flag", default=["temporal", "spatiotemporal"])
    return parser


def main(args):
    parser = create_parser()
    args = parser.parse_args(args)

    t4c_apply_basic_logging_config()

    model_str = args.model_str
    resume_checkpoint = args.resume_checkpoint

    device = args.device
    competitions = args.competitions

    logging.info("Start build dataset")
    # Data set
    dataset_config = configs[model_str].get("dataset_config", {})

    data_raw_path = args.data_raw_path
    file_filter = args.file_filter

    geometric = configs[model_str].get("geometric", False)
    if data_raw_path is not None:
        logging.info("Check if files need to be untarred...")
        if args.data_compressed_path is not None:
            tar_files = list(Path(args.data_compressed_path).glob("**/*.tar"))
            logging.info("Going to untar %s tar balls to %s. ", len(tar_files), data_raw_path)
            untar_files(files=tar_files, destination=data_raw_path)
            logging.info("Done untar %s tar balls to %s.", len(tar_files), data_raw_path)

    if geometric:
        dataset = T4CGeometricDataset(root=str(Path(data_raw_path).parent), file_filter=file_filter, num_workers=args.num_workers, **dataset_config)
    else:
        dataset = T4CDataset(root_dir=data_raw_path, file_filter=file_filter, **dataset_config)
    logging.info("Dataset has size %s", len(dataset))
    assert len(dataset) > 0

    # Model
    logging.info("Create train_model.")
    model_class = configs[model_str]["model_class"]
    model_config = configs[model_str].get("model_config", {})
    model = model_class(**model_config)
    if not model_str.startswith("naive"):
        dataloader_config = configs[model_str].get("dataloader_config", {})
        optimizer_config = configs[model_str].get("optimizer_config", {})
        if resume_checkpoint is not None:
            logging.info("Reload checkpoint %s", resume_checkpoint)
            load_torch_model_from_checkpoint(checkpoint=resume_checkpoint, model=model)

        logging.info("Going to run train_model.")
        logging.info(system_status())
        _, device = run_model(
            train_model=model, dataset=dataset, dataloader_config=dataloader_config, optimizer_config=optimizer_config, geometric=geometric, **(vars(args))
        )

    for competition in competitions:
        additional_args = {}
        if geometric:
            processed_dir = str(Path(data_raw_path).parent)
            additional_args = {
                "gt": GraphTransformer(processed_dir=processed_dir, raw_dir=data_raw_path, batch_size=1),
                "processed_dir": processed_dir,
            }
        submission = package_submission(
            data_raw_path=data_raw_path,
            competition=competition,
            model=model,
            model_str=model_str,
            device=device,
            h5_compression_params={"compression_level": None},
            submission_output_dir=Path(args.submission_output_dir if args.submission_output_dir is not None else "."),
            # batch mode for submission
            batch_size=1 if geometric else args.batch_size,
            num_tests_per_file=args.num_tests_per_file,
            **additional_args,
        )
        ground_truth_dir = args.ground_truth_dir
        if ground_truth_dir is not None:
            ground_truth_dir = Path(ground_truth_dir)
            scorecomp.score_participant(ground_truth_archive=str(ground_truth_dir / f"ground_truth_{competition}.zip"), input_archive=str(submission))
        else:
            scorecomp.verify_submission(input_archive=submission, competition=competition)


if __name__ == "__main__":
    main(sys.argv[1:])
