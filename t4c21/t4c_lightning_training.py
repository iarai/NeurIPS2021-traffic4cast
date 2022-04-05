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
import traceback

import pytorch_lightning as pl
import torch

from t4c21.t4c_lightning_datamodule import T4CDataModule
from t4c21.t4c_lightning_datamodule import T4CDataset
from t4c21.t4c_lightning_system import T4CSystem
from t4c21.util.logging import t4c_apply_basic_logging_config
from t4c21.vanilla_unet import UNet


def main(loglevel: str, root_dir: str):
    t4c_apply_basic_logging_config(loglevel=loglevel.upper())
    system = T4CSystem(
        model=UNet(in_channels=96, n_classes=48, depth=5, wf=6, padding=True, up_mode="upconv", batch_norm=True),
        criterion=torch.nn.MSELoss(),
        optimizer_cls=torch.optim.Adam,
        optimizer_parameters={"lr": 1e-4},
        scheduler_cls=torch.optim.lr_scheduler.StepLR,
        scheduler_parameters={"step_size": 1, "gamma": 1.0},
        lr_scheduler_parameters={"interval": "epoch"},
    )
    data_module = T4CDataModule(
        val_train_split=0.9,
        num_workers=10,
        batch_size={"train": 4, "val": 4},
        dataset_cls=T4CDataset,
        dataset_parameters={"root_dir": root_dir, "file_filter": "BERLIN/training/*8ch.h5", "limit": 100},
        dataloader_config={},
    )
    trainer = pl.Trainer(
        gpus=None,
        max_epochs=1,
        progress_bar_refresh_rate=10,
        deterministic=True,
    )
    trainer.fit(system, data_module)


def create_parser() -> argparse.ArgumentParser:
    """Creates the argument parser for this program."""
    parser = argparse.ArgumentParser(description=("Extract distances and angles from the checkpoints into npz files."))
    parser.add_argument("-log", "--loglevel", default="info", help="Log level.")
    parser.add_argument("--root_dir", type=str, default="./data/raw")
    return parser


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
