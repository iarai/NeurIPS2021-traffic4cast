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
from typing import Any
from typing import Optional

import pytorch_lightning as pl
import torch
from overrides import overrides
from pytorch_lightning.utilities.types import STEP_OUTPUT


class T4CSystem(pl.LightningModule):
    """"""

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer_cls: type,
        optimizer_parameters: dict,
        scheduler_cls: type,
        scheduler_parameters: dict,
        lr_scheduler_parameters: dict,
        optimizer_wrapper_cls: Optional[type] = None,
        optimizer_wrapper_parameters: Optional[dict] = None,
        backward_parameters: Optional[dict] = None,
    ):
        super().__init__()

        self.model = model

        # Criterion used to minimize during training
        self.criterion = criterion

        self.optimizer = optimizer_cls(self.model.parameters(), **optimizer_parameters)

        if optimizer_wrapper_cls is not None:
            optimizer_wrapper_parameters = optimizer_wrapper_parameters if optimizer_wrapper_parameters is not None else {}
            self.optimizer = optimizer_wrapper_cls(self.optimizer, **optimizer_wrapper_parameters)

        self.scheduler = scheduler_cls(self.optimizer, **scheduler_parameters)

        self.lr_scheduler_parameters = lr_scheduler_parameters

    @overrides
    def forward(self, x, *args, **kwargs) -> Any:
        x = self.model(x)
        return x

    @overrides
    def training_step(self, batch, batch_idx, *args, **kwargs) -> STEP_OUTPUT:
        inputs = batch[0]
        outputs = self(inputs)

        labels = batch[1]

        loss = self.criterion(outputs, labels)

        # Logging to TensorBoard by default
        self.log("Loss/train", loss, on_epoch=True, on_step=False)
        return loss

    @overrides
    def validation_step(self, batch, batch_idx, *args, **kwargs) -> STEP_OUTPUT:
        inputs = batch[0]
        labels = batch[1]

        outputs = self(inputs)

        loss = self.criterion(outputs, labels)
        self.log("Loss/val", loss, on_epoch=True, on_step=False)

    @overrides
    def test_step(self, batch, batch_idx, *args, **kwargs) -> STEP_OUTPUT:
        inputs = batch[0]
        labels = batch[1]

        outputs = self(inputs)

        loss = self.criterion(outputs, labels)

        self.log("Loss/test", loss)

    @overrides
    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {"scheduler": self.scheduler, **self.lr_scheduler_parameters},
        }
