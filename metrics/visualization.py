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
from typing import Optional
from typing import Union

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from IPython.display import HTML


def plot_prediction(test_data: np.ndarray, ground_truth_prediction: np.ndarray, prediction: np.ndarray):  # noqa
    """Display test data along with ground truth and prediction in a 18 x 3
    grid of subplots. The sum over all channels in the last dimension is
    displayed.

    Parameters
    ----------
    test_data
    ground_truth_prediction
    prediction
    """
    _, axs = plt.subplots(18, 3, figsize=(30, 150))
    for i in range(12):
        axs[i, 0].set_title(f"test_data t=-{(12 - i) * 5}min, sum of all channels")
        axs[i, 0].imshow(test_data[i].sum(axis=-1))
        axs[i, 1].set_title(f"no prediction at t=-{(12 - i) * 5}min")
        axs[i, 1].imshow(np.zeros((495, 436)))
        axs[i, 2].set_title(f"no prediction at t=-{(12 - i) * 5}min")
        axs[i, 2].imshow(np.zeros((495, 436)))
    for i in range(6):
        axs[12 + i, 0].set_title(f"ground truth prediction {i}, sum of all channels")
        axs[12 + i, 0].imshow(ground_truth_prediction[i].sum(-1))
    for i in range(6):
        axs[12 + i, 1].set_title(f"actual prediction {i}")
        axs[12 + i, 1].imshow(prediction[i].sum(-1))
    for i in range(6):
        axs[12 + i, 2].set_title(f"squared prediction error {i}, summed over all channels")
        axs[12 + i, 2].imshow(((ground_truth_prediction[i] - prediction[i]) ** 2).sum(-1))


def animate(tensors: Union[torch.Tensor, np.ndarray], interval=200, replay_delay=1000, file: Optional[str] = None) -> HTML:  # noqa
    """Animate frames stored in numpy or torch tensor.

    Parameters
    ----------
    tensors: Union[torch.Tensor,np.ndarray]
        images
    interval: int
        frame interval
    replay_delay: int
        interval before the movie is replayed

    file: str=None
        if given, the movie is stored to this file as well.

    Returns
    -------
        iPython HTML movie
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    imgs = []
    for img in tqdm.tqdm(tensors):
        if hasattr(img, "numpy"):
            img = img.numpy()

        img = ax.imshow(img, animated=True)
        imgs.append([img])

    ani = animation.ArtistAnimation(fig, imgs, interval=interval, blit=True, repeat_delay=replay_delay)
    if file is not None:
        ani.save(file)
    return HTML(ani.to_html5_video())
