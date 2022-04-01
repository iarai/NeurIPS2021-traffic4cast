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
import itertools
import logging
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch
import tqdm

from baselines.naive_weighted_average import NaiveWeightedAverage
from util.data_range import generate_date_range
from util.data_range import weekday_parser
from util.h5_util import load_h5_file
from util.h5_util import write_data_to_h5
from util.logging import t4c_apply_basic_logging_config

spatio_temporal_cities = ["VIENNA", "CHICAGO"]
temporal_cities = ["BERLIN", "ISTANBUL", "MOSCOW", "NEWYORK"]


class NaiveWeightedAverageWithStats(NaiveWeightedAverage):
    def __init__(self, stats_dir: Path, w_stats: float = 0.0, w_random: float = 0.5):
        """
        Parameters
        ----------
        stats_dir: Path
            where CITY_WEEKDAY_[means|zeros].h5 are stored
        w_stats: float
            weight `w_stats` given to hourly and daily mean from stats and `1-w_stats` given to mean in test input for each channel.
        w_random
            scale fraction of no volume by `w_random` when sampling no data.
        """
        super(NaiveWeightedAverageWithStats, self).__init__()
        self.w_stats = w_stats
        self.w_random = w_random
        self.means = {}
        self.zeros = {}
        for city in spatio_temporal_cities + temporal_cities:
            for weekday in range(7):
                self.means[(city, weekday)] = load_h5_file(str(stats_dir / f"{city}_{weekday}_means.h5"))
                self.zeros[(city, weekday)] = load_h5_file(str(stats_dir / f"{city}_{weekday}_zeros.h5"))
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    logging.debug("%s on %s has means %s", city, weekday, np.unique(self.means[(city, weekday)]))
                    logging.debug("%s on %s has zeros %s", city, weekday, np.unique(self.zeros[(city, weekday)]))

    def forward(self, x: torch.Tensor, additional_data: torch.Tensor, city: str, *args, **kwargs):
        x = x.numpy()
        additional_data = additional_data.numpy()

        batch_size = x.shape[0]
        y = np.zeros(shape=(batch_size, 6, 495, 436, 8))
        for b in range(batch_size):
            weekday, slot = additional_data[b]

            y_b = (1 - self.w_stats) * np.mean(x[b], axis=0) + self.w_stats * self.means[(city, weekday)][slot % 24]
            assert y_b.shape == (495, 436, 8)
            y_b = np.expand_dims(y_b, axis=0)
            y_b = np.repeat(y_b, repeats=6, axis=0)
            assert y_b.shape == (6, 495, 436, 8), y_b.shape

            for t in range(6):
                # volume channels
                for ch in [0, 2, 4, 6]:
                    # mask: put pixel to zero with the same probability as in the stats scaled by `w_random`
                    mask = np.where(np.random.random(size=(495, 436)) * 12 * self.w_random < self.zeros[(city, weekday)][slot % 24][:, :, ch], 0, 1)
                    assert mask.shape == (495, 436)
                    # apply mask to volume and corresponding speed channel
                    y_b[t, :, :, ch] = y_b[t, :, :, ch] * mask
                    y_b[t, :, :, ch + 1] = y_b[t, :, :, ch + 1] * mask
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug("unique values in forward item %s: %s ", b, np.unique(y_b))
            y[b] = y_b

        y = torch.from_numpy(y).float()
        return y


def process_hourly_means_and_zero_fraction_by_weekday(
    task: Tuple[str, int, List[str]], data_raw_dir: Path, output_dir: Path, max_num_files: Optional[int] = None
):
    try:
        city, weekday, dates = task
        logging.debug(f"start for {task}")
        files = list(
            itertools.chain(
                *[
                    (data_raw_dir / city).rglob(
                        f"{date}_*8ch.h5",
                    )
                    for date in dates
                ]
            )
        )

        if max_num_files is not None:
            files = files[:max_num_files]

        num_files = len(files)
        if num_files == 0:
            logging.error(f"no files for {city} {weekday} {dates} in {data_raw_dir}")
        data = np.zeros(shape=(num_files, 288, 495, 436, 8), dtype=np.uint8)
        for c, f in enumerate(files):
            logging.debug(f"{city} {weekday} {c}/{num_files}")
            data[c] = load_h5_file(f)

        logging.debug(f"{city} {weekday} reshape")
        data = np.reshape(data, newshape=(num_files, 24, 12, 495, 436, 8))

        logging.debug(f"{city} {weekday} mean/std")
        # aggregate over files and within-hour slots to get hourly mean/std
        mean = np.mean(
            data,
            axis=(0, 2),
        )
        std = np.std(data, axis=(0, 2))
        average_non_zero_volume_counts = np.count_nonzero(data, axis=(0, 2)) / (num_files * 12)
        for ch in [1, 3, 5, 7]:
            average_non_zero_volume_counts[:, :, :, ch] = average_non_zero_volume_counts[:, :, :, ch - 1]

        logging.debug(f"{city} {weekday} stacking")
        stats = []
        stats.append(mean)
        stats.append(std)
        stats.append(average_non_zero_volume_counts)
        stacked_stats = np.stack(stats)
        logging.debug(f"{city} {weekday} stacked {stacked_stats.shape}")

        logging.debug(f"writing for {task}")
        write_data_to_h5(data=stacked_stats, filename=str(output_dir / f"{city}_{weekday}_distribution.h5"), compression_level=6, dtype=stacked_stats.dtype)
        logging.debug(f"done for {task}")

        logging.debug(f"{city} {weekday} mean/std normalized for non-zero volume")
        for ch in [1, 3, 5, 7]:
            speed_data = data[:, :, :, :, :, ch]
            volume_data = data[:, :, :, :, :, ch - 1]

            # `np.mean(..., where=....)` and `np.std(..., where=...)` do not work for too many dimensions, grr...

            # mean over all speeds where volume > 0, aggregated by file (all files have same weekday) and hour
            s = np.sum(speed_data, axis=(0, 2))
            c = np.count_nonzero(volume_data, axis=(0, 2))
            speed_normalized_mean = np.zeros_like(s)
            non_zero = c != 0
            speed_normalized_mean[non_zero] = s[non_zero] / c[non_zero]
            mean[:, :, :, ch] = speed_normalized_mean

            # std by putting normalized mean speed instead of 0 where volume is zero

            # move axis for broadcasting to work...
            speed_data = np.moveaxis(speed_data, 1, 2)
            volume_data = np.moveaxis(volume_data, 1, 2)
            assert speed_data.shape == (num_files, 12, 24, 495, 436), f"{speed_data.shape}"

            std[:, :, :, ch] = np.sqrt(np.mean(np.square(np.where(volume_data > 0, speed_data, speed_normalized_mean) - speed_normalized_mean), axis=(0, 1)))

        stats = []
        stats.append(mean)
        stats.append(std)
        stats.append(average_non_zero_volume_counts)

        logging.debug(f"{city} {weekday} stacking")
        stacked_stats = np.stack(stats)
        logging.debug(f"{city} {weekday} stacked {stacked_stats.shape}")

        logging.debug(f"writing for {task}")
        write_data_to_h5(
            data=stacked_stats,
            filename=str(output_dir / f"{city}_{weekday}_distribution_normalized_nonzeros.h5"),
            compression_level=6,
            dtype=stacked_stats.dtype,
        )
        logging.debug(f"done for {task}")

    except BaseException as e:
        logging.error("something went wrong...", exc_info=e)


def generate_stats_files(city: str, dates: List[str], data_raw_dir: Path, stats_dir: Path, max_num_files: Optional[int] = 3, pool_size: int = 2):
    date2weekday = {d: weekday_parser(d) for d in dates}
    logging.debug(dates)
    logging.debug(date2weekday)

    weekday2dates = {}
    for date, weekday in sorted(date2weekday.items()):
        weekday2dates.setdefault(weekday, []).append(date)

    tasks = [(city, weekday, dates) for city in [city] for weekday, dates in weekday2dates.items()]

    with Pool(processes=pool_size) as pool:

        for _ in tqdm.tqdm(
            pool.imap_unordered(
                partial(process_hourly_means_and_zero_fraction_by_weekday, data_raw_dir=data_raw_dir, output_dir=stats_dir, max_num_files=max_num_files),
                tasks,
            ),
            total=len(tasks),
        ):
            pass


def main():
    t4c_apply_basic_logging_config(loglevel="DEBUG")
    data_raw_dir = Path("privat/competition_prep/01_normalized-less-compressed")
    stats_dir = Path("privat/competition_prep/stats")
    stats_dir.mkdir(parents=True, exist_ok=True)
    pool_size = 2
    max_num_files = 4

    all_cities = spatio_temporal_cities + temporal_cities
    for city in tqdm.tqdm(all_cities):
        if city in spatio_temporal_cities:
            dates = generate_date_range(f"2019-04-01", f"2019-05-31") + generate_date_range(f"2020-04-01", f"2020-05-31")
        else:
            dates = generate_date_range(f"2020-04-01", f"2020-05-31")
        generate_stats_files(city=city, dates=dates, data_raw_dir=data_raw_dir, stats_dir=stats_dir, max_num_files=max_num_files, pool_size=pool_size)


if __name__ == "__main__":
    main()
