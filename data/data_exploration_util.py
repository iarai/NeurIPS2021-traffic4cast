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
import glob
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data.data_layout import channel_labels
from data.data_layout import volume_channel_indices
from util.h5_util import load_h5_file


# In order to be used as function in multiprocessing jupyter requires passed functions to be
# imported from a separate file. Hence, we're defining all larger processing functions in here.
def process_log_sums_per_city_and_year(task):  # noqa
    city, base_folder, num_files_per_city_and_year = task
    log_of_sum_per_city_and_year = {}
    log_of_sum_per_city = {city: np.zeros((495, 436))}
    for year in [2019, 2020]:
        files = glob.glob(f"{base_folder}/{city}/training/*{year}*8ch.h5", recursive=True)
        if len(files) == 0:
            print(f"no files for {city} {year}")
            continue
        log_of_sum_per_city_and_year[(city, year)] = np.zeros((495, 436))
        for f in files[:num_files_per_city_and_year]:
            data = load_h5_file(f)
            if data.shape != (288, 495, 436, 8):
                print(f"!!!! {f} {data.shape}")
                continue
            log_of_sum_per_city_and_year[(city, year)] += data.sum(axis=(0, -1))
            log_of_sum_per_city[city] += data.sum(axis=(0, -1))
        log_of_sum_per_city_and_year[(city, year)] = np.log(log_of_sum_per_city_and_year[(city, year)])
    log_of_sum_per_city[city] = np.log(log_of_sum_per_city[city])
    return list(log_of_sum_per_city.items()), list(log_of_sum_per_city_and_year.items())


# In order to be used as function in multiprocessing jupyter requires passed functions to be
# imported from a separate file. Hence, we're defining all larger processing functions in here.
def process_sum_per_bin_per_city_year_and_month(task):  # noqa
    city, base_folder, num_files, years, months = task
    sum_per_bin_per_city_and_year = {}
    for (year, month) in itertools.product(years, months):
        files = sorted(glob.glob(f"{base_folder}/{city}/training/*{year}-{month}*8ch.h5", recursive=True))
        if len(files) == 0:
            print(f"no files for {city} {year}")
            continue
        sum_per_bin_per_city_and_year[(city, year, month)] = np.zeros((288,))
        for f in files[:num_files]:
            print(f)
            data = load_h5_file(f)
            if data.shape != (288, 495, 436, 8):
                print(f"!!!! {f} {data.shape}")
                continue
            sum_per_bin_per_city_and_year[(city, year, month)] += data[:, :, :, volume_channel_indices].sum(axis=(1, 2, 3))
        sum_per_bin_per_city_and_year[(city, year, month)] /= len(files)
    return list(sum_per_bin_per_city_and_year.items())


def plot_stats_file(city, r=None, c=None, normalized=False, stats_dir="./stats"):  # noqa
    stats_dir = Path(stats_dir)
    stats_dir.mkdir(parents=True, exist_ok=True)
    extension = "distribution"
    if normalized:
        extension = "distribution_normalized_nonzeros"
    data = np.concatenate([load_h5_file(f"{str(stats_dir)}/{city}_{w}_{extension}.h5") for w in range(7)], axis=1)
    print(data.shape)
    print(data.dtype)

    mean, std, counts = data
    mean_orig = mean
    if r is not None:
        r_ub = r + 1
        c_ub = c + 1
    else:
        r = 0
        c = 0
        r_ub = 495
        c_ub = 436

    mean = np.mean(mean[:, r:r_ub, c:c_ub, :], axis=(1, 2))
    std = np.mean(std[:, r:r_ub, c:c_ub, :], axis=(1, 2))
    counts = np.mean(counts[:, r:r_ub, c:c_ub, :], axis=(1, 2))

    if normalized:

        print(mean.dtype)
        for ch in [1, 3, 5, 7]:
            speed_data = mean_orig[:, r:r_ub, c:c_ub, ch]
            volume_data = mean_orig[:, r:r_ub, c:c_ub, ch - 1]
            s = np.sum(speed_data, axis=(1, 2))

            speed_normalized_mean = np.zeros_like(s)
            # renormalization constant `k`: volume_data is zero iff all (numerically almost all) raw data points are zero
            k = np.count_nonzero(volume_data, axis=(1, 2))
            non_zero = k != 0
            speed_normalized_mean[non_zero] = s[non_zero] / k[non_zero]
            mean[:, ch] = speed_normalized_mean

            speed_normalized_mean = np.expand_dims(speed_normalized_mean, -1)
            speed_normalized_mean = np.repeat(speed_normalized_mean, repeats=495, axis=-1)
            speed_normalized_mean = np.expand_dims(speed_normalized_mean, -1)
            speed_normalized_mean = np.repeat(speed_normalized_mean, repeats=436, axis=-1)

            std[:, ch] = np.sqrt(
                np.mean(np.square(np.where(speed_normalized_mean > 0, speed_data, speed_normalized_mean) - speed_normalized_mean), axis=(1, 2))
            )

    # means with error bars and zero ratio
    fig, axs = plt.subplots(4, 2, figsize=(100, 50))
    for ch in range(8):
        ax = axs[ch // 2, ch % 2]
        ax.set_title(f"{channel_labels[ch]} mean, std and zero volume ratio for {city} at {(r, c)}")

        ax.plot(np.arange(24 * 7), mean[:, ch], label="mean", color="blue")
        if not normalized:
            ax.errorbar(np.arange(24 * 7), mean[:, ch], yerr=std[:, ch], fmt="o")
        ax.set_ylim((-20, 300))
        ax2 = ax.twinx()
        ax2.plot(np.arange(24 * 7), 255 - counts[:, ch] * 255, color="green", label="zero volume ratio")
        ax2.set_ylim((0, 255))
        # TODO add ticks to see the scale better

    # std comparison
    fig, axs = plt.subplots(4, 2, figsize=(70, 50))
    for ch in range(8):
        #  TODO colors misleading. Use same color for volume and speed in the above mean plots
        ax = axs[ch // 2, ch % 2]
        ax.set_title(f"std and zero ratio {channel_labels[ch]} for {city} at {(r, c)}")
        ax.plot(np.arange(24 * 7), std[:, ch], color="blue", label=f"std")
        ax.plot(np.arange(24 * 7), 255 - counts[:, ch] * 255, color="green", label="zero ratio")
        ax.legend()

    # histogram
    fig, axs = plt.subplots(4, 2, figsize=(100, 50))

    for ch in [0, 2, 4, 6]:
        ax = axs[ch // 2, ch % 2]
        ax.set_yscale("log")
        ax.set_title(f"{channel_labels[ch]}")
        counts, bins = np.histogram(mean_orig[:, :, :, ch].flatten(), bins=np.arange(0, 255, 1))
        ax.hist(bins[:-1], bins, weights=counts, color="blue")

    for ch in [1, 3, 5, 7]:
        ax = axs[ch // 2, ch % 2]
        ax.set_title(f"{channel_labels[ch]}")
        counts, bins = np.histogram(mean[:, ch].flatten(), bins=np.arange(0, 255, 1))
        ax.hist(bins[:-1], bins, weights=counts, color="blue")
