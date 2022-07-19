## The data.

### Get the data

You can obtain our competition's data if from [HERE](https://developer.here.com/sample-data), available for academic and non-commercial purposes. It consists of a compressed file for each
city that forms part of this core challenge. As a tip of the hat to our 2019/2020 competitions, we aggregate all our data in the same 100mx100m fashion and
maintain the same grid resolution of 495x436 pixels for each city, and we've included the cities of Berlin and Istanbul in the core competition for
comparability.

### Cities

There are three types of different cities:

* **spatial transfer** (core challenge): for 4 cities (Berlin, Istanbul, Melbourne, Chicago), we provide a training set of 181 full training days for pre-Covid 2019 only; 50 tests for in-Covid 2020
  per city.
* **spatio-temporal transfer** (extended challenge): for 2 cities (Vienna and New York), we provide no training data; 50 tests for pre-Covid 2019 and 50 tests for in-Covid 2020 per
  city.
* **training**: for 4 cities (Moscow, Barcelona, Antwerp, Bangkok), we provide a training set of 181 full training days both for pre-Covid 2019 and in-Covid 2020; no tests.


### Dynamic data

The dynamic data for one day is given in a compressed representation of a `(1,288,495,436,8)` tensor. As last year, the first 3 dimensions encode the 5-min time
bin, height and width of each "image". The first two of the 8 channels encode the aggregated volume and average speed of all underlying probes whose heading is
between `0` and `90` degrees (i.e. `NE`), the next two the same aggregation of volume and speed for all probes heading `SE`, the following two for SW and NW,
respectively.

### Static data

Moreover, we will provide two static files for each city,

* `<CITY NAME>_static.h5` contains a `(9,495,436)` tensor. The first channel is a gray-scale representation of the city map in the same resolution as
  the dynamic data. The other 8 layers are a binary encoding of whether the cell is connected to it neighbor cell/pixel to the `N`, `NE`, ..., `NW`.

```python
from competition.prepare_test_data.prepare_test_data import prepare_test
from data.data_layout import static_channel_labels, channel_labels, volume_channel_indices, speed_channel_indices

print(channel_labels)
# -> ['volume_NE', 'speed_NE', 'volume_NW', 'speed_NW', 'volume_SE', 'speed_SE', 'volume_SW', 'speed_SW', 'incidents']
print(static_channel_labels)
# -> ['base_map', 'connectivity_N', 'connectivity_NE', 'connectivity_E', 'connectivity_SE', 'connectivity_S', 'connectivity_SW', 'connectivity_W', 'connectivity_NW']
print(volume_channel_indices)
#  -> [0, 2, 4, 6]
print(speed_channel_indices)
# -> [1, 3, 5, 7]
```
* `<CITY NAME>_static_map_high_res.h5` contains a `(4950,4360)` tensor of higher-res gray-scale map where each pixel corresponds to approximately 10mx10m,
  and it is easy to map the pictures by factor 10. In fact, the lower-res map (first channel) is a down-sampled version of this higher-res map, and participants
  could generate their own static encoding at the prediction resolution. See `competition/static_data_preparation` for details.

### Data provisioning

For each city, we provide a `tar.gz` file with the city name in capital letters and the following folder structure:

```
    +-- data
        + -- compressed                                      <-- tarballs
            + -- <CITY NAME>.tar
                 ...
        + -- raw                                             <-- tarballs extracted
            + -- <CITY NAME>                                 <-- "training city" (4)
                 + -- <CITY NAME>_static.h5
                 + -- <CITY NAME>_map_high_res.h5
                 + -- training --
                        + -- 2019-01-01_<CITY NAME>_8ch.h5
                            ...
                        + -- 2019-06-30_<CITY NAME>_8ch.h5
                        + -- 2020-01-01_<CITY NAME>_8ch.h5
                            ...
                        + -- 2020-06-30_<CITY NAME>_8ch.h5
            + -- <CITY NAME>                                <-- "city for core challenge" (4)
                 + -- <CITY NAME>_static.h5
                 + -- <CITY NAME>_map_high_res.h5
                 + -- training --
                        + -- 2019-01-01_<CITY NAME>_8ch.h5
                        + -- 2019-01-02_<CITY NAME>_8ch.h5
                            ...
                + -- <CITY NAME>_test_temporal.h5
                + -- <CITY NAME>_test_additional_temporal.h5
            + -- <CITY NAME>                                <-- "city for extended challenge" (2)
                 + -- <CITY NAME>_static.h5
                 + -- <CITY NAME>_map_high_res.h5
                 + -- <CITY NAME>_test_spatiotemporal.h5
                 + -- <CITY NAME>_test_additional_spatiotemporal.h5

```

### Tests

Any file `<CITY NAME>_test_<temporal|spatiotemporal>.h5` in the testing set contains a tensor of size `(100,12,495,436,8)`.
The `12` indicates that we give 12 successive "images" of our 5min interval time bins, spanning a total of 1h. We will ask participants to predict 5min, 10min
and 15min, 30min, 45min and 60min into the future. Participants will then submit a directory for each city containing the same file names as the files in the
testing data set, but encoding a tensor of dimension `(100,6,495,436,8)`
reflecting the 6 time predictions 5min, 10min
and 15min, 30min, 45min and 60min into the future.


In addition, `<CITY NAME>_test_additional_<temporal|spatiotemporal>.h5` contains a `(100,2)` tensor:
the first channel contains `1` indicates the day of week (`0` = Monday, ..., `6` = Sunday) and the second channel the time of day ot the test slot (`0`, ...`240`) in local time.

Notice that the time slots are local and not UTC in previous competitions.

To practice the transfer, participants can generate their own test sets. See `competition/prepare_test_data` for details.

### Movies

It might be helpful to "see" the movies one can make from our data. You can find / generate clips in the [data exploration notebook](data_exploration.ipynb).



### Blog posts

We also refer to our [blog posts](https://www.iarai.ac.at/traffic4cast/forums/forum/competition/):
- [Competition Data Details](https://www.iarai.ac.at/traffic4cast/forums/topic/competition-data-details/)
- [Exploring the Temporal Shift from pre-COVID to in-COVID](https://www.iarai.ac.at/traffic4cast/forums/topic/exploring-the-temporal-shift-from-pre-covid-to-in-covid/)
- [Exploring the Spatial Data Properties](https://www.iarai.ac.at/traffic4cast/forums/topic/exploring-the-spatial-data-properties/)
- [Looking into data format](https://www.iarai.ac.at/traffic4cast/forums/topic/looking-into-data-format/)
- [Looking into the road graph](https://www.iarai.ac.at/traffic4cast/forums/topic/looking-into-the-road-graph/)

### Get your hands dirty!

Continue in the [data exploration notebook](data_exploration.ipynb).
