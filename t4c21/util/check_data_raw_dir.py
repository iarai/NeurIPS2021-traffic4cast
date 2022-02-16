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
from pathlib import Path

from importlib_resources import files

import data


def check_raw_data_dir(data_raw_dir: str, show_missing=False):  # noqa
    """Check whether all expected raw files are present in the expected
    location.

    Parameters
    ----------
    data_raw_dir: str
        Location to check.
    show_missing: bool=False
        Print list of missing files
    """
    expected_files = files(data).joinpath("index.txt").read_text().split(os.linesep)
    expected_files = [Path(data_raw_dir) / Path(f) for f in expected_files]
    actual_files = list(Path(data_raw_dir).rglob("**/*.h5"))

    difference = set(actual_files).difference(set(expected_files))
    python_check = "(\u2713)" if len(difference) == 0 else "(\u2717)"
    print(f"File structure {data_raw_dir}:  {python_check}")
    if len(difference) > 0 and show_missing:
        print(f"Missing files in {data_raw_dir}: {difference}")
