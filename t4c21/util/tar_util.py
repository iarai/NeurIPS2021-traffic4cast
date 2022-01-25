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
import tarfile
import tempfile
from pathlib import Path
from typing import List
from typing import Optional
from typing import Union

import tqdm


def untar_files(files: List[Union[str, Path]], destination: Optional[str] = None):
    """Untar files to a destination repo."""
    pbar = tqdm.tqdm(files, total=len(files))
    for f in pbar:
        pbar.set_description(str(f))
        with tarfile.open(f, "r") as tar:
            if destination is not None:
                Path(destination).mkdir(exist_ok=True)
                tar.extractall(path=destination)
            else:
                with tempfile.TemporaryDirectory() as temp_dir:
                    tar.extractall(path=temp_dir)
