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
import numpy as np

from metrics.mse import mse


def test_mse():
    # Simple test
    model_input = np.asarray([0.5, 0.75])
    model_output = np.asarray([0.2, 0.5])
    expected = (0.3**2 + 0.25**2) / 2
    actual = mse(model_input, model_output)
    assert np.isclose(actual, expected)

    # Test torch vs numpy
    actual = (np.square(np.subtract(model_input, model_output))).mean(axis=0)
    assert np.isclose(actual, expected)

    # Test indices
    model_input = np.asarray([[[[0.5, 0.75, 0.25]], [[0.5, 0.75, 0.25]]]])
    assert model_input.shape == (1, 2, 1, 3)
    model_output = np.asarray([[[[0.2, 0.75, 0.5]], [[0.4, 0.75, 0.75]]]])
    assert model_output.shape == model_input.shape
    expected = (0.3**2 + 0.25**2 + 0.1**2 + 0.5**2) / 4
    actual = mse(model_input, model_output, indices=[0, 2])
    assert np.isclose(actual, expected)

    # Test masking
    model_input = np.asarray([[[[0.5, 0.75, 0.25]], [[0.5, 0.75, 0.25]]]])
    assert model_input.shape == (1, 2, 1, 3)
    model_output = np.asarray([[[[0.2, 0.75, 0.5]], [[0.4, 0.75, 0.75]]]])
    assert model_output.shape == model_input.shape
    # The mask doesn't need the first dimension as we do broadcast
    mask = np.asarray([[[0, 1, 1]], [[1, 0, 0]]])
    assert mask.shape == (2, 1, 3)
    # Without mask normalization
    expected = (0.0**2 + 0.0**2 + 0.25**2 + 0.1**2 + 0.0**2 + 0.0**2) / 6
    actual = mse(model_input, model_output, mask=mask, mask_norm=False)
    assert np.isclose(actual, expected)
    # With mask normalization
    expected = (0.0**2 + 0.25**2 + 0.1**2) / 3
    actual = mse(model_input, model_output, mask=mask)
    assert np.isclose(actual, expected)
