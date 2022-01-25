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
import shutil

import GPUtil
import psutil
from tabulate import tabulate


def system_status() -> str:
    s = _make_title("GPU Details")
    try:
        gpus = GPUtil.getGPUs()
        list_gpus = []
        for gpu in gpus:
            # get the GPU id
            gpu_id = gpu.id
            # name of GPU
            gpu_name = gpu.name
            # get % percentage of GPU usage of that GPU
            gpu_load = f"{gpu.load * 100}%"
            # get free memory in MB format
            gpu_free_memory = f"{gpu.memoryFree}MB"
            # get used memory
            gpu_used_memory = f"{gpu.memoryUsed}MB"
            # get total memory
            gpu_total_memory = f"{gpu.memoryTotal}MB"
            # get GPU temperature in Celsius
            gpu_temperature = f"{gpu.temperature} °C"
            gpu_uuid = gpu.uuid
            list_gpus.append((gpu_id, gpu_name, gpu_load, gpu_free_memory, gpu_used_memory, gpu_total_memory, gpu_temperature, gpu_uuid))
        s += tabulate(list_gpus, headers=("id", "name", "load", "free memory", "used memory", "total memory", "temperature", "uuid"))
    except Exception as e:
        s += str(e)
    s += "\n"

    s += _make_title("System memory usage")
    mem = psutil.virtual_memory()
    virtual_memory_fields = ["total", "available", "percent", "used", "free", "active", "inactive", "buffers", "cached", "shared", "slab"]
    virtual_memory_fields = [f for f in virtual_memory_fields if hasattr(mem, f)]
    s += tabulate([[str(mem.__getattribute__(a)) for a in virtual_memory_fields]], headers=virtual_memory_fields) + "\n"

    s += _make_title("Disk usage")
    du = psutil.disk_usage("/")
    du_fields = ["total", "used", "free", "percent"]
    du_fields = [f for f in du_fields if hasattr(du, f)]
    s += tabulate([[str(du.__getattribute__(a)) for a in du_fields]], headers=du_fields)
    return s


def _make_title(title):
    s = "=" * 40 + title + "=" * 40 + "\n"
    return s


def disk_usage_human_readable(path):
    du = shutil.disk_usage(path)
    return f"usage(total={du[0] / (1024 * 1024 * 1024):.2f}GB, used={du[1] / (1024 * 1024 * 1024):.2f}GB, free={du[2] / (1024 * 1024 * 1024):.2f}GB)"
