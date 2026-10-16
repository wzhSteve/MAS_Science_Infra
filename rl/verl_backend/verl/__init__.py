# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os

# 兼容新 setuptools：pkg_resources 在 setuptools>=81 后移除，改为可选导入
try:
    import pkg_resources
    from pkg_resources import DistributionNotFound
except ImportError:  # setuptools>=81
    import importlib.metadata as _im

    class DistributionNotFound(Exception):
        pass

    class _PkgResourcesShim:
        """最小化 shim：仅提供 verl 用到的 get_distribution / require 接口"""

        @staticmethod
        def get_distribution(name):
            try:
                return _im.metadata(name)
            except _im.PackageNotFoundError as e:
                raise DistributionNotFound(str(e))

        @staticmethod
        def require(*names):
            for n in names:
                _PkgResourcesShim.get_distribution(n)

    pkg_resources = _PkgResourcesShim()
from packaging.version import parse as parse_version

from .protocol import DataProto
from .utils.device import is_npu_available
from .utils.logging_utils import set_basic_config

version_folder = os.path.dirname(os.path.join(os.path.abspath(__file__)))

with open(os.path.join(version_folder, "version/version")) as f:
    __version__ = f.read().strip()


set_basic_config(level=logging.WARNING)


__all__ = ["DataProto", "__version__"]

if os.getenv("VERL_USE_MODELSCOPE", "False").lower() == "true":
    import importlib

    if importlib.util.find_spec("modelscope") is None:
        raise ImportError("You are using the modelscope hub, please install modelscope by `pip install modelscope -U`")
    # Patch hub to download models from modelscope to speed up.
    from modelscope.utils.hf_util import patch_hub

    patch_hub()

if is_npu_available:
    package_name = 'transformers'
    required_version_spec = '4.51.0'
    try:
        installed_version = pkg_resources.get_distribution(package_name).version
        installed = parse_version(installed_version)
        required = parse_version(required_version_spec)

        if not installed >= required:
            raise ValueError(f"{package_name} version >= {required_version_spec} is required on ASCEND NPU, current version is {installed}.")
    except DistributionNotFound:
        raise ImportError(
            f"package {package_name} is not installed, please run pip install {package_name}=={required_version_spec}")
