# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import yaml
from dotenv import load_dotenv

load_dotenv()

# Default to a path resolved from this file, not CWD — `llava` transitively
# imports `data`, so this module loads in contexts (e.g. lmms-eval launched
# from a cache dir) where CWD isn't the repo root.
_DEFAULT_REGISTRY_FILE = os.path.join(os.path.dirname(__file__), "registry.yaml")

DATASETS_ROOT: str = os.getenv("DATASETS_ROOT") if os.getenv("DATASETS_ROOT") is not None else "datasets" # type: ignore
REGISTRY_FILE: str = os.getenv("REGISTRY_PATH") if os.getenv("REGISTRY_PATH") is not None else _DEFAULT_REGISTRY_FILE # type: ignore
with open(REGISTRY_FILE, 'r') as f:
    DATASETS: dict = yaml.safe_load(f)
