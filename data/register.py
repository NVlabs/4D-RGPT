# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from .base import BaseDatamodule

_DATAMODULES = {}

def register_datamodule():

    def decorator(module_class):
        module_name = module_class.dirname
        if module_name in _DATAMODULES:
            raise ValueError(f"Datamodule {module_name} already registered.")
        _DATAMODULES[module_name] = module_class
        return module_class

    return decorator

def get_datamodule(name: str) -> BaseDatamodule:
    if name not in _DATAMODULES:
        raise ValueError(f"Datamodule {name} not registered. Available: {list(_DATAMODULES.keys())}.")
    return _DATAMODULES[name]()
