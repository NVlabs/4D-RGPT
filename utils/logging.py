# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import typing

if typing.TYPE_CHECKING:
    from loguru import Logger
else:
    Logger = None

__all__ = ["logger"]


def __get_logger() -> Logger:
    from loguru import logger

    return logger


logger = __get_logger()
