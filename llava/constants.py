# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

# This file is modified from https://github.com/haotian-liu/LLaVA/

CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
DEFAULT_IMAGE_TOKEN = "<image>"

SENTINEL_TOKEN = "<vila/sentinel>"
MEDIA_TOKENS = {
    "image": "<image>",
    "video": "<vila/video>",
}

# Legacy haotian-liu/LLaVA + VILA-1.x constants. NOT used by 4D-RGPT's own
# pipeline (we use MEDIA_TOKENS + SENTINEL_TOKEN), but the upstream
# lmms-eval `vila` model adapter imports them from `llava.constants` at
# module load. Provided so `--model vila` works without forking the adapter.
# At runtime they only fire when `config.mm_use_im_start_end is True`, which
# is False for NVILA-Lite-8B and other current VILA checkpoints.
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_TOKEN_INDEX = -200
