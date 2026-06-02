# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
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
#
# SPDX-License-Identifier: Apache-2.0

__all__ = ["Media", "File", "Image", "Video", "Depth", "Intrinsic"]


class Media:
    pass


class File(Media):
    def __init__(self, path: str) -> None:
        self.path = path


class Image(File):
    pass


class Depth(File):
    pass


class Video(File):
    pass


class Intrinsic(File):
    pass


class Mask(Media):
    def __init__(self, modality_type: str, content) -> None:
        self.modality_type = modality_type
        self.content = content


class Box3D(Media):
    def __init__(self, content) -> None:
        self.content = content


class OBox3DMask(Media):
    def __init__(self, content) -> None:
        self.content = content


class PointMap(Media):
    def __init__(self, content) -> None:
        self.content = content


class PointMapNorm(Media):
    def __init__(self, content) -> None:
        self.content = content


class GTDepth(File):
    def __init__(self, path: str, camera, raw_depth=None) -> None:
        self.path = path
        self.camera = camera
        self.raw_depth = raw_depth
