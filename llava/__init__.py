import os
import sys

# Make the bundled L4P submodule (third_party/L4P/l4p) importable as `l4p`
# without a pip install. Upstream L4P has no pyproject.toml of its own, so we
# inject the submodule root onto sys.path on package import. Idempotent;
# silently no-op if the submodule isn't checked out yet.
_L4P_SUBMODULE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "third_party", "L4P",
)
if os.path.isdir(os.path.join(_L4P_SUBMODULE, "l4p")) and _L4P_SUBMODULE not in sys.path:
    sys.path.insert(0, _L4P_SUBMODULE)


def _patch_qwen_vl_utils_torchvision_fallback() -> None:
    """qwen-vl-utils (0.0.14, latest) hard-codes a fallback to
    `torchvision.io.read_video` when its primary backend errors. That API
    was removed in torchvision 0.27, so the fallback raises a confusing
    `AttributeError: module 'torchvision.io' has no attribute 'read_video'`
    that masks the real (primary-backend) failure.

    We can't ship a fix upstream without bumping the pinned package, so we
    replace the broken fallback with one that re-raises the primary error
    using a clearer message. Idempotent; skipped if qwen-vl-utils is absent.
    """
    try:
        from qwen_vl_utils import vision_process as _vp
    except ImportError:
        return

    def _fail_torchvision(ele):
        raise RuntimeError(
            "qwen-vl-utils torchvision fallback is disabled: "
            "torchvision>=0.27 removed `torchvision.io.read_video`. The "
            "primary backend (torchcodec or decord) must have failed first — "
            "check the warning above for the underlying cause."
        )

    _vp.VIDEO_READER_BACKENDS["torchvision"] = _fail_torchvision


_patch_qwen_vl_utils_torchvision_fallback()

from .entry import *
from .media import *
