"""Public API for inference."""
from vox.inference.autotune import get_scale, preserve_vibrato, snap_to_scale
from vox.inference.pipeline import InferencePipeline, InferenceRequest, InferenceResult
from vox.inference.style_blend import slerp, slerp_barycentric

__all__ = [
    "InferencePipeline",
    "InferenceRequest",
    "InferenceResult",
    "get_scale",
    "preserve_vibrato",
    "snap_to_scale",
    "slerp",
    "slerp_barycentric",
]
