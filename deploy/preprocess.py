"""Static B1 image input: NumPy/OpenCV only, shared by future PT and ORT runners."""
from dataclasses import dataclass

import cv2
import numpy as np


INPUT_SHAPE_HW = (640, 640)
LETTERBOX_OPTIONS = {
    "new_shape": [640, 640], "auto": False, "scale_fill": False,
    "scaleup": True, "center": True, "stride": 32,
    "padding_value": 114, "interpolation": "cv2.INTER_LINEAR",
}


@dataclass(frozen=True)
class LetterboxGeometry:
    """Shapes are (height, width); ratio is (x, y); padding is (left, top, right, bottom)."""
    original_shape_hw: tuple
    input_shape_hw: tuple
    resized_shape_hw: tuple
    ratio_xy: tuple
    padding_ltrb: tuple

    def to_dict(self):
        return {key: list(getattr(self, key)) for key in (
            "original_shape_hw", "input_shape_hw", "resized_shape_hw", "ratio_xy", "padding_ltrb")}


def validate_preprocessing_contract(metadata):
    """Call once when loading B1; do not silently accept dynamic/FP16/auto-padding models."""
    expected = [{"name": "images", "shape": [1, 3, 640, 640], "onnx_dtype": 1}]
    contract = metadata.get("preprocessing_contract", {})
    if (metadata.get("graph", {}).get("inputs") != expected or
            contract.get("input") != "BGR uint8 HWC" or
            contract.get("model_input") != "RGB float32 NCHW / 255, contiguous" or
            contract.get("letterbox") != LETTERBOX_OPTIONS):
        raise ValueError("Expected the B1 static 640 / FP32 / centered letterbox input contract")


def preprocess_bgr(image):
    """Return contiguous RGB float32 [1,3,640,640] plus geometry, without changing image.

    ratio_xy preserves the ideal official gain r, not the rounded resize width / original width.
    The rounded content size and integer padding are recorded separately for later mask handling.
    This function does not restore boxes/masks or perform model inference.
    """
    if (not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3 or
            image.dtype != np.uint8 or min(image.shape[:2]) <= 0):
        raise ValueError("Expected a nonempty BGR uint8 HWC image with exactly three channels")
    height, width = image.shape[:2]
    target_height, target_width = INPUT_SHAPE_HW
    ratio = min(target_height / height, target_width / width)
    resized_width, resized_height = round(width * ratio), round(height * ratio)
    if min(resized_width, resized_height) <= 0:
        raise ValueError("Aspect ratio produces a zero-sized letterbox resize")
    half_width = (target_width - resized_width) / 2
    half_height = (target_height - resized_height) / 2
    left, right = round(half_width - .1), round(half_width + .1)
    top, bottom = round(half_height - .1), round(half_height + .1)
    resized = image if (width, height) == (resized_width, resized_height) else cv2.resize(
        image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=(114, 114, 114))
    tensor = np.ascontiguousarray(padded[..., ::-1].transpose(2, 0, 1)[None], dtype=np.float32)
    tensor /= np.float32(255)
    geometry = LetterboxGeometry((height, width), INPUT_SHAPE_HW,
                                (resized_height, resized_width), (ratio, ratio),
                                (left, top, right, bottom))
    return tensor, geometry
