"""B1 raw-output postprocessing with NumPy/OpenCV, no training-framework dependency."""
from dataclasses import dataclass

import cv2
import numpy as np

from deploy.preprocess import LetterboxGeometry


@dataclass
class SelectedCandidates:
    boxes_input_xyxy: np.ndarray
    scores: np.ndarray
    class_ids: np.ndarray
    mask_coefficients: np.ndarray
    candidate_indices: np.ndarray
    counts: dict


@dataclass
class ProductInstances:
    """All rows refer to the same instance; masks are uint8 (0/1), [N, original H, original W]."""
    boxes_xyxy: np.ndarray
    scores: np.ndarray
    class_ids: np.ndarray
    masks: np.ndarray
    mask_coefficients: np.ndarray
    candidate_indices: np.ndarray
    counts: dict


def validate_postprocessing_contract(metadata):
    expected_outputs = [{"name": "output0", "shape": [1, 39, 8400], "onnx_dtype": 1},
                        {"name": "output1", "shape": [1, 32, 160, 160], "onnx_dtype": 1}]
    expected_options = {"display_conf": .25, "metric_conf": .001, "nms_iou": .7, "max_det": 300,
                        "classes": None, "display_multi_label": False, "metric_multi_label": True,
                        "agnostic_nms": False, "retina_masks": True}
    contract = metadata.get("postprocessing_contract", {})
    if (metadata.get("graph", {}).get("outputs") != expected_outputs or
            any(contract.get(key) != value for key, value in expected_options.items()) or
            metadata.get("class_names") != {"0": "sam_whole_milk", "1": "yili_shuhua", "2": "luckin_cup"}):
        raise ValueError("Expected the B1 three-class FP32 raw-output / class-aware native-mask contract")


def _float32_array(value, shape_tail, name):
    if (not isinstance(value, np.ndarray) or value.dtype != np.float32 or
            value.ndim != len(shape_tail) or
            any(expected is not None and actual != expected for actual, expected in zip(value.shape, shape_tail)) or
            not np.isfinite(value).all()):
        raise ValueError(name + " has an invalid shape, FP32 dtype or non-finite values")


def _check_geometry(geometry):
    if not isinstance(geometry, LetterboxGeometry):
        raise ValueError("Use the LetterboxGeometry returned by shared preprocessing")
    h, w = geometry.original_shape_hw
    if min(h, w) <= 0 or geometry.input_shape_hw != (640, 640):
        raise ValueError("Expected nonempty original shape and static 640x640 input")
    r = min(640 / h, 640 / w)
    content = (round(h * r), round(w * r))
    half_h, half_w = (640 - content[0]) / 2, (640 - content[1]) / 2
    pads = (round(half_w - .1), round(half_h - .1), round(half_w + .1), round(half_h + .1))
    if geometry.ratio_xy != (r, r) or geometry.resized_shape_hw != content or geometry.padding_ltrb != pads:
        raise ValueError("Geometry differs from the B1 centered letterbox contract")


def _nms(boxes, scores, class_ids, iou, max_det):
    # Match the locked NMS class-offset arithmetic in float32, including its 7680 offset.
    shifted = boxes + class_ids[:, None].astype(np.float32) * np.float32(7680)
    areas = (shifted[:, 2] - shifted[:, 0]) * (shifted[:, 3] - shifted[:, 1])
    # Equal scores use ascending input order. Official tied-score ordering is backend-dependent.
    order = np.argsort(-scores, kind="stable")
    kept = []
    while order.size and len(kept) < max_det:
        chosen, rest = order[0], order[1:]
        kept.append(chosen)
        if not rest.size:
            break
        lower = np.maximum(shifted[chosen, :2], shifted[rest, :2])
        upper = np.minimum(shifted[chosen, 2:], shifted[rest, 2:])
        size = np.maximum(upper - lower, np.float32(0))
        intersection = size[:, 0] * size[:, 1]
        if not intersection.any():
            order = rest
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            overlap = intersection / (areas[chosen] + areas[rest] - intersection)
        order = rest[overlap <= iou]  # strictly greater IoU is suppressed
    return np.asarray(kept, dtype=np.int64)


def select_candidates(output0, conf=.25, iou=.7, max_det=300, multi_label=False):
    """Score filtering and class-aware NMS. N may vary in unit fixtures; B1 graph fixes it to 8400.

    Scores are already sigmoid outputs, with no extra objectness channel. Every coefficient row
    retains its candidate index, including multi-label rows from a single candidate.
    """
    _float32_array(output0, (1, 39, None), "output0")
    if (not np.isfinite(conf) or not 0 <= conf <= 1 or not np.isfinite(iou) or not 0 <= iou <= 1 or
            not isinstance(max_det, int) or isinstance(max_det, bool) or max_det <= 0 or
            not isinstance(multi_label, bool)):
        raise ValueError("Invalid confidence / IoU / max_det / multi_label options")
    rows = output0[0].T
    if np.any(rows[:, 2:4] < 0) or np.any(rows[:, 4:7] < 0) or np.any(rows[:, 4:7] > 1):
        raise ValueError("Expected nonnegative decoded box size and sigmoid class scores in [0,1]")
    source_ids = np.flatnonzero(rows[:, 4:7].max(1) > conf) if rows.shape[0] else np.empty(0, np.int64)
    rows = rows[source_ids]
    if multi_label:
        row_ids, class_ids = np.nonzero(rows[:, 4:7] > conf)
    else:
        row_ids = np.arange(len(rows), dtype=np.int64)
        class_ids = rows[:, 4:7].argmax(1)
    scores = rows[row_ids, 4 + class_ids]
    rows, source_ids = rows[row_ids], source_ids[row_ids]
    centers, half = rows[:, :2], rows[:, 2:4] / np.float32(2)
    boxes = np.concatenate((centers - half, centers + half), axis=1)
    count_rows = len(rows)
    if count_rows > 30000:  # same excess-candidate limit as the locked reference
        order = np.argsort(-scores, kind="stable")[:30000]
        boxes, scores, class_ids, rows, source_ids = [a[order] for a in (boxes, scores, class_ids, rows, source_ids)]
    keep = _nms(boxes, scores, class_ids, iou, max_det)
    counts = {"candidate_locations": output0.shape[-1], "score_passed_locations": len(
        np.flatnonzero(output0[0, 4:7].max(0) > conf)) if output0.shape[-1] else 0,
        "label_rows_before_nms": count_rows, "after_nms": len(keep)}
    return SelectedCandidates(boxes[keep], scores[keep], class_ids[keep].astype(np.int64),
                              rows[keep, 7:39].copy(), source_ids[keep].astype(np.int64), counts)


def restore_boxes(boxes_input_xyxy, geometry):
    _check_geometry(geometry)
    _float32_array(boxes_input_xyxy, (None, 4), "boxes")
    boxes = boxes_input_xyxy.copy()
    left, top, _, _ = geometry.padding_ltrb
    boxes[:, [0, 2]] -= left
    boxes[:, [1, 3]] -= top
    boxes[:, [0, 2]] /= geometry.ratio_xy[0]
    boxes[:, [1, 3]] /= geometry.ratio_xy[1]
    h, w = geometry.original_shape_hw
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)
    return boxes


def scale_mask_logits(logits, original_shape_hw):
    """Match scale_masks' prototype-resolution crop and align_corners=False bilinear semantics.

    OpenCV and PyTorch may differ by floating-point rounding; acceptance measures logits and
    binary-mask IoU separately. Prototype padding is recomputed at its own resolution, rather
    than dividing the input's integer padding by four.
    """
    _float32_array(logits, (None, None, None), "mask logits")
    h, w = original_shape_hw
    mh, mw = logits.shape[1:]
    if min(h, w, mh, mw) <= 0:
        raise ValueError("Expected positive mask and original image dimensions")
    if (mh, mw) == (h, w):
        return logits.copy()
    gain = min(mh / h, mw / w)
    content_h, content_w = round(h * gain), round(w * gain)
    top, left = round((mh - content_h) / 2 - .1), round((mw - content_w) / 2 - .1)
    if min(content_h, content_w) <= 0:
        raise ValueError("Original aspect ratio produces a zero-sized prototype content crop")
    result = np.empty((len(logits), h, w), np.float32)
    for index, mask in enumerate(logits):
        result[index] = cv2.resize(mask[top:top + content_h, left:left + content_w], (w, h),
                                   interpolation=cv2.INTER_LINEAR)
    return result


def native_masks(output1, coefficients, boxes_original_xyxy, original_shape_hw):
    _float32_array(output1, (1, 32, 160, 160), "output1")
    _float32_array(coefficients, (None, 32), "coefficients")
    _float32_array(boxes_original_xyxy, (len(coefficients), 4), "boxes")
    h, w = original_shape_hw
    if min(h, w) <= 0:
        raise ValueError("Expected positive original image dimensions")
    masks = np.empty((len(coefficients), h, w), np.uint8)
    # Keep float intermediates bounded. Output masks still occupy N*H*W bytes.
    step = max(1, 32_000_000 // (h * w))
    columns, rows = np.arange(w, dtype=np.float32)[None, :], np.arange(h, dtype=np.float32)[:, None]
    for start in range(0, len(coefficients), step):
        end = start + step
        logits = (coefficients[start:end] @ output1[0].reshape(32, -1)).reshape(-1, 160, 160)
        scaled = scale_mask_logits(logits, original_shape_hw)
        for offset, mask in enumerate(scaled):
            index = start + offset
            x1, y1, x2, y2 = boxes_original_xyxy[index]
            binary = (mask > 0).astype(np.uint8)  # no sigmoid on prototypes or extra sigmoid on scores
            binary *= (columns >= x1) & (columns < x2)
            binary *= (rows >= y1) & (rows < y2)
            masks[index] = binary
    return masks


def postprocess_b1(output0, output1, geometry, conf=.25, iou=.7, max_det=300, multi_label=False):
    """Convert both raw outputs into original-image instances, including well-formed empty results."""
    _check_geometry(geometry)
    selected = select_candidates(output0, conf, iou, max_det, multi_label)
    boxes = restore_boxes(selected.boxes_input_xyxy, geometry)
    masks = native_masks(output1, selected.mask_coefficients, boxes, geometry.original_shape_hw)
    keep = masks.any(axis=(1, 2))
    counts = dict(selected.counts, empty_masks_removed=int((~keep).sum()), instances=int(keep.sum()))
    return ProductInstances(boxes[keep], selected.scores[keep], selected.class_ids[keep], masks[keep],
                            selected.mask_coefficients[keep], selected.candidate_indices[keep], counts)
