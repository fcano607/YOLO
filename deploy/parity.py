"""Match same-class instances for deployment parity; these are not quality metrics."""
import numpy as np


TOLERANCE = {"raw": {"atol": .001, "rtol": .0001}, "match_box_iou": .9,
             "max_box_abs": .01, "max_score_abs": .0001, "min_mask_iou": .999}


def box_iou_matrix(left, right):
    lower = np.maximum(left[:, None, :2], right[None, :, :2])
    upper = np.minimum(left[:, None, 2:], right[None, :, 2:])
    intersection = np.maximum(upper - lower, 0).prod(axis=2)
    area_l = np.maximum(left[:, 2:] - left[:, :2], 0).prod(axis=1)
    area_r = np.maximum(right[:, 2:] - right[:, :2], 0).prod(axis=1)
    union = area_l[:, None] + area_r[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def compare_instances(actual, reference, tolerance=TOLERANCE):
    """Greedy descending box-IoU matching within class, stable row ordering for exact ties.

    Require all instances to match and then check box, score and binary-mask differences.
    Empty/empty is a valid empty comparison and has no measured mask IoU.
    """
    if actual.masks.shape[1:] != reference.masks.shape[1:]:
        raise ValueError("Compare masks in the same original-image geometry")
    if not all(np.isfinite(v).all() for v in (actual.boxes_xyxy, reference.boxes_xyxy,
                                             actual.scores, reference.scores)):
        raise ValueError("Non-finite instance values")
    overlap = box_iou_matrix(actual.boxes_xyxy, reference.boxes_xyxy)
    compatible = ((actual.class_ids[:, None] == reference.class_ids[None, :]) &
                  (overlap >= tolerance["match_box_iou"]))
    rows, columns = np.nonzero(compatible)
    order = np.argsort(-overlap[rows, columns], kind="stable")
    used_a, used_r, pairs = set(), set(), []
    for position in order:
        a, r = int(rows[position]), int(columns[position])
        if a in used_a or r in used_r:
            continue
        used_a.add(a)
        used_r.add(r)
        left, right = actual.masks[a].astype(bool), reference.masks[r].astype(bool)
        union = int(np.count_nonzero(left | right))
        mask_iou = float(np.count_nonzero(left & right) / union) if union else 1.
        pairs.append({"actual_row": a, "reference_row": r, "class_id": int(actual.class_ids[a]),
                      "box_iou": float(overlap[a, r]),
                      "box_abs": float(np.abs(actual.boxes_xyxy[a] - reference.boxes_xyxy[r]).max()),
                      "score_abs": float(abs(actual.scores[a] - reference.scores[r])),
                      "mask_iou": mask_iou, "mask_disagreement_pixels": int(np.count_nonzero(left != right)),
                      "candidate_indices_equal": bool(actual.candidate_indices[a] == reference.candidate_indices[r])})
    unmatched_a = sorted(set(range(len(actual.scores))) - used_a)
    unmatched_r = sorted(set(range(len(reference.scores))) - used_r)
    box_error = max((p["box_abs"] for p in pairs), default=0.)
    score_error = max((p["score_abs"] for p in pairs), default=0.)
    minimum = min((p["mask_iou"] for p in pairs), default=None)
    return {"actual_instances": len(actual.scores), "reference_instances": len(reference.scores),
            "matched": len(pairs), "unmatched_actual": unmatched_a, "unmatched_reference": unmatched_r,
            "pairs": pairs, "both_empty": not len(actual.scores) and not len(reference.scores),
            "max_box_abs": box_error, "max_score_abs": score_error, "min_mask_iou": minimum,
            "mask_disagreement_pixels": sum(p["mask_disagreement_pixels"] for p in pairs),
            "candidate_indices_equal_after_matching": all(p["candidate_indices_equal"] for p in pairs),
            "passed": bool(not unmatched_a and not unmatched_r and box_error <= tolerance["max_box_abs"] and
                           score_error <= tolerance["max_score_abs"] and
                           (minimum is None or minimum >= tolerance["min_mask_iou"]))}


def score_ties(raw, conf=.25):
    scores = raw[0, 4:7].max(axis=0)
    values, counts = np.unique(scores[scores > conf], return_counts=True)
    return [{"score": float(value), "candidates": int(count)}
            for value, count in zip(values, counts) if count > 1]


def assert_binding(instances, raw):
    indices = instances.candidate_indices
    if (not np.array_equal(instances.mask_coefficients, raw[0, 7:, indices]) or
            not np.array_equal(instances.scores, raw[0, 4 + instances.class_ids, indices])):
        raise ValueError("Instance masks/scores lost their original candidate binding")
