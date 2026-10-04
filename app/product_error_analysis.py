"""Fixed-threshold diagnostics, separate from official AP and sealed final testing."""
import numpy as np


def analysis_samples(loading):
    """Select only frozen train/val; folder names never determine their role."""
    pools = {key: [s for s in loading["samples"] if s["split"] == key]
             for key in ("train", "val")}
    if {k: len(v) for k, v in pools.items()} != {"train": 20, "val": 5}:
        raise ValueError("Expected frozen train20/val5 diagnostics; test inference is forbidden")
    ids = [s["image_id"] for pool in pools.values() for s in pool]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate diagnostic image IDs")
    return pools


def box_iou(pred, truth):
    pred, truth = np.asarray(pred).reshape(-1, 4), np.asarray(truth).reshape(-1, 4)
    intersection = np.maximum(0, np.minimum(pred[:, None, 2:], truth[None, :, 2:]) -
                              np.maximum(pred[:, None, :2], truth[None, :, :2])).prod(2)
    area_p = np.maximum(0, pred[:, 2:] - pred[:, :2]).prod(1)
    area_t = np.maximum(0, truth[:, 2:] - truth[:, :2]).prod(1)
    return intersection / np.maximum(area_p[:, None] + area_t[None, :] - intersection, 1e-9)


def match_instances(ious, pred_classes, scores, truth_classes, conf, class_aware=True):
    """Descending score, highest IoU available truth, each instance used once (IoU>=.5)."""
    ious = np.asarray(ious)
    pred_classes, truth_classes = np.asarray(pred_classes), np.asarray(truth_classes)
    scores = np.asarray(scores)
    if ious.shape != (len(scores), len(truth_classes)) or len(pred_classes) != len(scores):
        raise ValueError("IoU matrix and instances differ")
    available = set(range(len(truth_classes)))
    matches = []
    for p in np.argsort(-scores, kind="stable"):
        if scores[p] < conf:
            continue
        eligible = [g for g in sorted(available) if ious[p, g] >= .5 and
                    (not class_aware or pred_classes[p] == truth_classes[g])]
        if eligible:
            g = max(eligible, key=lambda t: ious[p, t])
            available.remove(g)
            matches.append({"prediction": int(p), "target": int(g), "iou": float(ious[p, g])})
    return matches


def instance_counts(ious, pred_classes, scores, truth_classes, conf):
    matches = match_instances(ious, pred_classes, scores, truth_classes, conf)
    pred_classes, scores, truth_classes = map(np.asarray, (pred_classes, scores, truth_classes))
    rows = []
    for cls in range(3):
        tp = sum(truth_classes[m["target"]] == cls for m in matches)
        predictions = int(((pred_classes == cls) & (scores >= conf)).sum())
        targets = int((truth_classes == cls).sum())
        rows.append({"class_id": cls, "targets": targets, "predictions": predictions,
                     "tp": int(tp), "fp": predictions - int(tp), "fn": targets - int(tp)})
    return {"per_class": rows, "matches": matches}


def aggregate_counts(records, conf, kind):
    rows = []
    for cls in range(3):
        values = [r["thresholds"][str(conf)][kind]["per_class"][cls] for r in records]
        total = {k: sum(v[k] for v in values) for k in ("targets", "predictions", "tp", "fp", "fn")}
        total.update(class_id=cls,
                     precision=total["tp"] / total["predictions"] if total["predictions"] else None,
                     recall=total["tp"] / total["targets"] if total["targets"] else None)
        rows.append(total)
    return rows


def best_candidate(ious, pred_classes, scores, truth_index, target_class, same_class=True):
    eligible = [p for p in range(len(scores)) if ious[p, truth_index] >= .5 and
                (not same_class or pred_classes[p] == target_class)]
    if not eligible:
        return None
    p = max(eligible, key=lambda p: scores[p])
    return {"prediction": p, "class_id": int(pred_classes[p]), "score": float(scores[p]),
            "box_iou": float(ious[p, truth_index])}
