"""Validation-only deployment evaluation; inference imports no Torch, metric core is lazy."""
from pathlib import Path
import cv2
import numpy as np
import yaml

from deploy.base_backend import file_sha256, load_bundle, read_json
from deploy.paths import PROJECT_ROOT as ROOT
from deploy.postprocess import ProductInstances

CONFIG = ROOT / "configs/evaluate_deployment_B1.yaml"
POLICY = {"schema_version": 1, "purpose": "b1_validation_only_deployment_quality",
          "baseline": "configs/baseline_products_v1.json", "inference_config": "configs/infer_products.yaml",
          "loading_contract": "data/desktop/metadata/products-v1_loading.json",
          "previous_receipt": "reports/deployment/M5-05_B1_backend_parity.json", "split": "val",
          "input": {"shape_hw": [640, 640], "batch": 1, "fp32": True, "rect": False},
          "metric_postprocess": {"conf": .001, "iou": .7, "max_det": 300, "multi_label": True},
          "display_postprocess": {"conf": .25, "iou": .7, "max_det": 300, "multi_label": False},
          "ground_truth": {"source": "frozen_yolo_polygon_labels", "coordinate_space": "original_image",
                           "mask_downsample": 1, "overlap": False},
          "metric_core": "pinned_SegmentationValidator_process_batch_and_SegmentMetrics",
          "iou_thresholds": [.5, .55, .6, .65, .7, .75, .8, .85, .9, .95], "max_map50_95_drop": .005}


def validation_samples(loading, baseline):
    selected = [s for s in loading["samples"] if s["split"] == "val"]
    ids = [s["image_id"] for s in selected]
    if (len(selected) != 5 or len(set(ids)) != 5 or ids != baseline["data"]["image_ids"]["val"] or
            sum(len(s["instances"]) for s in selected) != 8):
        raise ValueError("Evaluation requires exactly the frozen val5 / eight instances; test remains sealed")
    return selected


def configuration(path=CONFIG):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if config != POLICY:
        raise ValueError("Deployment quality requires the fixed validation-only policy")
    baseline = read_json(ROOT / config["baseline"])
    loading = read_json(ROOT / config["loading_contract"])
    if file_sha256(ROOT / config["loading_contract"]) != baseline["references"]["loading"]["sha256"]:
        raise ValueError("Frozen loading contract changed")
    if load_bundle().baseline_id != baseline["baseline_id"]:
        raise ValueError("Deployment baseline mismatch")
    return config, validation_samples(loading, baseline)


def parse_labels(text, shape_hw):
    """Rasterize each original normalized YOLO polygon, matching polygon2mask at ratio 1.

    Int32 conversion truncates coordinates as in the pinned official rasterizer. Masks are
    separate per instance, full original resolution; this is not historical quarter-resolution GT.
    """
    h, w = shape_hw
    classes, boxes, masks = [], [], []
    for line in text.splitlines():
        if not line.strip():
            continue
        values = np.asarray([float(v) for v in line.split()], np.float32)
        if (len(values) < 7 or (len(values) - 1) % 2 or not np.isfinite(values).all() or
                values[0] not in (0, 1, 2) or (values[1:] < 0).any() or (values[1:] > 1).any()):
            raise ValueError("Invalid frozen three-class polygon label")
        points = values[1:].reshape(-1, 2) * np.asarray([w, h], np.float32)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [points.astype(np.int32)], 1)
        if not mask.any():
            raise ValueError("Empty ground-truth polygon mask")
        boxes.append(np.concatenate((points.min(axis=0), points.max(axis=0))))
        classes.append(int(values[0]))
        masks.append(mask)
    n = len(classes)
    return ProductInstances(np.asarray(boxes, np.float32).reshape(n, 4), np.ones(n, np.float32),
        np.asarray(classes, np.int64), np.asarray(masks, np.uint8).reshape(n, h, w),
        np.zeros((n, 32), np.float32), np.arange(n, dtype=np.int64), {"instances": n})


def ground_truth(sample, shape_hw):
    if sample["split"] != "val":
        raise ValueError("Ground-truth evaluation is validation-only")
    path = ROOT / sample["label_path"]
    if file_sha256(path) != sample["label_sha256"] or list(shape_hw) != sample["shape_hw"]:
        raise ValueError("Frozen label or original image geometry changed")
    truth = parse_labels(path.read_text(encoding="utf-8"), shape_hw)
    if sorted(truth.class_ids.tolist()) != sorted(s["class_id"] for s in sample["instances"]):
        raise ValueError("Frozen label class counts differ from the loading contract")
    return truth


def pack_instances(instances, prefix):
    n, h, w = instances.masks.shape
    return {prefix + "boxes": instances.boxes_xyxy, prefix + "scores": instances.scores,
            prefix + "classes": instances.class_ids, prefix + "coefficients": instances.mask_coefficients,
            prefix + "indices": instances.candidate_indices,
            prefix + "mask_shape": np.asarray([n, h, w], np.int64),
            prefix + "packed_masks": np.packbits(instances.masks.reshape(n, h*w), axis=1)}


def unpack_instances(saved, prefix):
    n, h, w = map(int, saved[prefix + "mask_shape"])
    masks = np.unpackbits(saved[prefix + "packed_masks"], axis=1, count=h*w).reshape(n, h, w)
    return ProductInstances(saved[prefix + "boxes"], saved[prefix + "scores"], saved[prefix + "classes"],
        masks, saved[prefix + "coefficients"], saved[prefix + "indices"], {"instances": n})


def metric_processor():
    import torch
    from ultralytics.models.yolo.segment import SegmentationValidator
    validator = object.__new__(SegmentationValidator)
    validator.iouv = torch.linspace(.5, .95, 10)
    validator.niou = 10
    return validator


def process_instances(validator, prediction, truth):
    import torch
    pred = {"bboxes": torch.from_numpy(prediction.boxes_xyxy),
            "cls": torch.from_numpy(prediction.class_ids), "masks": torch.from_numpy(prediction.masks)}
    target = {"bboxes": torch.from_numpy(truth.boxes_xyxy),
              "cls": torch.from_numpy(truth.class_ids), "masks": torch.from_numpy(truth.masks).float()}
    return validator._process_batch(pred, target)


def evaluate_instances(records, names):
    """Use the pinned validator IoU matching and SegmentMetrics AP/P/R for every route.

    records streams (image_id, truth, metric_predictions, display_predictions). Low-confidence
    predictions, including those on negative images, are retained in AP statistics.
    """
    from ultralytics.utils.metrics import SegmentMetrics
    validator = metric_processor()
    metrics = SegmentMetrics(names=names)
    details = []
    totals = {kind: [{"class_id": c, "targets": 0, "predictions": 0, "tp": 0, "fp": 0, "fn": 0}
                      for c in range(3)] for kind in ("box", "mask")}
    for image_id, truth, prediction, display in records:
        stats = process_instances(validator, prediction, truth)
        metrics.update_stats({**stats, "conf": prediction.scores, "pred_cls": prediction.class_ids,
                              "target_cls": truth.class_ids, "target_img": np.unique(truth.class_ids),
                              "im_name": image_id})
        fixed = process_instances(validator, display, truth)
        rows = {}
        for kind, key in (("box", "tp"), ("mask", "tp_m")):
            counts = []
            for c in range(3):
                nt = int(np.count_nonzero(truth.class_ids == c))
                npred = int(np.count_nonzero(display.class_ids == c))
                tp = int(np.count_nonzero(fixed[key][:, 0] & (display.class_ids == c)))
                count = {"class_id": c, "targets": nt, "predictions": npred, "tp": tp,
                         "fp": npred-tp, "fn": nt-tp}
                counts.append(count)
                for field in ("targets", "predictions", "tp", "fp", "fn"):
                    totals[kind][c][field] += count[field]
            rows[kind] = counts
        details.append({"image_id": image_id, "targets": len(truth.scores), "metric_predictions": len(prediction.scores),
                        "display_predictions": len(display.scores), "display_class_ids": display.class_ids.tolist(),
                        "display_scores": display.scores.tolist(), "fixed_conf_0_25_iou_0_5": rows})
    if not details:
        raise ValueError("Metric evaluation needs images, including negative images")
    metrics.process(plot=False)
    per_class = []
    for row, c in enumerate(metrics.ap_class_index):
        c = int(c)
        values = metrics.class_result(row)
        per_class.append({"class_id": c, "name": names[c], "targets": int(metrics.nt_per_class[c]),
            **dict(zip(("box_precision", "box_recall", "box_mAP50", "box_mAP50_95",
                        "mask_precision", "mask_recall", "mask_mAP50", "mask_mAP50_95"), map(float, values)))})
    for counts in totals.values():
        for row in counts:
            row["precision"] = row["tp"] / row["predictions"] if row["predictions"] else None
            row["recall"] = row["tp"] / row["targets"] if row["targets"] else None
    overall = {}
    for kind, rows in totals.items():
        sums = {k: sum(r[k] for r in rows) for k in ("targets", "predictions", "tp", "fp", "fn")}
        sums.update(precision=sums["tp"]/sums["predictions"] if sums["predictions"] else None,
                    recall=sums["tp"]/sums["targets"] if sums["targets"] else None)
        overall[kind] = sums
    return {"images": len(details), "targets": int(metrics.nt_per_class.sum()),
            "metrics": {k: float(v) for k, v in metrics.results_dict.items() if k.startswith("metrics/")},
            "per_class": per_class, "per_image": details,
            "AP_by_class_and_IoU": {"class_ids": metrics.ap_class_index.tolist(),
                                   "box": metrics.box.all_ap.tolist(), "mask": metrics.seg.all_ap.tolist()},
            "fixed_display_threshold": {"conf": .25, "iou": .5, "overall": overall, "per_class": totals},
            "precision_recall_rule": "Official smoothed maximum-mean-F1 operating point; distinct from display conf=0.25"}


def metric_deltas(reference, actual, limit=.005):
    if set(reference) != set(actual) or not all(np.isfinite(v) for v in [*reference.values(), *actual.values()]):
        raise ValueError("Metric key mismatch or non-finite values")
    signed = {k: actual[k]-reference[k] for k in reference}
    drops = {k: max(0., -signed[k]) for k in ("metrics/mAP50-95(B)", "metrics/mAP50-95(M)")}
    return {"actual_minus_reference": signed, "map50_95_drop": drops, "max_allowed_drop": limit,
            "passed": all(v <= limit for v in drops.values())}
