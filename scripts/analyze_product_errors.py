"""M3-05: inspect E1 train20/val5 errors without training, changing labels or evaluating test."""
import argparse
import csv
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch

from app.product_data import catalog, digest, identifier, image_read, image_write, now, read_json, write_json
from app.product_dataset import locked_source
from app.product_error_analysis import (aggregate_counts, analysis_samples, best_candidate,
                                        box_iou, instance_counts, match_instances)
from app.product_experiment import experiment_configuration, verify_product_model
from app.product_training import check_guards, relative, verify_formal_loading
from scripts.train import environment_fingerprint

THRESHOLDS = (.001, .05, .1, .25)
COLORS = ((0, 180, 255), (255, 180, 0), (110, 220, 70))


def targets(sample, shape):
    review = read_json(ROOT / sample["review_path"])
    h, w = shape[:2]
    masks, boxes, classes, geometry = [], [], [], []
    for polygon in review["polygons"]:
        points = np.array(polygon["points"], dtype=np.float32) * [w, h]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
        box = np.r_[points.min(0), points.max(0)]
        masks.append(mask.astype(bool))
        boxes.append(box)
        classes.append(polygon["class_id"])
        rect = cv2.boxPoints(cv2.minAreaRect(points.astype(np.float32)))
        rect_edges = np.roll(rect, -1, axis=0) - rect
        longest = rect_edges[np.argmax((rect_edges ** 2).sum(1))]
        angle = abs(float(np.degrees(np.arctan2(longest[1], longest[0])))) % 180
        geometry.append({"class_id": polygon["class_id"], "mask_area_fraction": float(mask.mean()),
                         "bbox_width": float(box[2] - box[0]), "bbox_height": float(box[3] - box[1]),
                         "long_edge_angle_to_horizontal_deg": min(angle, 180 - angle),
                         "touches_image_border": bool((points <= 1).any() or
                                                      (points[:, 0] >= w - 1).any() or
                                                      (points[:, 1] >= h - 1).any()),
                         "vertices": len(points)})
    return np.array(boxes).reshape(-1, 4), np.array(classes, dtype=int), masks, geometry


def mask_overlaps(result, truths, shape):
    """Remove predictor letterbox padding before original-pixel IoU; work in chunks."""
    from ultralytics.utils import ops
    count = len(result.boxes)
    ious = np.zeros((count, len(truths)))
    areas = np.zeros(count, dtype=int)
    display_masks = {}
    if not count:
        return ious, areas, display_masks
    if result.masks is None or len(result.masks.data) != count:
        raise ValueError("Prediction boxes and masks differ")
    scores = result.boxes.conf.cpu().numpy()
    for start in range(0, count, 16):
        scaled = ops.scale_masks(result.masks.data[start:start + 16][None].float(), shape[:2])[0]
        masks = (scaled > .5).cpu().numpy()
        for offset, mask in enumerate(masks):
            p = start + offset
            areas[p] = int(mask.sum())
            for g, truth in enumerate(truths):
                intersection = int((mask & truth).sum())
                ious[p, g] = intersection / max(areas[p] + int(truth.sum()) - intersection, 1)
            if scores[p] >= .05:
                display_masks[p] = mask
        del scaled, masks
    return ious, areas, display_masks


def raw_target_scores(model, image, truth_boxes, truth_classes):
    """Audit correct-class scores before argmax/NMS, at boxes overlapping each truth."""
    from ultralytics.utils import ops
    tensor = model.predictor.preprocess([image])
    # predict() owns its device-specific AutoBackend; the public .model can remain on CPU.
    if tensor.dtype != torch.float32:
        raise ValueError("Diagnostics require FP32 inference")
    output = model.predictor.model(tensor)
    while isinstance(output, (tuple, list)):
        output = output[0]
    if not isinstance(output, torch.Tensor) or output.ndim != 3 or output.shape[1] != 39:
        raise ValueError("Expected three-class segmentation candidate tensor [1,39,N]")
    boxes = ops.xywh2xyxy(output[0, :4].T)
    boxes = ops.scale_boxes(tensor.shape[2:], boxes, image.shape).cpu().numpy()
    scores = output[0, 4:7].T.cpu().numpy()
    ious = box_iou(boxes, truth_boxes)
    rows = []
    for g, cls in enumerate(truth_classes):
        eligible = np.flatnonzero(ious[:, g] >= .5)
        if len(eligible):
            p = eligible[np.argmax(scores[eligible, cls])]
            rows.append({"target": g, "class_id": int(cls), "overlapping_locations": len(eligible),
                         "max_correct_class_score": float(scores[p, cls]),
                         "all_class_scores_at_that_location": scores[p].tolist(),
                         "argmax_class": int(scores[p].argmax()), "box_iou": float(ious[p, g]),
                         "box_xyxy": boxes[p].tolist()})
        else:
            rows.append({"target": g, "class_id": int(cls), "overlapping_locations": 0,
                         "max_correct_class_score": None, "best_box_iou": float(ious[:, g].max())})
    return {"input_shape": list(tensor.shape), "output_shape": list(output.shape),
            "targets": rows}


def panel(image, boxes, classes, masks, names, title, scores=None, conf=None):
    canvas = image.copy()
    for p, (box, cls) in enumerate(zip(boxes, classes)):
        if scores is not None and scores[p] < conf:
            continue
        color = COLORS[cls]
        if p in masks:
            mask = masks[p]
            canvas[mask] = (.65 * canvas[mask] + .35 * np.array(color)).astype(np.uint8)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, contours, -1, color, 2)
        x1, y1, x2, y2 = np.rint(box).astype(int)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = names[cls] + (f" {scores[p]:.3f}" if scores is not None else " GT")
        cv2.putText(canvas, label, (max(x1, 0), max(y1 + 20, 20)), cv2.FONT_HERSHEY_SIMPLEX, .7, color, 2)
    canvas = cv2.resize(canvas, (640, 360))
    banner = np.full((42, 640, 3), 245, dtype=np.uint8)
    cv2.putText(banner, title, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, .65, (30, 30, 30), 1)
    return np.vstack((banner, canvas))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_baseline.yaml")
    parser.add_argument("--run", default="E1_products_v1_seed42")
    args = parser.parse_args()
    source = locked_source()
    config_path = ROOT / args.config
    config, overrides, loading = experiment_configuration(config_path)
    pools = analysis_samples(loading)
    run = identifier(args.run)
    training_path = ROOT / "reports/experiments" / (run + ".json")
    training = read_json(training_path)
    if (training["status"] != "completed" or training["purpose"] != config["purpose"] or
            training["experiment"] != config["experiment"] or training["config_sha256"] != digest(config_path)):
        raise ValueError("Expected a completed product run under its frozen configuration")
    check_guards(training["guarded_inputs"])
    fingerprint = environment_fingerprint()
    if fingerprint != training["environment_fingerprint_after"]:
        raise ValueError("Environment changed after E1")
    checkpoint = training["training"]["checkpoints"]["best"]
    if digest(ROOT / checkpoint["path"]) != checkpoint["sha256"]:
        raise ValueError("Trained checkpoint changed")
    stamp = datetime.fromisoformat(now()).strftime("%Y%m%d_%H%M%S")
    name = "M3-05_" + run + "_" + stamp
    output = ROOT / "runs/analysis" / name
    report_path = ROOT / "reports/experiments" / (name + ".json")
    if output.exists() or report_path.exists():
        raise ValueError("Analysis outputs already exist")
    output.mkdir(parents=True)
    from ultralytics import YOLO
    model = YOLO(str(ROOT / checkpoint["path"]))
    verify_product_model(model)
    names = {c["id"]: c["name"] for c in catalog()["classes"]}
    settings = {"imgsz": overrides["imgsz"], "device": 0, "quantize": None, "rect": True,
                "conf": THRESHOLDS[0], "iou": config["evaluation"]["nms_iou"],
                "max_det": config["evaluation"]["max_det"], "classes": None,
                "augment": False, "retina_masks": False, "verbose": False, "save": False}
    records, sheets = [], {}
    with torch.inference_mode():
        for split, samples in pools.items():
            tiles = []
            for sample in samples:
                image = image_read(ROOT / sample["image_path"])
                truth_boxes, truth_classes, truth_masks, geometry = targets(sample, image.shape)
                result = model.predict(image, **settings)[0]
                pred_boxes = result.boxes.xyxy.cpu().numpy()
                pred_classes = result.boxes.cls.int().cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                biou = box_iou(pred_boxes, truth_boxes)
                miou, mask_areas, display_masks = mask_overlaps(result, truth_masks, image.shape)
                raw = raw_target_scores(model, image, truth_boxes, truth_classes)
                audits = []
                for g, cls in enumerate(truth_classes):
                    same = best_candidate(biou, pred_classes, scores, g, cls)
                    any_class = best_candidate(biou, pred_classes, scores, g, cls, same_class=False)
                    for candidate in (same, any_class):
                        if candidate is not None:
                            p = candidate["prediction"]
                            candidate.update(mask_iou=float(miou[p, g]),
                                             mask_area_ratio_to_truth=float(mask_areas[p] / truth_masks[g].sum()))
                    best_same_iou = max((biou[p, g] for p in range(len(scores)) if pred_classes[p] == cls), default=0)
                    audits.append({"target": g, "class_id": int(cls), "geometry": geometry[g],
                                   "same_class_box_candidate": same, "any_class_box_candidate": any_class,
                                   "best_same_class_box_iou": float(best_same_iou), "raw": raw["targets"][g]})
                row = {"image_id": sample["image_id"], "split": split, "group_id": sample["group_id"],
                       "status": sample["status"], "image_sha256": sample["image_sha256"],
                       "target_classes": truth_classes.tolist(), "target_boxes_xyxy": truth_boxes.tolist(),
                       "predicted_classes": pred_classes.tolist(), "scores": scores.tolist(),
                       "boxes_xyxy": pred_boxes.tolist(), "mask_areas_pixels": mask_areas.tolist(),
                       "box_iou_matrix": biou.tolist(), "mask_iou_matrix": miou.tolist(),
                       "candidate_cap_reached": len(scores) >= settings["max_det"],
                       "raw_input_shape": raw["input_shape"], "raw_output_shape": raw["output_shape"],
                       "target_audits": audits, "thresholds": {}}
                for conf in THRESHOLDS:
                    row["thresholds"][str(conf)] = {
                        "box": instance_counts(biou, pred_classes, scores, truth_classes, conf),
                        "mask": instance_counts(miou, pred_classes, scores, truth_classes, conf),
                        "class_agnostic_box_matches": match_instances(biou, pred_classes, scores,
                                                                       truth_classes, conf, class_aware=False)}
                truth_panel = panel(image, truth_boxes, truth_classes, dict(enumerate(truth_masks)), names,
                                    sample["image_id"][-3:] + " / human GT")
                predicted_panel = panel(image, pred_boxes, pred_classes, display_masks, names,
                                        config["experiment"] + " conf >= 0.25", scores, .25)
                low_panel = panel(image, pred_boxes, pred_classes, display_masks, names,
                                  "Diagnostic only: conf >= 0.05", scores, .05)
                tile = np.hstack((truth_panel, predicted_panel, low_panel))
                if split == "val":
                    preview = output / (sample["image_id"] + ".jpg")
                    image_write(preview, tile)
                    row["preview"] = relative(preview)
                tiles.append(tile)
                records.append(row)
                print(split, sample["image_id"], "GT", truth_classes.tolist(),
                      "shown", pred_classes[scores >= .25].tolist(), flush=True)
            sheets[split] = []
            # Four train sheets and one validation sheet; do not copy the original images.
            for index in range(0, len(tiles), 5):
                sheet = output / (split + "_" + str(index // 5 + 1) + ".jpg")
                image_write(sheet, np.vstack(tiles[index:index + 5]))
                sheets[split].append(relative(sheet))
    summary = {split: {str(conf): {kind: aggregate_counts([r for r in records if r["split"] == split], conf, kind)
                                  for kind in ("box", "mask")} for conf in THRESHOLDS}
               for split in pools}
    geometry_summary = {split: [] for split in pools}
    for split in pools:
        for cls in range(3):
            values = [a["geometry"] for r in records if r["split"] == split
                      for a in r["target_audits"] if a["class_id"] == cls]
            metrics = {key: {"min": float(np.min([v[key] for v in values])),
                             "median": float(np.median([v[key] for v in values])),
                             "max": float(np.max([v[key] for v in values]))}
                       for key in ("mask_area_fraction", "bbox_width", "bbox_height",
                                   "long_edge_angle_to_horizontal_deg")}
            geometry_summary[split].append({"class_id": cls, "instances": len(values),
                                             "border_instances": sum(v["touches_image_border"] for v in values), **metrics})
    check_guards(training["guarded_inputs"])
    verify_formal_loading()
    if environment_fingerprint() != fingerprint or digest(ROOT / checkpoint["path"]) != checkpoint["sha256"]:
        raise ValueError("Environment or weights changed during analysis")
    locked_source()
    results_csv = ROOT / training["training"]["results_csv"]["path"]
    with results_csv.open(encoding="utf-8", newline="") as stream:
        curves = [{k.strip(): float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
    report = {"schema_version": 1, "status": "completed", "purpose": "M3-05_products_error_analysis",
              "checked_at": now(), "source": source, "experiment": config["experiment"], "training_run": run,
              "training_report_sha256": digest(training_path), "checkpoint": checkpoint,
              "implementation_sha256": {p: digest(ROOT / p) for p in
                                         ("scripts/analyze_product_errors.py", "app/product_error_analysis.py")},
              "environment_fingerprint": fingerprint, "settings": settings,
              "matching": "Confidence-ordered greedy one-to-one same-class IoU>=0.5; box and mask matched separately.",
              "mask_coordinates": "Default predictor masks; scale_masks removes padding, bilinear upsample to original H/W, threshold>0.5. GT polygon vertices rounded to original pixels.",
              "thresholds": list(THRESHOLDS), "summary": summary, "geometry": geometry_summary,
              "records": records, "contact_sheets": sheets,
              "curve_samples": [r for r in curves if int(r["epoch"]) in (1, 10, 20, 30, 40, 50)],
              "guarded_inputs_unchanged": True, "optimizer_steps": 0, "new_photos": 0,
              "test_images_inferred": 0, "checkpoint_unchanged": True,
              "limitations": ["Train predictions diagnose fit to known images, not generalization.",
                              "Five val images and one physical item per class; threshold diagnostics are not AP or broad accuracy.",
                              "Low-threshold filters reuse the same post-NMS pool (max_det=300); raw scores separately inspect pre-NMS candidates.",
                              "No other milk brand negatives, systematic illumination/occlusion groups or sealed-test predictions."]}
    write_json(report_path, report)
    print("Saved:", relative(report_path))
    for split in pools:
        for conf in (.05, .25):
            print(split, conf, "box", [(r["tp"], r["fp"], r["fn"]) for r in summary[split][str(conf)]["box"]],
                  "mask", [(r["tp"], r["fp"], r["fn"]) for r in summary[split][str(conf)]["mask"]])


if __name__ == "__main__":
    main()
