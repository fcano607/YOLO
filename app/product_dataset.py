"""Inventory and diagnostic previews; never creates a training split or new truth."""

import math
import os
from pathlib import Path
import random
import subprocess
from types import SimpleNamespace

import cv2
import numpy as np
import yaml

from app.product_data import ROOT, catalog, digest, identifier, image_read, read_json, validate_polygon

os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "logs/ultralytics_settings"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("YOLO_OFFLINE", "true")


def locked_source():
    import ultralytics
    lock = read_json(ROOT / "configs/source-lock.json")
    source = ROOT / lock["path"]
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip()
    if (commit != lock["commit"] or dirty or ultralytics.__version__ != lock["version"] or
            Path(ultralytics.__file__).resolve() != source / "ultralytics/__init__.py"):
        raise ValueError("Dataset previews must use the clean locked Ultralytics source")
    return {"version": lock["version"], "commit": commit, "module": str(ultralytics.__file__)}


def load_policy(path):
    policy = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if policy["schema_version"] != 1 or policy["mapping_version"] != "products-v1":
        raise ValueError("Expected a products-v1 augmentation policy")
    args = policy["train_args"]
    # This harness has no assigned training pool and intentionally cannot mix samples.
    for key in ("mosaic", "mixup", "cutmix", "copy_paste", "fliplr", "flipud", "perspective", "shear"):
        if args.get(key) != 0:
            raise ValueError("Diagnostic policy must keep " + key + " disabled")
    if args.get("augmentations") != []:
        raise ValueError("Optional Albumentations must be explicitly disabled for this preview")
    from ultralytics.cfg import get_cfg
    get_cfg(overrides=args)  # Validate supported settings with the locked implementation.
    if not isinstance(policy["imgsz"], int) or not 32 <= policy["imgsz"] <= 2048:
        raise ValueError("Invalid preview image size")
    return policy


def perceptual_hash(image):
    gray = cv2.cvtColor(cv2.resize(image, (32, 32)), cv2.COLOR_BGR2GRAY)
    low = cv2.dct(gray.astype(np.float32))[:8, :8].flatten()
    bits = low[1:] > np.median(low[1:])
    return "".join("1" if bit else "0" for bit in bits)


def inspect_sample(raw, sample):
    raw = Path(raw)
    image_id = identifier(sample["image_id"])
    relative = Path(sample["image_path"])
    image_path = (raw / relative).resolve()
    if relative.is_absolute() or not image_path.is_relative_to(raw.resolve()):
        raise ValueError("Original image is outside the raw dataset")
    if digest(image_path) != sample["image_sha256"]:
        raise ValueError("Original image changed: " + image_id)
    image = image_read(image_path)
    if list(image.shape[:2]) != [sample["height"], sample["width"]]:
        raise ValueError("Image dimensions changed: " + image_id)
    review_path = raw / "annotations/reviewed" / (image_id + ".json")
    review = read_json(review_path)
    if (review.get("confirmed") is not True or review["image_id"] != image_id or
            review["group_id"] != sample["group_id"] or
            review["mapping_version"] != "products-v1" or
            review["image_sha256"] != sample["image_sha256"]):
        raise ValueError("Review identity or human confirmation failed: " + image_id)
    draft_path = raw / "annotations/drafts" / (image_id + ".json")
    if digest(draft_path) != review["draft_sha256"]:
        raise ValueError("Draft provenance changed: " + image_id)
    label_path = review_path.with_suffix(".txt")
    status = review["status"]
    polygons = review["polygons"]
    if status == "excluded":
        if label_path.exists() or polygons or not review.get("reason"):
            raise ValueError("Excluded image exports labels")
    elif status in {"reviewed", "negative"}:
        if (status == "negative" and polygons) or (status == "reviewed" and not polygons):
            raise ValueError("Review status contradicts its polygons")
        lines = label_path.read_text(encoding="utf-8").splitlines()
        if len(lines) != len(polygons):
            raise ValueError("TXT instance count differs from human review")
        for line, polygon in zip(lines, polygons):
            points = validate_polygon(polygon["points"])
            tokens = line.split()
            if (isinstance(polygon["class_id"], bool) or not isinstance(polygon["class_id"], int) or
                    polygon["class_id"] not in (0, 1, 2) or
                    int(tokens[0]) != polygon["class_id"] or len(tokens) != 1 + 2 * len(points) or
                    not np.allclose(np.array(tokens[1:], dtype=float).reshape(-1, 2), points, rtol=0, atol=1e-7)):
                raise ValueError("TXT class/geometry differs from human review")
    else:
        raise ValueError("Unreviewed sample cannot enter the inventory")
    summaries = []
    for polygon in polygons:
        points = np.array(polygon["points"])
        area = abs(np.sum(points[:, 0] * np.roll(points[:, 1], -1) -
                          points[:, 1] * np.roll(points[:, 0], -1))) / 2
        summaries.append({"class_id": polygon["class_id"], "vertices": len(points),
                          "normalized_area": float(area),
                          "bbox_xyxy_normalized": [*points.min(0).tolist(), *points.max(0).tolist()]})
    return {
        "image_id": image_id, "group_id": sample["group_id"], "status": status,
        "image_path": "data/raw/camera/" + relative.as_posix(),
        "review_path": "data/raw/camera/annotations/reviewed/" + image_id + ".json",
        "label_path": None if status == "excluded" else
                      "data/raw/camera/annotations/reviewed/" + image_id + ".txt",
        "image_sha256": digest(image_path), "review_sha256": digest(review_path),
        "label_sha256": None if status == "excluded" else digest(label_path),
        "draft_sha256": digest(draft_path), "shape_hw": list(image.shape[:2]),
        "physical_ids": review.get("physical_ids", [
            catalog()["classes"][cls]["physical_id"] for cls in sorted({p["class_id"] for p in polygons})]),
        "physical_id_basis": review.get("physical_id_basis", "Derived from reviewed classes and catalog; one known item per class."),
        "instances": summaries,
        "perceptual_hash_63": perceptual_hash(image), "split": "unassigned",
    }


def inventory(raw, manifest):
    ids = [sample["image_id"] for sample in manifest["samples"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate image IDs in the review bundle")
    samples = [inspect_sample(raw, sample) for sample in manifest["samples"]]
    hashes = [sample["image_sha256"] for sample in samples]
    if len(hashes) != len(set(hashes)):
        raise ValueError("Exact duplicate originals in the dataset")
    groups = {}
    class_counts = [0, 0, 0]
    for sample in samples:
        group = groups.setdefault(sample["group_id"], {"images": 0, "target_images": 0,
                                                       "negative_images": 0, "class_instances": [0, 0, 0]})
        group["images"] += 1
        group["target_images"] += sample["status"] == "reviewed"
        group["negative_images"] += sample["status"] == "negative"
        for obj in sample["instances"]:
            cls = obj["class_id"]
            class_counts[cls] += 1
            group["class_instances"][cls] += 1
    near = []
    for i, a in enumerate(samples):
        for b in samples[i + 1:]:
            distance = sum(x != y for x, y in zip(a["perceptual_hash_63"], b["perceptual_hash_63"]))
            if distance <= 6:
                near.append({"images": [a["image_id"], b["image_id"]], "hash_distance": distance,
                             "cross_capture_group": a["group_id"] != b["group_id"],
                             "meaning": "Similarity candidate only; not automatically a duplicate or leakage."})
    return {"schema_version": 1, "mapping_version": "products-v1", "review_group_id": manifest["group_id"],
            "split_status": "unassigned_diagnostic_inventory_only", "samples": samples, "capture_groups": groups,
            "class_instance_counts": class_counts,
            "target_images": sum(s["status"] == "reviewed" for s in samples),
            "negative_images": sum(s["status"] == "negative" for s in samples),
            "excluded_images": sum(s["status"] == "excluded" for s in samples),
            "exact_duplicate_pairs": 0, "similarity_candidates": near,
            "independence_note": "Existing capture batches are related; neither is an independent final test."}


def sampling_budget(raw, baseline_ids, existing=None):
    baseline = set(existing["baseline_image_ids"] if existing else baseline_ids)
    if existing and set(baseline_ids) != baseline:
        raise ValueError("Do not reset the baseline when updating the cumulative sampling budget")
    known = set()
    for path in (Path(raw) / "sessions").glob("*.json"):
        known.update(sample["image_id"] for sample in read_json(path).get("samples", []))
    if not baseline <= known:
        raise ValueError("Budget baseline is missing from the capture manifests")
    additional = sorted(known - baseline)
    if len(additional) > 30:
        raise ValueError("Additional capture budget exceeded; do not request more automatic collection")
    return {"schema_version": 1, "mapping_version": "products-v1",
            "baseline_image_ids": sorted(baseline), "baseline_real_images": len(baseline),
            "approved_additional_real_images_max": 30, "additional_image_ids": additional,
            "additional_real_images_used": len(additional), "remaining_additional_real_images": 30 - len(additional),
            "real_images_max": len(baseline) + 30, "requested_capture_this_run": 0,
            "scope": "All newly captured real training, validation and test images and their human review.",
            "next_suggested_capture_count": [6, 10],
            "capture_deferred_to": "2026-10-04 Asia/Shanghai, only when the user is available",
            "note": "Bundles refer to originals; do not count bundle references or augmentation previews as new photos."}


def make_pipeline(policy):
    from ultralytics.cfg import get_cfg
    from ultralytics.data.augment import v8_transforms
    hyp = get_cfg(overrides=policy["train_args"])
    stub = SimpleNamespace(cache=False, data={}, use_keypoints=False, use_obb=False)
    return v8_transforms(stub, policy["imgsz"], hyp)


def make_labels(image, polygons, imgsz):
    from ultralytics.utils.instance import Instances
    from ultralytics.utils.ops import resample_segments
    h, w = image.shape[:2]
    ratio = imgsz / max(h, w)
    if ratio != 1:
        image = cv2.resize(image, (min(math.ceil(w * ratio), imgsz), min(math.ceil(h * ratio), imgsz)),
                           interpolation=cv2.INTER_LINEAR)
    segment_list = [np.array(p["points"], dtype=np.float32) for p in polygons]
    count = max(1000, max((len(s) + 1 for s in segment_list), default=1000))
    segments = np.stack(resample_segments(segment_list, n=count)) if segment_list else np.zeros((0, count, 2), np.float32)
    boxes = np.array([[*s.min(0), *s.max(0)] for s in segment_list], dtype=np.float32).reshape(-1, 4)
    return {"img": image.copy(), "cls": np.array([p["class_id"] for p in polygons], np.float32).reshape(-1, 1),
            "instances": Instances(boxes, segments, bbox_format="xyxy", normalized=True)}


def apply_preview(pipeline, image, polygons, imgsz, seed):
    # Only the diagnostic process is seeded. Actual training will manage its own RNG.
    random.seed(seed)
    np.random.seed(seed)
    labels = pipeline(make_labels(image, polygons, imgsz))
    # RandomFlip converts boxes to xywh even when its probability is zero.
    # Expose one explicit box format to diagnostic consumers.
    labels["instances"].convert_bbox(format="xyxy")
    return labels


def inspect_transformed(labels, original_class_ids):
    from ultralytics.data.utils import polygons2masks
    image = labels["img"]
    cls = labels["cls"].flatten().astype(int).tolist()
    instances = labels["instances"]
    if len(cls) != len(instances.segments) or len(cls) != len(instances.bboxes):
        raise ValueError("Augmented class/segment/box counts differ")
    for class_id in cls:
        if class_id not in (0, 1, 2) or cls.count(class_id) > original_class_ids.count(class_id):
            raise ValueError("Augmentation invented a class or instance")
    segments = instances.segments
    if not np.isfinite(segments).all() or not np.isfinite(instances.bboxes).all():
        raise ValueError("Non-finite augmented geometry")
    h, w = image.shape[:2]
    if len(segments) and (segments.min() < -1e-5 or segments[..., 0].max() > w + 1e-5 or
                          segments[..., 1].max() > h + 1e-5):
        raise ValueError("Augmented polygon outside image")
    masks = polygons2masks((h, w), list(segments), 1, 4) if len(segments) else []
    areas = [int(mask.sum()) for mask in masks]
    if any(area == 0 for area in areas):
        raise ValueError("Empty augmented mask")
    return {"output_shape_hw": [h, w], "retained_class_ids": cls,
            "input_instances": len(original_class_ids), "retained_instances": len(cls),
            "filtered_instances": len(original_class_ids) - len(cls),
            "mask_nonzero_pixels_at_stride4": areas, "negative_still_empty": not cls if not original_class_ids else None}


COLORS_BGR = [(85, 155, 40), (220, 110, 40), (35, 130, 225)]


def overlay(image, polygons, class_ids):
    canvas = image.copy()
    layer = canvas.copy()
    h, w = canvas.shape[:2]
    for points, cls in zip(polygons, class_ids):
        points = np.rint(points).astype(np.int32)
        cv2.fillPoly(layer, [points], COLORS_BGR[cls])
    canvas = cv2.addWeighted(layer, 0.22, canvas, 0.78, 0)
    for points, cls in zip(polygons, class_ids):
        points = np.rint(points).astype(np.int32)
        cv2.polylines(canvas, [points], True, COLORS_BGR[cls], max(1, w // 320))
        if len(points):
            x, y = points[0]
            cv2.putText(canvas, str(cls), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        COLORS_BGR[cls], 2, cv2.LINE_AA)
    return canvas
