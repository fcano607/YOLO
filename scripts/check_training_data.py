"""Read every frozen debug sample through the actual locked segmentation training loader."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch

from app.product_data import digest, image_write, now, read_json, write_json
from app.product_dataset import COLORS_BGR, locked_source
from app.product_training import check_guards, debug_configuration, original_guards, relative, verify_debug_split


def check_batch(batch, samples):
    images, boxes, classes = batch["img"], batch["bboxes"], batch["cls"]
    masks, indexes = batch["masks"], batch["batch_idx"]
    if images.dtype != torch.uint8 or images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("Training loader must return uint8 BCHW images before trainer normalization")
    if len(boxes) != len(classes) or len(classes) != len(indexes) or masks.shape[0] != len(images):
        raise ValueError("Overlap-mask, class, box or batch index counts differ")
    h, w = images.shape[-2:]
    if tuple(masks.shape[-2:]) != (h // 4, w // 4):
        raise ValueError("Unexpected loss mask resolution")
    if not torch.isfinite(boxes).all() or (boxes < 0).any() or (boxes > 1).any():
        raise ValueError("Invalid normalized loss boxes")
    if len(boxes) and (boxes[:, 2:] <= 0).any():
        raise ValueError("Empty loss box")
    records = []
    for i, file in enumerate(batch["im_file"]):
        image_id = Path(file).stem
        original = samples[image_id]
        selected = indexes == i
        cls = classes[selected].flatten().tolist()
        source_cls = [p["class_id"] for p in original["instances"]]
        if any(c not in (0, 1, 2) or cls.count(c) > source_cls.count(c) for c in cls):
            raise ValueError("Loss input invented instances or class IDs")
        values = torch.unique(masks[i]).tolist()
        if sorted(values) != list(range(len(cls) + 1)):
            raise ValueError("Overlap mask IDs do not correspond to this image's ordered targets")
        areas = []
        for j, box in enumerate(boxes[selected]):
            foreground = masks[i] == j + 1
            ys, xs = torch.where(foreground)
            areas.append(int(foreground.sum()))
            # Rasterization and downsampling can shift an edge by about one mask pixel.
            center, size = box[:2].numpy(), box[2:].numpy()
            low, high = (center - size / 2) * [w / 4, h / 4], (center + size / 2) * [w / 4, h / 4]
            if (xs.min() < low[0] - 2 or xs.max() > high[0] + 2 or
                    ys.min() < low[1] - 2 or ys.max() > high[1] + 2):
                raise ValueError("Loss mask extends outside its matching box")
        records.append({"image_id": image_id, "source_class_ids": source_cls, "loss_class_ids": cls,
                        "filtered_instances": len(source_cls) - len(cls), "mask_areas": areas,
                        "shape_chw": list(images[i].shape), "loss_mask_hw": list(masks[i].shape),
                        "negative_still_empty": not cls and not bool(masks[i].any()) if not source_cls else None})
    return records


def tile(batch, index, split):
    img = batch["img"][index].permute(1, 2, 0).numpy()[..., ::-1].copy()
    h, w = img.shape[:2]
    ids = batch["cls"][batch["batch_idx"] == index].flatten().int().tolist()
    label = cv2.resize(batch["masks"][index].numpy().astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    for j, cls in enumerate(ids):
        pixels = label == j + 1
        img[pixels] = (0.72 * img[pixels] + 0.28 * np.array(COLORS_BGR[cls])).astype(np.uint8)
    width = 240
    canvas = np.full((266, width, 3), 245, np.uint8)
    ratio = width / max(h, w)
    resized = cv2.resize(img, (round(w * ratio), round(h * ratio)))
    top, left = 26 + (240 - resized.shape[0]) // 2, (width - resized.shape[1]) // 2
    canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    image_id = Path(batch["im_file"][index]).stem.rsplit("_", 1)[-1]
    cv2.putText(canvas, f"{split} {image_id} | IDs {ids}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1)
    return canvas


def quality_summary(manifest):
    summary = {}
    for cls in range(3):
        objects = [p for s in manifest["samples"] for p in s["instances"] if p["class_id"] == cls]
        areas = [p["normalized_area"] for p in objects]
        boxes = [p["bbox_xyxy_normalized"] for p in objects]
        summary[str(cls)] = {"instances": len(objects), "vertices_min_max": [min(p["vertices"] for p in objects), max(p["vertices"] for p in objects)],
                             "normalized_polygon_area_min_median_max": [min(areas), float(np.median(areas)), max(areas)],
                             "normalized_bbox_width_min_max": [min(b[2] - b[0] for b in boxes), max(b[2] - b[0] for b in boxes)],
                             "normalized_bbox_height_min_max": [min(b[3] - b[1] for b in boxes), max(b[3] - b[1] for b in boxes)]}
    raw_manifest = read_json(ROOT / read_json(ROOT / manifest["inventory_path"])["source_manifest"])
    poses = {}
    for sample in raw_manifest["samples"]:
        pose = sample.get("capture_hint", {}).get("pose", "unknown")
        poses[pose] = poses.get(pose, 0) + 1
    return {"class_geometry": summary, "capture_prompt_pose_counts": poses,
            "capture_prompt_limit": "Prompts describe the collection plan, not verified per-instance occlusion/pose labels. Human-reviewed class IDs override prompt IDs.",
            "original_image_shapes_hw": sorted({tuple(s["shape_hw"]) for s in manifest["samples"]}),
            "independent_test_images": 0, "formal_dataset_quality_pending": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_debug.yaml")
    args = parser.parse_args()
    source = locked_source()
    manifest = verify_debug_split()
    config_path = ROOT / args.config
    config, overrides = debug_configuration(config_path)
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils.torch_utils import init_seeds
    init_seeds(overrides["seed"], deterministic=True)
    data = check_det_dataset(overrides["data"], autodownload=False)
    samples = {s["image_id"]: s for s in manifest["samples"]}
    results, tiles, loaders = {}, [], {}
    for split in ("train", "val"):
        hyp = get_cfg(overrides=overrides)
        batch_size = overrides["batch"] if split == "train" else overrides["batch"] * 2
        dataset = build_yolo_dataset(hyp, data[split], batch_size, data, mode=split, rect=split == "val", stride=32)
        loader = build_dataloader(dataset, batch_size, workers=0, shuffle=split == "train", device=torch.device("cuda:0"))
        loaders[split] = dataset
        records = []
        for batch in loader:
            records.extend(check_batch(batch, samples))
            for i in range(len(batch["img"])):
                tiles.append(tile(batch, i, split))
        expected = sorted(s["image_id"] for s in samples.values() if s["split"] == split)
        if sorted(r["image_id"] for r in records) != expected:
            raise ValueError("Actual loader skipped or duplicated frozen samples")
        if split == "val" and any(r["filtered_instances"] for r in records):
            raise ValueError("Unaugmented validation lost instances")
        results[split] = {"images_loaded": len(records), "batches": len(loader), "workers": loader.num_workers,
                          "augment": dataset.augment, "rectangular": dataset.rect, "records": records,
                          "loss_instances": sum(len(r["loss_class_ids"]) for r in records)}
    dataset = loaders["train"]
    negative_indexes = [i for i, label in enumerate(dataset.labels) if not len(label["cls"])]
    negative_batch = dataset.collate_fn([dataset[i] for i in negative_indexes])
    negative_check = check_batch(negative_batch, samples)
    if not all(r["negative_still_empty"] for r in negative_check):
        raise ValueError("All-negative batch produced a target")
    contact = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    contact_path = ROOT / "data/desktop/images/debug-v1_loader_contact.jpg"
    image_write(contact_path, contact)
    guards = original_guards(manifest["samples"])
    check_guards(guards)
    verify_debug_split()
    report = {"schema_version": 1, "checked_at": now(), "purpose": "pipeline_debug_only", "source": source,
              "split_manifest": config["split_manifest"], "split_sha256": digest(ROOT / config["split_manifest"]),
              "train_config": relative(config_path), "train_config_sha256": digest(config_path),
              "implementation_sha256": {relative(Path(__file__)): digest(Path(__file__)),
                                         "app/product_training.py": digest(ROOT / "app/product_training.py")},
              "augment_policy_sha256": digest(ROOT / config["augment_policy"]),
              "configuration": overrides, "actual_loader": results,
              "quality_distribution": quality_summary(manifest),
              "all_negative_batch": negative_check, "originals_and_reviews_unchanged": True,
              "preview": relative(contact_path), "preview_sha256": digest(contact_path),
              "formal_independent_evaluation": False,
              "limitation": "Actual data loading passed; M2 formal independence requires new held-out capture groups."}
    write_json(ROOT / "reports/data/M2_debug_precheck.json", report)
    print({s: {k: v for k, v in r.items() if k != "records"} for s, r in results.items()})
    print("all-negative batch:", len(negative_check), "passed; originals unchanged")


if __name__ == "__main__":
    main()
