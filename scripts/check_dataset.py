"""Inspect reviewed originals and the locked online augmentation; no capture or training."""

import argparse
import hashlib
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["YOLO_CONFIG_DIR"] = str(ROOT / "logs/ultralytics_settings")
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ["YOLO_OFFLINE"] = "true"

import cv2
import numpy as np

from app.product_data import RAW, catalog, digest, identifier, image_read, image_write, now, read_json, write_json
from app.product_dataset import (apply_preview, inspect_transformed, inventory, load_policy, locked_source,
                                 make_pipeline, overlay, sampling_budget)


def tile(image, title, width=236):
    canvas = np.full((width + 26, width, 3), 245, np.uint8)
    h, w = image.shape[:2]
    ratio = width / max(h, w)
    resized = cv2.resize(image, (round(w * ratio), round(h * ratio)))
    top = 26 + (width - resized.shape[0]) // 2
    left = (width - resized.shape[1]) // 2
    canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    cv2.putText(canvas, title, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (30, 30, 30), 1, cv2.LINE_AA)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", default="pilot_20261003130657_all")
    parser.add_argument("--augment-policy", default="configs/augment_products.yaml")
    args = parser.parse_args()
    group = identifier(args.group)
    source = locked_source()
    policy_path = ROOT / args.augment_policy
    policy = load_policy(policy_path)
    manifest_path = RAW / "sessions" / (group + ".json")
    manifest = read_json(manifest_path)
    checked = inventory(RAW, manifest)
    policy_hash = digest(policy_path)
    classes_hash = digest(ROOT / "data/desktop/metadata/classes.json")
    catalog()
    checked.update({"updated_at": now(), "source_manifest": str(manifest_path.relative_to(ROOT)).replace("\\", "/"),
                    "manifest_sha256": digest(manifest_path), "classes_sha256": classes_hash})
    budget_path = ROOT / "data/desktop/metadata/sampling_budget.json"
    prior_budget = read_json(budget_path) if budget_path.exists() else None
    budget = sampling_budget(RAW, [s["image_id"] for s in checked["samples"]], prior_budget)
    pipeline = make_pipeline(policy)
    results = []
    rows = []
    selected = {0, 2, 4, 8, 12, 18}
    for index, sample in enumerate(checked["samples"]):
        if sample["status"] == "excluded":
            continue
        image = image_read(ROOT / sample["image_path"])
        review = read_json(ROOT / sample["review_path"])
        polygons = review["polygons"]
        class_ids = [p["class_id"] for p in polygons]
        row = []
        if index in selected:
            original_points = [np.array(p["points"]) * [image.shape[1], image.shape[0]] for p in polygons]
            row.append(tile(overlay(image, original_points, class_ids), f"{index + 1:02d} original | {len(polygons)} obj"))
        for base_seed in policy["preview_seeds"]:
            seed = base_seed + 1000 * index
            augmented = apply_preview(pipeline, image, polygons, policy["imgsz"], seed)
            check = inspect_transformed(augmented, class_ids)
            if check["output_shape_hw"] != [policy["imgsz"], policy["imgsz"]]:
                raise ValueError("Unexpected augmentation output dimensions")
            # Check the same input/seed reproduces image bytes AND geometry/class filtering.
            repeated = apply_preview(pipeline, image, polygons, policy["imgsz"], seed)
            if (not np.array_equal(augmented["img"], repeated["img"]) or
                    not np.array_equal(augmented["cls"], repeated["cls"]) or
                    not np.array_equal(augmented["instances"].segments, repeated["instances"].segments)):
                raise ValueError("Seeded augmentation is not reproducible")
            check.update({"image_id": sample["image_id"], "capture_group_id": sample["group_id"],
                          "effective_seed": seed, "base_seed": base_seed,
                          "augmented_image_sha256": hashlib.sha256(augmented["img"].tobytes()).hexdigest(),
                          "repeated_seed_identical": True, "usage": "diagnostic_preview_only"})
            results.append(check)
            if index in selected:
                row.append(tile(overlay(augmented["img"], augmented["instances"].segments,
                                        augmented["cls"].flatten().astype(int)),
                                f"seed {base_seed} | {check['retained_instances']} obj"))
        if row:
            rows.append(np.concatenate(row, axis=1))
    for sample in checked["samples"]:
        for key, path_key in (("image_sha256", "image_path"), ("review_sha256", "review_path"),
                              ("label_sha256", "label_path")):
            if sample[path_key] and digest(ROOT / sample[path_key]) != sample[key]:
                raise ValueError("Source changed during diagnostic preview")
    protected = read_json(ROOT / "reports/data/M2-02_tool_check.json")["protected_files"]
    for item in protected:
        if digest(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("Protected M0/M1 input changed: " + item["path"])
    if digest(policy_path) != policy_hash or digest(manifest_path) != checked["manifest_sha256"]:
        raise ValueError("Policy or source manifest changed during preview")
    preview_path = RAW / "annotations" / (group + "_augmentation_contact.jpg")
    image_write(preview_path, np.concatenate(rows, axis=0))
    budget["updated_at"] = now()
    report = {
        "schema_version": 1, "task": "M2-03", "checked_at": now(), "status": "inventory_and_augmentation_preview_passed",
        "source": source, "augmentation_policy": str(policy_path.relative_to(ROOT)),
        "augmentation_policy_sha256": policy_hash, "script_sha256": digest(Path(__file__)),
        "inventory_images": len(checked["samples"]), "class_instance_counts": checked["class_instance_counts"],
        "capture_groups": checked["capture_groups"], "exact_duplicate_pairs": checked["exact_duplicate_pairs"],
        "similarity_candidates": checked["similarity_candidates"], "preview_variants": len(results),
        "input_instance_appearances": sum(r["input_instances"] for r in results),
        "retained_instance_appearances": sum(r["retained_instances"] for r in results),
        "filtered_instance_appearances": sum(r["filtered_instances"] for r in results),
        "negative_variants_still_empty": sum(r["negative_still_empty"] is True for r in results),
        "preview": str(preview_path.relative_to(ROOT)).replace("\\", "/"), "preview_sha256": digest(preview_path),
        "augmentation_checks": results, "sampling_budget": budget,
        "protected_input_hash_checks": protected,
        "source_module_sha256": digest(ROOT / "app/product_dataset.py"),
        "unchanged_inputs": ["original images", "reviewed JSON/TXT", "COCO drafts", "E0 config and weights", "source lock"],
        "limits": [
            "Preview harness uses v8_transforms and normalized reviewed polygons; not the training loader or a training run.",
            "No split assigned. All previews are diagnostic, not new training/validation/test images.",
            "Existing capture groups are related. Formal independent validation/test capture is deferred.",
            "Cropping/filtering by the official transform is counted explicitly, not a model detection result.",
            "Perceptual hash candidates are advisory and never auto-delete or relabel originals.",
        ],
    }
    write_json(ROOT / "data/desktop/metadata/dataset_inventory.json", checked)
    write_json(budget_path, budget)
    output = ROOT / "reports/data/M2-03_dataset_check.json"
    write_json(output, report)
    print(f"Inventory: {len(checked['samples'])} originals; instances {checked['class_instance_counts']}; "
          f"capture groups {len(checked['capture_groups'])}.")
    print(f"Preview: {len(results)} variants; retained {report['retained_instance_appearances']}/"
          f"{report['input_instance_appearances']} instance appearances; "
          f"negative variants {report['negative_variants_still_empty']}.")
    print(f"Additional real-image budget: {budget['additional_real_images_used']}/30 used; no capture requested.")
    print("Report:", output)
    print("Preview:", preview_path)


if __name__ == "__main__":
    main()
