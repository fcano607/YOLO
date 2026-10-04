"""Check E1-A train-only augmentation through the real loader before any optimizer update."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from app.product_data import digest, image_write, now, read_json, write_json
from app.product_dataset import load_policy, locked_source
from app.product_experiment import (experiment_configuration, experiment_guards,
                                    verify_augmentation_acceptance, verify_pool_files)
from app.product_training import check_guards, relative, verify_formal_loading
from scripts.check_training_data import check_batch, contact_sheet, tile
from scripts.train import environment_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_augment_control.yaml")
    parser.add_argument("--check", action="store_true", help="Verify the saved acceptance without creating outputs")
    args = parser.parse_args()
    source = locked_source()
    path = ROOT / args.config
    config, overrides, loading = experiment_configuration(path)
    if config["purpose"] != "formal_products_augmentation_control":
        raise ValueError("This check is only for the separate augmentation control")
    if args.check:
        verify_augmentation_acceptance(path, config)
        print("Saved augmentation acceptance is current")
        return
    report_path = ROOT / config["augmentation_acceptance"]
    output = ROOT / "runs/precheck/E1A_augmentation"
    if report_path.exists() or output.exists():
        raise ValueError("Augmentation evidence already exists; use --check to verify it")
    guards = experiment_guards(path, config, loading)
    fingerprint = environment_fingerprint()
    policy = load_policy(ROOT / config["augment_policy"])
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset, img2label_paths
    from ultralytics.utils.torch_utils import init_seeds
    data = check_det_dataset(str(ROOT / config["data"]), autodownload=False)
    samples = {s["image_id"]: s for s in loading["samples"]}
    runs, negatives, previews = [], [], []
    for seed in policy["preview_seeds"]:
        init_seeds(seed, deterministic=True)
        hyp = get_cfg(overrides={**overrides, "seed": seed})
        dataset = build_yolo_dataset(hyp, data["train"], 8, data, mode="train", rect=False, stride=32)
        verify_pool_files(dataset, loading, "train")
        if dataset.augment is not True:
            raise ValueError("Control must use the training transform")
        for image, label in zip(dataset.im_files, img2label_paths(dataset.im_files)):
            sample = samples[Path(image).stem]
            if (sample["split"] != "train" or Path(label).resolve() !=
                    (ROOT / sample["staged_label_path"]).resolve() or digest(label) != sample["label_sha256"]):
                raise ValueError("Control redirected the reviewed train image/label pair")
        loader = build_dataloader(dataset, 8, workers=0, shuffle=True, device=torch.device("cpu"))
        records, tiles = [], []
        for batch in loader:
            records.extend(check_batch(batch, samples))
            tiles.extend(tile(batch, i, "E1A s" + str(seed)) for i in range(len(batch["img"])))
        if sorted(r["image_id"] for r in records) != sorted(s["image_id"] for s in samples.values() if s["split"] == "train"):
            raise ValueError("Augmentation loader skipped or duplicated train sources")
        negative_indices = [i for i, label in enumerate(dataset.labels) if not len(label["cls"])]
        checks = check_batch(dataset.collate_fn([dataset[i] for i in negative_indices]), samples)
        if len(checks) != 4 or not all(r["negative_still_empty"] for r in checks):
            raise ValueError("Augmentation introduced labels into negative images")
        negative_checks = {"seed": seed, "records": checks}
        negatives.append(negative_checks)
        preview = output / ("train_seed" + str(seed) + ".jpg")
        image_write(preview, contact_sheet(tiles, columns=4))
        previews.append(relative(preview))
        runs.append({"seed": seed, "images": len(records), "batches": len(loader), "records": records,
                     "retained_instances": sum(len(r["loss_class_ids"]) for r in records),
                     "filtered_instances": sum(r["filtered_instances"] for r in records)})
    # Validate the five unchanged validation inputs; no final test loader or model is created.
    hyp = get_cfg(overrides=overrides)
    val = build_yolo_dataset(hyp, data["val"], 8, data, mode="val", rect=True, stride=32)
    verify_pool_files(val, loading, "val")
    validation = []
    for batch in build_dataloader(val, 8, workers=0, shuffle=False, device=torch.device("cpu")):
        validation.extend(check_batch(batch, samples))
    if val.augment or len(validation) != 5 or any(r["filtered_instances"] for r in validation):
        raise ValueError("Augmentation control changed the validation inputs")
    check_guards(guards)
    verify_formal_loading()
    if environment_fingerprint() != fingerprint:
        raise ValueError("Environment changed during augmentation check")
    report = {"schema_version": 1, "purpose": "E1A_joint_augmentation_loader_check", "checked_at": now(),
              "status": "automated_checks_passed_pending_visual_acceptance", "source": source,
              "config_path": relative(path), "config_sha256": digest(path),
              "policy_path": config["augment_policy"], "policy_sha256": digest(ROOT / config["augment_policy"]),
              "loading_contract_sha256": digest(ROOT / config["loading_contract"]),
              "guarded_inputs": guards, "environment_fingerprint": fingerprint,
              "implementation_sha256": {p: digest(ROOT / p) for p in
                                         ("scripts/check_augmentation_control.py", "scripts/check_training_data.py",
                                          "app/product_experiment.py", "app/product_dataset.py")},
              "training_seeds": runs, "negative_batches": negatives, "validation_records": validation,
              "preview_paths": previews, "guarded_inputs_unchanged": True,
              "optimizer_steps": 0, "new_photos": 0, "test_images_loaded": 0, "test_images_inferred": 0,
              "note": "Official clipping/filtering may remove transformed targets; retained targets must have matching classes, valid boxes and nonempty aligned masks. Visual acceptance is required separately."}
    write_json(report_path, report)
    print("Automatic checks completed; inspect three train-only contact sheets:", relative(report_path))
    print("Retained/filtered instances:", [(r["seed"], r["retained_instances"], r["filtered_instances"]) for r in runs])


if __name__ == "__main__":
    main()
