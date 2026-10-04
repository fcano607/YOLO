"""Compare completed E1/E1-A on the same validation pool; never select using final test."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from app.product_data import digest, image_read, image_write, now, read_json, write_json
from app.product_dataset import locked_source
from app.product_experiment import experiment_configuration
from app.product_training import check_guards, relative, verify_formal_loading


def latest_report(pattern, expected_run):
    for path in sorted((ROOT / "reports/experiments").glob(pattern), reverse=True):
        report = read_json(path)
        if report.get("status") == "completed" and report.get("training_run") == expected_run:
            return path, report
    raise ValueError("No completed evaluation/analysis for " + expected_run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_augment_control.yaml")
    args = parser.parse_args()
    source = locked_source()
    config, unused_args, loading = experiment_configuration(ROOT / args.config)
    if config["experiment"] != "E1-A":
        raise ValueError("Expected the explicit augmentation control")
    base = read_json(ROOT / config["reference"]["training_report"])
    run = config["train_args"]["name"]
    candidate_path = ROOT / "reports/experiments" / (run + ".json")
    candidate = read_json(candidate_path)
    if candidate["status"] != "completed" or candidate["config_sha256"] != digest(ROOT / args.config):
        raise ValueError("Control training is not completed under this config")
    check_guards(candidate["guarded_inputs"])
    differences = {k: {"E1": base["configuration"].get(k), "E1-A": candidate["configuration"].get(k)}
                   for k in set(base["configuration"]) | set(candidate["configuration"])
                   if base["configuration"].get(k) != candidate["configuration"].get(k)}
    if set(differences) != {"degrees", "scale", "name"}:
        raise ValueError("Unexpected training-factor changes: " + repr(differences))
    for training in (base, candidate):
        check_guards(training["guarded_inputs"])
        if (len(training["training"]["epochs"]) != 50 or len(training["training"]["batches"]) != 150 or
                training["training"]["optimizer_steps"] != 150 or training["training"]["accumulate"] != 1 or
                training["source"]["commit"] != source["commit"] or training["test_evaluated"] is not False):
            raise ValueError("Experiments are not comparable under the declared budget/source")
        for checkpoint in training["training"]["checkpoints"].values():
            if digest(ROOT / checkpoint["path"]) != checkpoint["sha256"]:
                raise ValueError("Checkpoint changed after training")
    if base["environment_fingerprint_after"] != candidate["environment_fingerprint_after"]:
        raise ValueError("Runtime environments differ")
    for p in ("app/product_trainer.py",):
        if base["implementation_sha256"][p] != candidate["implementation_sha256"][p]:
            raise ValueError("Core trainer implementation differs")
    eval_path, evaluation = latest_report(run + "_best_val_*.json", run)
    analysis_path, analysis = latest_report("M3-05_" + run + "_*.json", run)
    base_eval = read_json(ROOT / config["reference"]["validation_report"])
    base_analysis = read_json(ROOT / config["reference"]["error_report"])
    best = candidate["training"]["checkpoints"]["best"]
    if (evaluation["checkpoint"] != best or analysis["checkpoint"] != best or
            analysis["test_images_inferred"] != 0 or evaluation["test_evaluated"] is not False):
        raise ValueError("Comparison evidence did not evaluate the selected best on val only")
    expected_ids = sorted(s["image_id"] for s in loading["samples"] if s["split"] == "val")
    for ev in (base_eval, evaluation):
        if (sorted(ev["metric_evaluation"]["image_ids"]) != expected_ids or
                ev["metric_evaluation"]["settings"] != base_eval["metric_evaluation"]["settings"]):
            raise ValueError("Validation inputs/settings differ")
    if analysis["settings"] != base_analysis["settings"]:
        raise ValueError("Threshold diagnostic settings differ")
    output = ROOT / "runs/analysis/E1A_vs_E1"
    report_path = ROOT / "reports/experiments/E1A_vs_E1_products_v1.json"
    if report_path.exists() or output.exists():
        raise ValueError("Comparison evidence already exists")
    # E1's report serialization was improved after its training. Re-evaluate both saved
    # checkpoints with one current metric implementation rather than comparing training-time hashes.
    from ultralytics import YOLO
    from app.product_evaluation import evaluate_model
    metric_checks = {}
    for label, training, recorded, config_path in (
            ("E1", base, base_eval, ROOT / config["reference"]["config"]),
            ("E1-A", candidate, evaluation, ROOT / args.config)):
        ev_config, ev_args, ev_loading = experiment_configuration(config_path)
        model = YOLO(str(ROOT / training["training"]["checkpoints"]["best"]["path"]))
        checked = evaluate_model(model.model, ev_config, ev_args, ev_loading,
                                 output / (label.replace("-", "") + "_metric_verification"), plots=False)
        for key, value in recorded["metric_evaluation"]["metrics"].items():
            if abs(checked["metrics"][key] - value) > 1e-8:
                raise ValueError("Saved validation metric could not be reproduced: " + label + " " + key)
        metric_checks[label] = checked
    metrics = {k: {"E1": v, "E1-A": evaluation["metric_evaluation"]["metrics"][k],
                   "delta": evaluation["metric_evaluation"]["metrics"][k] - v}
               for k, v in base_eval["metric_evaluation"]["metrics"].items()}
    per_class = []
    for b, c in zip(base_eval["metric_evaluation"]["per_class"], evaluation["metric_evaluation"]["per_class"]):
        if b["Class-ID"] != c["Class-ID"] or b["Instances"] != c["Instances"]:
            raise ValueError("Per-class validation support differs")
        per_class.append({"class_id": b["Class-ID"], "class": b["Class"], "instances": b["Instances"],
                          "metrics": {k: {"E1": b[k], "E1-A": c[k], "delta": c[k] - b[k]}
                                      for k in ("Box-mAP50", "Box-mAP50-95", "Mask-mAP50", "Mask-mAP50-95")}})
    records = {x["image_id"]: x for x in analysis["records"] if x["split"] == "val"}
    baseline_records = {x["image_id"]: x for x in base_analysis["records"] if x["split"] == "val"}
    if sorted(records) != expected_ids or sorted(baseline_records) != expected_ids:
        raise ValueError("Diagnostic pool IDs differ")
    tiles, case_rows = [], []
    for image_id in expected_ids:
        b, c = baseline_records[image_id], records[image_id]
        if b["target_classes"] != c["target_classes"] or b["target_boxes_xyxy"] != c["target_boxes_xyxy"]:
            raise ValueError("Comparison changed the truth")
        b_image, c_image = image_read(ROOT / b["preview"]), image_read(ROOT / c["preview"])
        if b_image.shape != c_image.shape or b_image.shape[:2] != (402, 1920):
            raise ValueError("Unexpected truth / .25 / .05 diagnostic panel layout")
        tiles.append(np.hstack((b_image[:, :1280], c_image[:, 640:1280])))
        def shown(row):
            return [{"class_id": cls, "score": score} for cls, score in zip(row["predicted_classes"], row["scores"]) if score >= .25]
        case_rows.append({"image_id": image_id, "target_classes": b["target_classes"],
                          "E1_shown": shown(b), "E1A_shown": shown(c),
                          "E1_targets": b["target_audits"], "E1A_targets": c["target_audits"]})
    preview = output / "val_GT_E1_E1A_conf025.jpg"
    image_write(preview, np.vstack(tiles))
    verify_formal_loading()
    check_guards(candidate["guarded_inputs"])
    report = {"schema_version": 1, "status": "completed", "purpose": "E1A_vs_E1_augmentation_control",
              "checked_at": now(), "source": source, "configuration": relative(ROOT / args.config),
              "training_factor_differences": differences, "comparable_budget": {"epochs": 50, "optimizer_steps": 150,
                         "train_images": 20, "val_images": 5, "test_images_inferred": 0, "seed": 42},
              "reference_paths": config["reference"], "candidate_evidence": {
                  "training": relative(candidate_path), "training_sha256": digest(candidate_path),
                  "evaluation": relative(eval_path), "evaluation_sha256": digest(eval_path),
                  "analysis": relative(analysis_path), "analysis_sha256": digest(analysis_path)},
              "metrics": metrics, "per_class": per_class,
              "metric_revalidation": {"same_implementation_sha256": digest(ROOT / "app/product_evaluation.py"),
                                      "both_previous_reports_reproduced": True, "absolute_tolerance": 1e-8,
                                      "results": metric_checks},
              "threshold_diagnostics": {"E1": base_analysis["summary"], "E1-A": analysis["summary"]},
              "cases": case_rows, "preview": relative(preview),
              "best_checkpoints": {"E1": base["training"]["checkpoints"]["best"], "E1-A": best},
              "selected_epochs": {"E1": base["training"]["selected_epoch"], "E1-A": candidate["training"]["selected_epoch"]},
              "new_photos": 0, "test_images_inferred": 0,
              "script_sha256": digest(Path(__file__)),
              "limitations": ["One seed and five repeatedly inspected validation images; no broad or final-test claim.",
                              "Rotation and scale changed together, so attribution is to the bundle, not an individual parameter.",
                              "Synthetic augmentation does not create new physical viewpoints or independent backgrounds."]}
    write_json(report_path, report)
    print("Saved comparison:", relative(report_path))
    print("mask mAP50-95:", metrics["metrics/mAP50-95(M)"])
    for name, diagnostic in report["threshold_diagnostics"].items():
        print(name, "val .25", [(x["tp"], x["fp"], x["fn"]) for x in diagnostic["val"]["0.25"]["mask"]])


if __name__ == "__main__":
    main()
