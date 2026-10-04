"""Evaluate a completed E1 checkpoint on frozen validation; --check only verifies the route."""
import argparse
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.product_data import digest, identifier, image_write, now, read_json, write_json
from app.product_dataset import locked_source
from app.product_experiment import experiment_configuration, verify_product_model
from app.product_evaluation import evaluate_model
from app.product_training import check_guards, relative, verify_formal_loading


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_baseline.yaml")
    parser.add_argument("--run", default="E1_products_v1_seed42")
    parser.add_argument("--checkpoint", choices=("best", "last"), default="best")
    parser.add_argument("--split", choices=("val",), default="val")
    parser.add_argument("--check", action="store_true", help="Only verify configuration and frozen val route; no model inference")
    args = parser.parse_args()
    source = locked_source()
    config_path = ROOT / args.config
    config, overrides, loading = experiment_configuration(config_path)
    if args.check:
        print("Evaluation route:", config["evaluation"], "; val images=", loading["summary"]["val"]["images"], "; test inference disabled")
        return
    run = identifier(args.run)
    training = read_json(ROOT / "reports/experiments" / (run + ".json"))
    if (training["status"] != "completed" or training["purpose"] != config["purpose"] or
            training["experiment"] != config["experiment"] or training["config_sha256"] != digest(config_path)):
        raise ValueError("Expected a completed formal product run under this exact configuration")
    check_guards(training["guarded_inputs"])
    checkpoint = training["training"]["checkpoints"][args.checkpoint]
    weight = ROOT / checkpoint["path"]
    if digest(weight) != checkpoint["sha256"]:
        raise ValueError("E1 checkpoint changed after training")
    from ultralytics import YOLO
    model = YOLO(str(weight))
    verify_product_model(model)
    stamp = datetime.fromisoformat(now()).strftime("%Y%m%d_%H%M%S")
    name = run + "_" + args.checkpoint + "_val_" + stamp
    save_dir = ROOT / "runs/eval" / name
    report_path = ROOT / "reports/experiments" / (name + ".json")
    if save_dir.exists() or report_path.exists():
        raise ValueError("Validation output already exists")
    result = evaluate_model(model.model, config, overrides, loading, save_dir)
    predictions = []
    for sample in loading["samples"]:
        if sample["split"] != "val":
            continue
        item = model.predict(str(ROOT / sample["image_path"]), imgsz=overrides["imgsz"], device=0,
                             conf=config["evaluation"]["display_conf"], iou=config["evaluation"]["nms_iou"],
                             max_det=config["evaluation"]["max_det"], verbose=False, save=False)[0]
        if len(item.boxes) and (item.masks is None or len(item.masks.data) != len(item.boxes)):
            raise ValueError("Validation boxes and instance masks differ")
        preview = save_dir / (sample["image_id"] + ".jpg")
        image_write(preview, item.plot())
        predictions.append({"image_id": sample["image_id"], "target_class_ids": [p["class_id"] for p in sample["instances"]],
                            "class_ids": item.boxes.cls.int().tolist(), "scores": item.boxes.conf.tolist(),
                            "boxes_xyxy": item.boxes.xyxy.tolist(), "prediction_count": len(item.boxes),
                            "preview": relative(preview)})
    check_guards(training["guarded_inputs"])
    verify_formal_loading()
    report = {"schema_version": 1, "status": "completed", "checked_at": now(), "source": source,
              "purpose": "formal_products_validation", "experiment": config["experiment"], "training_run": run,
              "checkpoint": checkpoint, "split": "val", "test_evaluated": False,
              "metric_evaluation": result, "display_threshold": config["evaluation"]["display_conf"],
              "predictions": predictions, "save_directory": relative(save_dir),
              "limitation": "Only five validation images of the same physical packages; metrics and previews do not establish broad generalization."}
    write_json(report_path, report)
    print("Validation completed:", relative(report_path))


if __name__ == "__main__":
    main()
