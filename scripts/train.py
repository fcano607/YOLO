"""Run the debug probe or the prechecked formal E1 configuration; preserve original E0 evidence."""

import argparse
import contextlib
import csv
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from app.product_data import catalog, digest, identifier, now, read_json, write_json
from app.product_dataset import locked_source
from app.product_training import (check_guards, debug_configuration, original_guards, relative, verify_debug_split)


class Tee:
    def __init__(self, terminal, file):
        self.terminal, self.file = terminal, file

    def write(self, value):
        self.terminal.write(value)
        self.file.write(value)
        self.file.flush()

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def isatty(self):
        return False


def environment_fingerprint():
    pairs = sorted((d.metadata["Name"].lower(), d.version) for d in metadata.distributions() if d.metadata["Name"])
    return hashlib.sha256(json.dumps(pairs).encode()).hexdigest()


def output_shapes(value):
    if isinstance(value, torch.Tensor):
        return [list(value.shape)]
    if isinstance(value, dict):
        return [shape for item in value.values() for shape in output_shapes(item)]
    if isinstance(value, (list, tuple)):
        return [shape for item in value for shape in output_shapes(item)]
    return []


def check_saved_threshold(name, report_path):
    """Explicitly recheck an existing run at a normal display threshold without training."""
    from ultralytics import YOLO
    report = read_json(report_path)
    if report["status"] != "passed" or Path(report["run_directory"]).name != name:
        raise ValueError("Expected the successful debug run with this exact name")
    check_guards(report["guarded_inputs"])
    manifest = verify_debug_split()
    results = []
    for kind, checkpoint in report["training"]["checkpoints"].items():
        path = ROOT / checkpoint["path"]
        if digest(path) != checkpoint["sha256"]:
            raise ValueError("Saved checkpoint changed")
        model = YOLO(str(path))
        predictions = []
        for sample in manifest["samples"]:
            if sample["split"] != "val":
                continue
            result = model.predict(str(ROOT / sample["image_path"]), imgsz=640, device=0,
                                   conf=0.25, iou=0.7, max_det=300, verbose=False, save=False)[0]
            predictions.append({"image_id": sample["image_id"], "prediction_count": len(result.boxes),
                                "class_ids": result.boxes.cls.int().tolist(), "scores": result.boxes.conf.tolist()})
        results.append({"checkpoint": kind, "predictions": predictions})
    report["normal_threshold_check"] = {"checked_at": now(), "conf": 0.25, "iou": 0.7, "max_det": 300,
                                         "script_sha256": digest(Path(__file__)), "results": results,
                                         "purpose": "Check current display readiness on debug validation, not independent accuracy."}
    check_guards(report["guarded_inputs"])
    write_json(report_path, report)
    print("conf=0.25:", [(r["checkpoint"], [p["prediction_count"] for p in r["predictions"]]) for r in results])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_debug.yaml")
    parser.add_argument("--name", help="A fresh run name for an intentional rerun; never overwrite an earlier run")
    parser.add_argument("--check-saved", action="store_true", help="Only check an existing successful run at conf=0.25")
    parser.add_argument("--preflight", action="store_true", help="Formal configuration/GPU checks only; no optimizer updates")
    args = parser.parse_args()
    source = locked_source()
    config_path = ROOT / args.config
    import yaml
    if yaml.safe_load(config_path.read_text(encoding="utf-8"))["purpose"] == "formal_products_e1":
        if args.check_saved:
            raise ValueError("Use evaluate.py for completed formal E1 checkpoints")
        from app.product_runner import run
        run(config_path, source, args.preflight, args.name)
        return
    if args.preflight:
        raise ValueError("--preflight requires configs/train_baseline.yaml")
    manifest = verify_debug_split()
    config, overrides = debug_configuration(config_path)
    if args.name:
        overrides["name"] = identifier(args.name)
    name = identifier(overrides["name"])
    run_path = Path(overrides["project"]) / name
    report_path = ROOT / "reports/experiments" / (name + ".json")
    log_path = ROOT / "logs/training" / (name + ".log")
    if args.check_saved:
        check_saved_threshold(name, report_path)
        return
    if any(p.exists() for p in (run_path, report_path, log_path)):
        raise ValueError("Debug run already exists; use a new --name to preserve its evidence")
    precheck = read_json(ROOT / "reports/data/M2_debug_precheck.json")
    if (precheck["train_config_sha256"] != digest(config_path) or
            precheck["split_sha256"] != digest(ROOT / config["split_manifest"]) or
            precheck["augment_policy_sha256"] != digest(ROOT / config["augment_policy"])):
        raise ValueError("Run check_training_data.py for this exact configuration before training")
    if not torch.cuda.is_available():
        raise ValueError("GPU debug training requires the existing CUDA environment")
    weight_path = ROOT / config["weights"]
    if digest(weight_path) != config["weights_sha256"]:
        raise ValueError("Pretrained weights differ from the locked M0 weights")
    guards = original_guards(manifest["samples"])
    for path in ("configs/e0.yaml", "configs/source-lock.json", "environment.yml",
                 "configs/constraints-runtime.txt", "reports/model/M1-04/postprocess_check.json",
                 config["weights"], relative(config_path), config["augment_policy"], config["data"],
                 config["split_manifest"]):
        guards[path] = digest(ROOT / path)
    environment = environment_fingerprint()
    report = {"schema_version": 1, "started_at": now(), "status": "running", "purpose": "pipeline_debug_only",
              "formal_independent_evaluation": False, "source": source, "dataset_summary": manifest["summary"],
              "configuration": overrides, "config_path": relative(config_path), "guarded_inputs": guards,
              "python": sys.version, "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(0), "environment_fingerprint_before": environment,
              "log": relative(log_path), "run_directory": relative(run_path)}
    write_json(report_path, report)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("w", encoding="utf-8") as log, contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            from ultralytics import YOLO
            model = YOLO(str(weight_path))
            pretrained = {k: v.detach().cpu().clone() for k, v in model.model.state_dict().items()}
            evidence = {"batches": [], "epochs": [], "optimizer_steps": 0, "nonzero_gradient_steps": 0}
            initial = {}

            def step_hook(optimizer, unused_args, unused_kwargs):
                evidence["optimizer_steps"] += 1
                gradients = [p.grad for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
                if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                    raise ValueError("Missing or non-finite optimizer gradients")
                if any(bool(g.count_nonzero()) for g in gradients):
                    evidence["nonzero_gradient_steps"] += 1

            def on_start(trainer):
                net = trainer.model
                names = {c["id"]: c["name"] for c in catalog()["classes"]}
                if net.names != names or net.model[-1].nc != 3:
                    raise ValueError("Training did not construct the actual three-class product head")
                initial.update({k: v.detach().cpu().clone() for k, v in net.named_parameters() if v.requires_grad})
                current = net.state_dict()
                matching = [k for k in current if k in pretrained and current[k].shape == pretrained[k].shape]
                different = [k for k in matching if not torch.equal(current[k].cpu(), pretrained[k])]
                if different:
                    raise ValueError("Compatible pretrained tensor did not transfer: " + str(different[:3]))
                evidence["initialization"] = {"names": net.names, "head_nc": net.model[-1].nc,
                                              "mask_coefficients": net.model[-1].nm,
                                              "matching_pretrained_tensors_verified_equal": len(matching),
                                              "new_or_shape_changed_keys": [k for k in current if k not in matching]}
                evidence["optimizer"] = type(trainer.optimizer).__name__
                evidence["actual_batch"] = trainer.batch_size
                evidence["accumulate"] = trainer.accumulate
                evidence["amp"] = trainer.amp
                evidence["optimizer_parameter_groups"] = [
                    {k: v for k, v in group.items() if k != "params"} for group in trainer.optimizer.param_groups]
                trainer.optimizer.register_step_post_hook(step_hook)

            def on_batch(trainer):
                losses = {k: float(v.detach()) for k, v in trainer.loss_items.items()}
                if not losses or not all(math.isfinite(v) for v in losses.values()):
                    raise ValueError("Non-finite batch loss")
                evidence["batches"].append({"epoch": trainer.epoch + 1, "losses": losses})

            def on_epoch(trainer):
                if trainer.epoch >= overrides["epochs"]:
                    evidence["final_best_debug_validation"] = {k: float(v) for k, v in trainer.metrics.items()}
                    return
                evidence["epochs"].append({"epoch": trainer.epoch + 1,
                    "mean_train_losses": {k: float(v.detach()) for k, v in trainer.tloss.items()},
                    "debug_validation_metrics": {k: float(v) for k, v in trainer.metrics.items()},
                    "lr": trainer.lr})

            def on_end(trainer):
                changed = {}
                for key, value in trainer.model.named_parameters():
                    if key in initial:
                        delta = float((value.detach().cpu() - initial[key]).abs().max())
                        if not math.isfinite(delta):
                            raise ValueError("Non-finite trained parameter")
                        if delta > 0:
                            changed[key] = delta
                evidence["updated_parameter_tensors"] = len(changed)
                evidence["parameter_max_abs_change"] = changed
                if not changed or evidence["optimizer_steps"] == 0 or evidence["nonzero_gradient_steps"] == 0:
                    raise ValueError("Training did not update parameters")

            model.add_callback("on_train_start", on_start)
            model.add_callback("on_train_batch_end", on_batch)
            model.add_callback("on_fit_epoch_end", on_epoch)
            model.add_callback("on_train_end", on_end)
            torch.cuda.reset_peak_memory_stats(0)
            started = time.perf_counter()
            model.train(**overrides)
            torch.cuda.synchronize(0)
            evidence["train_wall_seconds"] = time.perf_counter() - started
            evidence["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
            if (len(evidence["batches"]) != len(model.trainer.train_loader) * overrides["epochs"] or
                    len(evidence["epochs"]) != overrides["epochs"]):
                raise ValueError("Debug training did not consume the expected 4 batches per epoch")
            trainer = model.trainer
            evidence["checkpoints"] = {key: {"path": relative(path), "sha256": digest(path)}
                                       for key, path in (("best", trainer.best), ("last", trainer.last))}
            csv_path = trainer.save_dir / "results.csv"
            with csv_path.open(encoding="utf-8") as file:
                rows = [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(file)]
            if len(rows) != overrides["epochs"]:
                raise ValueError("Missing epoch rows in results.csv")
            evidence["results_csv"] = {"path": relative(csv_path), "sha256": digest(csv_path), "rows": rows}
            # Reload both saved checkpoints, then execute real inference on all six debug validation originals.
            reloaded = []
            val_samples = [s for s in manifest["samples"] if s["split"] == "val"]
            for checkpoint_name in ("best", "last"):
                fresh = YOLO(str(ROOT / evidence["checkpoints"][checkpoint_name]["path"]))
                fresh.model.to("cuda:0").eval()
                if fresh.model.names != {c["id"]: c["name"] for c in catalog()["classes"]}:
                    raise ValueError("Saved weight lost the product mapping")
                with torch.no_grad():
                    raw = fresh.model(torch.zeros(1, 3, 640, 640, device="cuda:0"))
                shapes = output_shapes(raw)
                if [1, 39, 8400] not in shapes or [1, 32, 160, 160] not in shapes:
                    raise ValueError("Reloaded three-class segmentation raw outputs have unexpected shapes")
                predictions = []
                for sample in val_samples:
                    result = fresh.predict(str(ROOT / sample["image_path"]), imgsz=640, device=0,
                                           conf=0.001, iou=0.7, max_det=10, verbose=False, save=False)[0]
                    count = len(result.boxes)
                    if count and (result.masks is None or len(result.masks.data) != count or
                                  not torch.isfinite(result.masks.data).all() or not torch.isfinite(result.boxes.data).all()):
                        raise ValueError("Reloaded boxes and instance masks are inconsistent")
                    predictions.append({"image_id": sample["image_id"], "prediction_count": count,
                                        "class_ids": result.boxes.cls.int().tolist(),
                                        "scores": result.boxes.conf.tolist(),
                                        "mask_shape": list(result.masks.data.shape) if result.masks is not None else None})
                reloaded.append({"checkpoint": checkpoint_name, "names": fresh.model.names, "raw_output_shapes": shapes,
                                 "inference_conf": 0.001, "inference_max_det": 10,
                                 "purpose": "Low-score diagnostic inference only, not a display or accuracy threshold.",
                                 "predictions": predictions})
            if not any(p["prediction_count"] for r in reloaded for p in r["predictions"]):
                raise ValueError("No instance-mask output even at the diagnostic threshold")
            evidence["reload_checks"] = reloaded
            report["training"] = evidence
        check_guards(guards)
        verify_debug_split()
        locked_source()
        after = environment_fingerprint()
        if environment != after:
            raise ValueError("Installed package versions changed during debug training")
        report.update(status="passed", completed_at=now(), guarded_inputs_unchanged=True,
                      environment_fingerprint_after=after,
                      conclusion="Real three-class training, finite losses, optimizer updates, save/reload and mask inference passed. Related debug validation is not independent quality evidence.")
        write_json(report_path, report)
        print("PASSED:", relative(report_path))
    except Exception as exc:
        report.update(status="failed", completed_at=now(), error=repr(exc))
        write_json(report_path, report)
        raise


if __name__ == "__main__":
    main()
