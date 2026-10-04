"""Formal E1 configuration, mask-first selection and reproducible run boundaries."""
import math
from pathlib import Path

import yaml

from app.product_data import ROOT, catalog, digest, identifier, read_json
from app.product_dataset import load_policy
from app.product_training import check_guards, original_guards, relative, verify_formal_loading

MASK_METRIC = "metrics/mAP50-95(M)"
SELECTION = {"metric": MASK_METRIC, "direction": "maximize", "ties": "latest_equal_epoch"}


def mask_fitness(metrics):
    if MASK_METRIC not in metrics:
        raise ValueError("Validation did not return the required mask mAP50-95")
    value = metrics[MASK_METRIC]
    if isinstance(value, bool) or not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
        raise ValueError("Invalid mask selection metric")
    return float(value)


def baseline_configuration(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if (config["schema_version"] != 1 or config["purpose"] != "formal_products_e1" or
            config["experiment"] != "E1" or config["dataset_version"] != "products-v1" or
            config["selection"] != SELECTION):
        raise ValueError("Expected the frozen E1 mask-first experiment definition")
    loading = verify_formal_loading()
    if (config["data"] != loading["data_yaml"] or
            config["loading_contract"] != "data/desktop/metadata/products-v1_loading.json" or
            config["data_acceptance"] != loading["acceptance_report"] or
            config["augment_policy"] != loading["augment_policy"]):
        raise ValueError("E1 must use the accepted formal dataset and augmentation contract")
    accepted = read_json(ROOT / config["data_acceptance"])
    if (accepted["status"] != "formal_quality_and_loader_accepted" or
            accepted["loading_contract_sha256"] != digest(ROOT / config["loading_contract"]) or
            accepted["training_loader_verified_for_this_version"] is not True):
        raise ValueError("Formal data acceptance is missing or stale")
    policy = load_policy(ROOT / config["augment_policy"])
    overrides = {**policy["train_args"], **config["train_args"]}
    if any(overrides[k] != v for k, v in policy["train_args"].items()) or overrides["imgsz"] != policy["imgsz"]:
        raise ValueError("E1 changed the accepted augmentation policy")
    ev = config["evaluation"]
    if (ev != {"split": "val", "metric_conf": 0.001, "nms_iou": 0.7, "max_det": 300,
               "display_conf": 0.25, "augment": False} or
            overrides["split"] != "val" or overrides["conf"] != ev["metric_conf"] or
            overrides["iou"] != ev["nms_iou"] or overrides["max_det"] != ev["max_det"]):
        raise ValueError("E1 validation must use the declared val pool and metric thresholds")
    if (overrides["resume"] is not False or overrides["pretrained"] is not True or
            config["weights"] != "artifacts/pretrained/yolo11n-seg.pt" or
            digest(ROOT / config["weights"]) != config["weights_sha256"]):
        raise ValueError("E1 must initialize from the locked original pretrained weights")
    if (not isinstance(overrides["batch"], int) or overrides["batch"] < 2 or
            overrides["nbs"] != overrides["batch"] or overrides["workers"] != 0 or
            overrides["amp"] is not False or overrides["optimizer"] != "AdamW" or
            overrides["device"] != 0 or overrides["patience"] != 0 or
            overrides["classes"] is not None or overrides["single_cls"] is not False or
            overrides["fraction"] != 1.0 or overrides["freeze"] is not None or
            overrides["compile"] is not False or overrides["exist_ok"] is not False or
            overrides["save"] is not True or overrides["val"] is not True or
            overrides["project"] != "runs/train"):
        raise ValueError("E1 requires the declared complete single-GPU FP32 training budget")
    identifier(overrides["name"])
    overrides.update(data=str(ROOT / config["data"]), model=str(ROOT / config["weights"]), task="segment",
                     project=str(ROOT / overrides["project"]))
    from ultralytics.cfg import get_cfg
    get_cfg(overrides=overrides)
    return config, overrides, loading


def experiment_guards(config_path, config, loading):
    guards = original_guards(loading["samples"])
    for path in (relative(config_path), config["data"], config["loading_contract"], config["data_acceptance"],
                 config["augment_policy"], config["weights"], loading["source_split_path"],
                 "configs/source-lock.json", "configs/e0.yaml", "environment.yml", "configs/constraints-runtime.txt"):
        guards[path] = digest(ROOT / path)
    for entry in loading["lists"].values():
        guards[entry["path"]] = entry["sha256"]
    return guards


def fresh_run_paths(overrides, name=None):
    chosen = identifier(name or overrides["name"])
    run = Path(overrides["project"]) / chosen
    report = ROOT / "reports/experiments" / (chosen + ".json")
    log = ROOT / "logs/training" / (chosen + ".log")
    if any(p.exists() for p in (run, report, log)):
        raise ValueError("Run already exists; use a fresh --name to preserve its evidence")
    return chosen, run, report, log


def verify_preflight(config_path, config):
    report = read_json(ROOT / config["preflight_report"])
    if report["status"] != "passed" or report["config_sha256"] != digest(config_path):
        raise ValueError("Run --preflight for this exact formal configuration before training")
    check_guards(report["guarded_inputs"])
    for path, sha in report["implementation_sha256"].items():
        if digest(ROOT / path) != sha:
            raise ValueError("Formal implementation changed after preflight; repeat --preflight")
    return report


def verify_product_model(model):
    from ultralytics.engine.model import Model
    # Loaded torch networks also have .task; only unwrap the public YOLO/Model wrapper.
    net = model.model if isinstance(model, Model) else model
    if (net.names != {c["id"]: c["name"] for c in catalog()["classes"]} or
            net.model[-1].nc != 3 or not hasattr(net.model[-1], "proto")):
        raise ValueError("Expected the trained three-product segmentation model")


def verify_pool_files(dataset, loading, split):
    expected = {str((ROOT / s["staged_image_path"]).resolve())
                for s in loading["samples"] if s["split"] == split}
    actual = {str(Path(p).resolve()) for p in dataset.im_files}
    if expected != actual or len(dataset.im_files) != len(expected):
        raise ValueError("Actual model loader differs from the frozen " + split + " pool")
