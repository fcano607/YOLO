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


def controlled_reference(config):
    """Freeze the completed E1 evidence; an augmentation run cannot rewrite its reference."""
    reference = config["reference"]
    for name in ("config", "training_report", "validation_report", "error_report"):
        if digest(ROOT / reference[name]) != reference[name + "_sha256"]:
            raise ValueError("Augmentation control reference changed: " + name)
    if reference["config"] != "configs/train_baseline.yaml":
        raise ValueError("Augmentation control must reference the frozen E1 baseline")
    training = read_json(ROOT / reference["training_report"])
    if (training["status"] != "completed" or training["purpose"] != "formal_products_e1" or
            training["config_sha256"] != reference["config_sha256"] or
            len(training["training"]["epochs"]) != 50 or training["training"]["optimizer_steps"] != 150):
        raise ValueError("Expected completed 50-epoch/150-update E1 reference")
    check_guards(training["guarded_inputs"])
    for weight in training["training"]["checkpoints"].values():
        if digest(ROOT / weight["path"]) != weight["sha256"]:
            raise ValueError("Reference E1 checkpoint changed")
    validation = read_json(ROOT / reference["validation_report"])
    if (validation["status"] != "completed" or validation["split"] != "val" or
            validation["checkpoint"] != training["training"]["checkpoints"]["best"] or
            validation["training_run"] != Path(training["run_directory"]).name):
        raise ValueError("Expected completed E1 best validation reference")
    return training


def experiment_configuration(path):
    """Keep E1 strict; allow only an explicit, separate rotation/scale control."""
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if config["purpose"] == "formal_products_e1":
        return baseline_configuration(path)
    if config["purpose"] != "formal_products_augmentation_control" or config["experiment"] != "E1-A":
        raise ValueError("Unknown formal product experiment")
    controlled_reference(config)
    base, base_args, loading = baseline_configuration(ROOT / config["reference"]["config"])
    expected = {**base, "purpose": "formal_products_augmentation_control", "experiment": "E1-A",
                "augment_policy": "configs/augment_products_e1a.yaml",
                "preflight_report": "reports/experiments/E1A_setup.json",
                "augmentation_acceptance": "reports/experiments/E1A_augmentation_precheck.json",
                "reference": config["reference"],
                "train_args": {**base["train_args"], "name": "E1A_products_v1_seed42"}}
    if config != expected:
        raise ValueError("E1-A may change only the declared augmentation policy and separate run metadata")
    policy = load_policy(ROOT / config["augment_policy"])
    original_policy = load_policy(ROOT / base["augment_policy"])
    expected_policy = {**original_policy, "train_args": {**original_policy["train_args"],
                                                       "degrees": 45.0, "scale": 0.4}}
    if policy != expected_policy:
        raise ValueError("E1-A must change only degrees=45 and scale=0.4")
    overrides = {**base_args, **policy["train_args"], **config["train_args"]}
    overrides.update(data=base_args["data"], model=base_args["model"], project=base_args["project"], task="segment")
    from ultralytics.cfg import get_cfg
    get_cfg(overrides=overrides)
    return config, overrides, loading


def verify_augmentation_acceptance(config_path, config):
    if config["purpose"] != "formal_products_augmentation_control":
        return
    report = read_json(ROOT / config["augmentation_acceptance"])
    if (report["status"] != "passed" or report["visual_review"]["accepted"] is not True or
            report["config_sha256"] != digest(config_path) or
            report["policy_sha256"] != digest(ROOT / config["augment_policy"]) or
            report["loading_contract_sha256"] != digest(ROOT / config["loading_contract"])):
        raise ValueError("E1-A augmentation needs current loader and visual acceptance before GPU preflight")
    check_guards(report["guarded_inputs"])
    for path, sha in report["implementation_sha256"].items():
        if digest(ROOT / path) != sha:
            raise ValueError("Augmentation check implementation changed; repeat acceptance")


def experiment_guards(config_path, config, loading):
    guards = original_guards(loading["samples"])
    for path in (relative(config_path), config["data"], config["loading_contract"], config["data_acceptance"],
                 config["augment_policy"], config["weights"], loading["source_split_path"],
                 "configs/source-lock.json", "configs/e0.yaml", "environment.yml", "configs/constraints-runtime.txt"):
        guards[path] = digest(ROOT / path)
    for entry in loading["lists"].values():
        guards[entry["path"]] = entry["sha256"]
    if config["purpose"] == "formal_products_augmentation_control":
        for name in ("config", "training_report", "validation_report", "error_report"):
            path = config["reference"][name]
            guards[path] = config["reference"][name + "_sha256"]
        baseline = read_json(ROOT / config["reference"]["training_report"])
        for checkpoint in baseline["training"]["checkpoints"].values():
            guards[checkpoint["path"]] = checkpoint["sha256"]
        original_policy = loading["augment_policy"]
        guards[original_policy] = digest(ROOT / original_policy)
        # The augmentation check itself creates this report; the runner requires it before starting.
        acceptance = ROOT / config["augmentation_acceptance"]
        if acceptance.exists():
            guards[relative(acceptance)] = digest(acceptance)
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
