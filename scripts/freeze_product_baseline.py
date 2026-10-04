"""Freeze the accepted E1-A baseline once, or verify its artifacts without inference."""
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from app.product_data import catalog, digest, read_json, write_json
from app.product_dataset import locked_source
from app.product_training import check_guards, verify_formal_loading

BASELINE_ID = "products-v1_E1A_seed42_B1"
MANIFEST = "configs/baseline_products_v1.json"
RECEIPT = "reports/experiments/M3-06_products_v1_baseline.json"
REFERENCES = {
    "training": "reports/experiments/E1A_products_v1_seed42.json",
    "evaluation": "reports/experiments/E1A_products_v1_seed42_best_val_20261004_132526.json",
    "comparison": "reports/experiments/E1A_vs_E1_products_v1.json",
    "errors": "reports/experiments/M3-05_E1A_products_v1_seed42_20261004_132532.json",
    "camera": "reports/application/M7-01_product_camera.json",
    "live_config": "configs/live_products.yaml",
    "loading": "data/desktop/metadata/products-v1_loading.json",
    "classes": "data/desktop/metadata/classes.json",
}


def timestamp():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def file_reference(path):
    return {"path": path, "sha256": digest(ROOT / path)}


def validate_contract(manifest, training, evaluation, camera, loading, live):
    """Reject mixed checkpoints, split leakage and conflated evaluation/display rules."""
    if manifest["baseline_id"] != BASELINE_ID or manifest["status"] != "frozen":
        raise ValueError("Expected the frozen B1 baseline")
    checkpoint = manifest["selected_model"]["checkpoint"]
    if (training["status"] != "completed" or training["experiment"] != "E1-A" or
            checkpoint != training["training"]["checkpoints"]["best"] or
            checkpoint != evaluation["checkpoint"] or checkpoint != live["weights"] or
            checkpoint != camera["weights"]):
        raise ValueError("All evidence must refer to the selected best checkpoint")
    actual = training["training"]
    if (len(actual["epochs"]) != 50 or actual["optimizer_steps"] != 150 or
            actual["selected_epoch"] != 50 or manifest["training"]["epochs"] != 50 or
            manifest["training"]["optimizer_steps"] != 150 or
            manifest["training"]["configuration"] != training["configuration"] or
            manifest["training"]["selection_rule"] != training["selection_rule"]):
        raise ValueError("Training budget, configuration or selection rule changed")
    pools = {role: [s["image_id"] for s in loading["samples"] if s["split"] == role]
             for role in ("train", "val", "test")}
    flat = [i for pool in pools.values() for i in pool]
    if (len(flat) != len(set(flat)) or [len(pools[k]) for k in pools] != [20, 5, 5] or
            manifest["data"]["image_ids"] != pools or manifest["data"]["summary"] != loading["summary"] or
            training["dataset_summary"] != loading["summary"]):
        raise ValueError("Frozen pools changed or contain split leakage")
    metric = evaluation["metric_evaluation"]
    if (evaluation["status"] != "completed" or evaluation["split"] != "val" or
            metric["split"] != "val" or metric["image_ids"] != pools["val"] or
            metric["images"] != 5 or training["test_evaluated"] is not False or
            evaluation["test_evaluated"] is not False or metric["test_images_evaluated"] != 0 or
            manifest["data"]["test_model_evaluated"] is not False):
        raise ValueError("Evaluation must use only val; final test remains sealed")
    if manifest["official_evaluation"] != metric or metric["settings"]["conf"] != 0.001:
        raise ValueError("Official metric settings/results changed")
    expected = {**live["predict"], "save": False, "save_txt": False,
                "show": False, "verbose": False, "classes": None}
    if (camera["status"] != "completed" or camera["camera_confirmed_by_user"] is not True or
            camera["test_images_inferred"] != 0 or
            camera["final_runtime_configuration_check"]["passed"] is not True or
            camera["final_runtime_configuration_check"]["predict"] != expected or
            manifest["live_runtime"]["predict"] != expected or
            expected["conf"] != 0.25 or expected["quantize"] != 32):
        raise ValueError("Live display settings must match the accepted FP32 / 0.25 configuration")


def verify_frozen_files(manifest, receipt):
    if (receipt["status"] != "completed" or receipt["baseline_id"] != manifest["baseline_id"] or
            receipt["manifest"] != file_reference(MANIFEST)):
        raise ValueError("Baseline manifest differs from its acceptance receipt")
    check_guards(manifest["guarded_files"])


def build_manifest():
    source = locked_source()
    loading = verify_formal_loading()
    training = read_json(ROOT / REFERENCES["training"])
    evaluation = read_json(ROOT / REFERENCES["evaluation"])
    comparison = read_json(ROOT / REFERENCES["comparison"])
    camera = read_json(ROOT / REFERENCES["camera"])
    live = yaml.safe_load((ROOT / REFERENCES["live_config"]).read_text(encoding="utf-8"))
    if source != training["source"] or comparison["status"] != "completed":
        raise ValueError("Source lock or accepted comparison differs")
    guards = dict(training["guarded_inputs"])
    for sample in loading["samples"]:
        for kind in ("image", "label"):
            guards[sample["staged_" + kind + "_path"]] = sample[kind + "_sha256"]
    refs = {name: file_reference(path) for name, path in REFERENCES.items()}
    for ref in [*refs.values(), *training["training"]["checkpoints"].values(),
                training["training"]["results_csv"], *camera["camera_sessions"]]:
        guards[ref["path"]] = ref["sha256"]
    for path in (training["config_path"], "configs/augment_products_e1a.yaml",
                 training["run_directory"] + "/args.yaml",
                 training["run_directory"] + "/preflight_used.json", "configs/source-lock.json"):
        guards[path] = digest(ROOT / path)
    check_guards(guards)
    from app.product_runner import implementation_hashes
    from scripts.train import environment_fingerprint
    manifest = {
        "schema_version": 1, "baseline_id": BASELINE_ID, "status": "frozen", "created_at": timestamp(),
        "purpose": "first_version_product_model_reference_for_deployment_and_structure_control",
        "project_git_reference": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "source": source, "references": refs,
        "selected_model": {"experiment": "E1-A", "checkpoint": training["training"]["checkpoints"]["best"],
                           "selected_epoch": 50, "task": "segment", "nc": 3, "mask_coefficients": 32,
                           "names": {str(c["id"]): c["name"] for c in catalog()["classes"]}},
        "data": {"version": "products-v1", "summary": loading["summary"], "image_ids": {
            role: [s["image_id"] for s in loading["samples"] if s["split"] == role]
            for role in ("train", "val", "test")}, "test_model_evaluated": False,
            "test_rule": "Sealed for M8 final evaluation; not for tuning, model selection or data expansion decisions."},
        "training": {"config": file_reference(training["config_path"]),
                     "augmentation": file_reference("configs/augment_products_e1a.yaml"),
                     "configuration": training["configuration"], "epochs": 50, "optimizer_steps": 150,
                     "selection_rule": training["selection_rule"]},
        "official_evaluation": evaluation["metric_evaluation"],
        "display_diagnostic_val_conf025": comparison["threshold_diagnostics"]["E1-A"]["val"]["0.25"],
        "live_runtime": {"config": refs["live_config"], "predict": camera["final_runtime_configuration_check"]["predict"],
                         "camera": live["camera"], "feedback_role": "Qualitative user observation, not measured accuracy.",
                         "user_feedback_conf025": "误检减少，三类基本能检出"},
        "experiment_table": [{"experiment": label, "checkpoint": read_json(ROOT / path)["checkpoint"],
                              "epochs": 50, "optimizer_steps": 150,
                              "degrees": degrees, "scale": scale,
                              "metrics": read_json(ROOT / path)["metric_evaluation"]["metrics"]}
                             for label, path, degrees, scale in (
                                 ("E1", "reports/experiments/E1_products_v1_seed42_best_val_20261004_123032.json", 5, .2),
                                 ("E1-A", REFERENCES["evaluation"], 45, .4))],
        "selection_reason": "Same-data/budget validation improved; 0.25 camera feedback supports the limited first demo.",
        "limitations": ["Only 20 train / 5 repeatedly inspected val images and one physical item per class.",
                        "One seed; mean improvement is not a statistically established generalization gain.",
                        "Cup AP decreased slightly versus E1; val 0.25 still misses two of eight targets.",
                        "Live false positives and weaker cup performance remain; no live ground truth or long stability test.",
                        "No final-test model inference; ONNX product parity/export not performed by this freeze."],
        "future_structure_control": {
            "status": "planned", "reference": BASELINE_ID,
            "rule": "Change only the intended structure. Keep original COCO initialization, train/val pools, E1-A augmentation, seed42, 50 epochs/150 updates, optimizer and selection/evaluation rules identical.",
            "data_or_budget_change": "Requires a separately named experiment and new baseline; never overwrite B1."},
        "future_deployment": {"status": "planned", "checkpoint": training["training"]["checkpoints"]["best"],
                              "profile": "batch1 / static640x640 / FP32",
                              "parity_rule": "Use identical square letterbox, normalization and postprocessing for PyTorch/ORT. Rectangular live inference is a separate profile."},
        "reference_implementation_sha256": {**implementation_hashes(), **{
            p: digest(ROOT / p) for p in ("app/live_camera.py", "scripts/freeze_product_baseline.py", "tests/test_product_baseline.py")}},
        "reference_environment_fingerprint": environment_fingerprint(),
        "drift_policy": "Artifact/config/data/evidence changes fail verification. Code/package differences are reported for review; unrelated project commits do not invalidate the saved model.",
        "guarded_files": guards,
    }
    validate_contract(manifest, training, evaluation, camera, loading, live)
    return manifest


def check_baseline():
    manifest, receipt = read_json(ROOT / MANIFEST), read_json(ROOT / RECEIPT)
    verify_frozen_files(manifest, receipt)
    source, loading = locked_source(), verify_formal_loading()
    if source != manifest["source"]:
        raise ValueError("Frozen source changed")
    validate_contract(manifest, read_json(ROOT / REFERENCES["training"]), read_json(ROOT / REFERENCES["evaluation"]),
                      read_json(ROOT / REFERENCES["camera"]), loading,
                      yaml.safe_load((ROOT / REFERENCES["live_config"]).read_text(encoding="utf-8")))
    from scripts.train import environment_fingerprint
    code_drift = [p for p, sha in manifest["reference_implementation_sha256"].items() if digest(ROOT / p) != sha]
    result = {"status": "passed", "baseline_id": BASELINE_ID, "guarded_files": len(manifest["guarded_files"]),
              "reference_code_drift": code_drift,
              "reference_environment_drift": environment_fingerprint() != manifest["reference_environment_fingerprint"],
              "model_inference": False, "files_written": False}
    print(result)
    return result


def freeze_baseline():
    if (ROOT / MANIFEST).exists() or (ROOT / RECEIPT).exists():
        raise FileExistsError("B1 already has artifacts; use --check, never overwrite the freeze")
    manifest = build_manifest()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py")
    tests = unittest.TextTestRunner(verbosity=1).run(suite)
    if not tests.wasSuccessful():
        raise ValueError("Engineering tests failed; baseline was not frozen")
    import torch
    from ultralytics import YOLO
    from app.product_experiment import verify_product_model
    from scripts.train import environment_fingerprint, output_shapes
    if not torch.cuda.is_available():
        raise ValueError("Freeze reload check requires the configured CUDA device")
    model = YOLO(str(ROOT / manifest["selected_model"]["checkpoint"]["path"]))
    verify_product_model(model.model)
    network = model.model.to("cuda:0").float().eval()
    with torch.inference_mode():
        shapes = output_shapes(network(torch.zeros(1, 3, 640, 640, device="cuda:0")))
    if [1, 39, 8400] not in shapes or [1, 32, 160, 160] not in shapes:
        raise ValueError("Reloaded three-class segmentation output differs")
    check_guards(manifest["guarded_files"])
    if environment_fingerprint() != manifest["reference_environment_fingerprint"]:
        raise ValueError("Package versions changed during acceptance")
    write_json(ROOT / MANIFEST, manifest)
    receipt = {"schema_version": 1, "task": "M3-06", "status": "completed", "completed_at": timestamp(),
               "baseline_id": BASELINE_ID, "manifest": file_reference(MANIFEST),
               "guarded_files_checked": len(manifest["guarded_files"]), "guarded_files_unchanged": True,
               "engineering_tests": {"passed": tests.testsRun, "failures": len(tests.failures), "errors": len(tests.errors)},
               "gpu_reload": {"passed": True, "input_shape": [1, 3, 640, 640], "raw_output_shapes": shapes,
                              "device": "cuda:0", "dtype": str(next(network.parameters()).dtype),
                              "gpu": torch.cuda.get_device_name(0), "input_role": "Synthetic zeros; not a quality evaluation."},
               "new_photos": 0, "new_training_runs": 0, "optimizer_updates": 0,
               "final_test_images_inferred": 0, "new_model_copies": 0,
               "progress": {"M3": "5/6", "M7": "1/6", "main": "24/54", "complete_modules": "3/9"},
               "deferred": ["M3-04 optional E2", "product ONNX deployment", "M8 final test"]}
    write_json(ROOT / RECEIPT, receipt)
    check_baseline()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Read-only checks; no camera, inference or training")
    args = parser.parse_args()
    check_baseline() if args.check else freeze_baseline()
