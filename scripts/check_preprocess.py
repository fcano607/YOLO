"""M5-02: verify shared input against the locked official implementation, no inference."""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from app.product_data import digest, image_read, read_json, write_json
from app.product_training import check_guards
from deploy.preprocess import preprocess_bgr, validate_preprocessing_contract
from scripts.export_onnx import check_export, now, reference


REPORT = ROOT / "reports/deployment/M5-02_B1_preprocess.json"
IMPLEMENTATION = ("deploy/preprocess.py", "scripts/check_preprocess.py", "tests/test_preprocess.py")
SYNTHETIC_SHAPES = ((720, 1280), (1280, 720), (640, 640), (481, 640), (640, 481), (333, 1001),
                    (1001, 333), (1, 1), (17, 29), (29, 17), (479, 641), (641, 479))


def historical_guards(export_report):
    return {**export_report["guarded_files"], **export_report["implementation_sha256"],
            **{v["path"]: v["sha256"] for v in export_report["artifacts"].values()},
            "reports/deployment/M5-01_B1_onnx_export.json": digest(ROOT / "reports/deployment/M5-01_B1_onnx_export.json"),
            "logs/deployment/M5-01_B1_onnx_export.log": digest(ROOT / "logs/deployment/M5-01_B1_onnx_export.log")}


def verify_case(image, case_id, predictor, letterbox):
    original = image.copy()
    tensor, geometry = preprocess_bgr(image)
    official = predictor.preprocess([image]).numpy()
    params = letterbox.get_params({"img": image})
    expected_geometry = {"original_shape_hw": list(params["orig_shape"]),
                        "input_shape_hw": list(params["new_shape"]),
                        "resized_shape_hw": list(params["new_unpad"][::-1]),
                        "ratio_xy": list(params["ratio"]),
                        "padding_ltrb": [params[k] for k in ("left", "top", "right", "bottom")]}
    if (not np.array_equal(tensor, official) or geometry.to_dict() != expected_geometry or
            tensor.shape != (1, 3, 640, 640) or tensor.dtype != np.float32 or
            not tensor.flags.c_contiguous or not np.isfinite(tensor).all() or
            tensor.min() < 0 or tensor.max() > 1 or not np.array_equal(image, original)):
        raise ValueError("Input/geometry mismatch or source mutation: " + case_id)
    return {"case_id": case_id, "geometry": geometry.to_dict(), "input_shape": list(tensor.shape),
            "dtype": str(tensor.dtype), "contiguous": True, "pixel_equal": True,
            "geometry_equal": True, "max_abs": float(np.abs(tensor - official).max()),
            "input_sha256": hashlib.sha256(tensor.tobytes()).hexdigest(), "source_unchanged": True}, tensor


def independent_import_check():
    code = """import json, sys
import numpy as np
from deploy.preprocess import preprocess_bgr
x, g = preprocess_bgr(np.zeros((17, 29, 3), dtype=np.uint8))
print(json.dumps({'torch_imported': 'torch' in sys.modules, 'ultralytics_imported': 'ultralytics' in sys.modules,
                  'shape': list(x.shape), 'dtype': str(x.dtype), 'contiguous': x.flags.c_contiguous}))
"""
    process = subprocess.run([sys.executable, "-B", "-X", "utf8", "-c", code], cwd=ROOT,
                             check=True, capture_output=True, text=True, encoding="utf-8")
    result = json.loads(process.stdout)
    if result != {"torch_imported": False, "ultralytics_imported": False,
                  "shape": [1, 3, 640, 640], "dtype": "float32", "contiguous": True}:
        raise ValueError("Standalone preprocessing unexpectedly depends on a model framework")
    return result


def verify_inputs(export_report):
    # Official code is used only as a reference, never inside deploy.preprocess.
    import torch
    from ultralytics.data.augment import LetterBox
    from ultralytics.engine.predictor import BasePredictor

    metadata = read_json(ROOT / "artifacts/B1/best_fp32.metadata.json")
    validate_preprocessing_contract(metadata)
    predictor = object.__new__(BasePredictor)
    predictor.imgsz, predictor.device = [640, 640], torch.device("cpu")
    predictor.args = SimpleNamespace(rect=False)
    predictor.model = SimpleNamespace(format="pt", stride=32, fp16=False, dynamic=False)
    letterbox = LetterBox(new_shape=(640, 640), auto=False, scale_fill=False,
                         scaleup=True, center=True, stride=32, padding_value=114)
    loading = read_json(ROOT / "data/desktop/metadata/products-v1_loading.json")
    samples = [s for s in loading["samples"] if s["split"] == "val"]
    if len(samples) != 5:
        raise ValueError("Expected five frozen validation sources")
    records, reference_matches = [], []
    saved_cases = {c["image_id"]: c for c in metadata["smoke_cases"]}
    with np.load(ROOT / "artifacts/B1/export_reference.npz", allow_pickle=False) as saved:
        for sample in samples:
            path = ROOT / sample["image_path"]
            if digest(path) != sample["image_sha256"]:
                raise ValueError("Frozen validation source changed")
            record, tensor = verify_case(image_read(path), sample["image_id"], predictor, letterbox)
            record.update(source=reference(path), split="val")
            records.append(record)
            if sample["image_id"] in saved_cases:
                case = saved_cases[sample["image_id"]]
                if (record["input_sha256"] != case["input_sha256"] or
                        not np.array_equal(tensor, saved[case["array_prefix"] + "_images"])):
                    raise ValueError("Shared input differs from the accepted M5-01 saved tensor")
                reference_matches.append({"image_id": sample["image_id"], "array_prefix": case["array_prefix"],
                                          "pixel_equal": True, "input_sha256": record["input_sha256"]})
    if len(reference_matches) != len(export_report["cases"]) or len(reference_matches) != 3:
        raise ValueError("Missing saved export input comparison")
    for index, (height, width) in enumerate(SYNTHETIC_SHAPES):
        y, x = np.indices((height, width))
        image = np.stack(((x * 3 + y) % 256, (x + y * 7 + 43) % 256,
                          (x * 11 + y * 5 + 137) % 256), axis=-1).astype(np.uint8)
        # Also exercise a noncontiguous, read-only input in the official comparison.
        if index == len(SYNTHETIC_SHAPES) - 1:
            image = image[:, ::-1]
            image.setflags(write=False)
        record, _ = verify_case(image, "synthetic_%02d" % index, predictor, letterbox)
        record["source"] = "deterministic in-memory BGR pattern; not training/evaluation data"
        records.append(record)
    return {"official_comparisons": records, "saved_export_input_matches": reference_matches,
            "independent_import": independent_import_check(),
            "summary": {"real_val_images": 5, "synthetic_cases": len(SYNTHETIC_SHAPES),
                        "all_pixel_equal": True, "all_geometry_equal": True, "max_abs": 0.,
                        "saved_export_inputs_equal": 3, "model_inference": False}}


def run(check_only):
    if not check_only and REPORT.exists():
        raise FileExistsError("M5-02 receipt exists; use --check rather than overwrite it")
    saved = read_json(REPORT) if check_only else None
    if saved:
        if saved["status"] != "completed" or saved["task"] != "M5-02":
            raise ValueError("Expected the accepted M5-02 receipt")
        check_guards(saved["guarded_files"])
        check_guards(saved["implementation_sha256"])
    with contextlib.redirect_stdout(sys.stderr):
        export_report = check_export(ROOT / "configs/export_products.yaml")
    guards = historical_guards(export_report)
    check_guards(guards)
    timestamps = {p: (ROOT / p).stat().st_mtime_ns for p in guards}
    checks = verify_inputs(export_report)
    check_guards(guards)
    if timestamps != {p: (ROOT / p).stat().st_mtime_ns for p in guards}:
        raise ValueError("Historical input modification timestamps changed")
    if check_only:
        if saved["guarded_files"] != guards or saved["checks"] != checks:
            raise ValueError("Saved preprocessing acceptance differs from current checks")
    else:
        report = {"schema_version": 1, "task": "M5-02", "status": "completed", "completed_at": now(),
                  "baseline_id": export_report["baseline_id"],
                  "export_receipt": reference(ROOT / "reports/deployment/M5-01_B1_onnx_export.json"),
                  "model_metadata": reference(ROOT / "artifacts/B1/best_fp32.metadata.json"),
                  "reference": {"source": export_report["source"], "model_loaded": False,
                      "methods": ["BasePredictor.preprocess: rect=False, fp16=False, CPU",
                                  "LetterBox.get_params: static centered 640x640"],
                      "code": [reference(ROOT / "third_party/ultralytics/ultralytics" / path) for path in
                               ("data/augment.py", "engine/predictor.py")]},
                  "implementation_sha256": {p: digest(ROOT / p) for p in IMPLEMENTATION},
                  "guarded_files": guards, "guarded_files_count": len(guards),
                  "historical_sha256_and_mtime_unchanged": True, "checks": checks,
                  "scope": {"new_photos": 0, "label_changes": 0, "training": 0,
                      "model_inference": 0, "sealed_test_inference": 0, "new_saved_arrays": 0,
                      "postprocessing_implemented": False, "full_ort_runner_implemented": False},
                  "progress": {"M5": "2/6", "main": "26/54", "complete_modules": "3/9",
                               "next": "M5-03 independent boxes/NMS/masks"}}
        write_json(REPORT, report)
    print(json.dumps({"status": "passed", "task": "M5-02", "checks": checks["summary"],
                      "guarded_files": len(guards), "files_written": not check_only}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Recheck the accepted receipt without writing files")
    args = parser.parse_args()
    run(args.check)


if __name__ == "__main__":
    main()
