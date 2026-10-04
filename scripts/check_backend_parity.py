"""M5-05: fixed non-test inputs, isolated raw runtimes, shared/official postprocessing."""
import argparse
from collections import Counter
import contextlib
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np

from app.product_data import image_read, write_json
from deploy.base_backend import create_backend, file_sha256, load_bundle, read_json
from deploy.onnx_contract import compare_raw
from deploy.parity import TOLERANCE, assert_binding, compare_instances, score_ties
from deploy.postprocess import ProductInstances, postprocess_b1
from deploy.preprocess import preprocess_bgr

MANIFEST = ROOT / "configs/parity_products_B1.json"
REPORT = ROOT / "reports/deployment/M5-05_B1_backend_parity.json"
PREVIOUS = ROOT / "reports/deployment/M5-04_B1_file_inference.json"
LOADING = ROOT / "data/desktop/metadata/products-v1_loading.json"
IMPLEMENTATION = ("configs/parity_products_B1.json", "deploy/parity.py",
                  "scripts/check_backend_parity.py", "tests/test_backend_parity.py")
ROUTES = ("torch_cuda", "ort_cpu", "ort_cuda")


def ref(path):
    path = Path(path).resolve()
    return {"path": path.relative_to(ROOT.resolve()).as_posix(), "sha256": file_sha256(path)}


def tensor_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def transform_frame(frame, transform):
    if transform == {"op": "identity"}:
        return frame
    if transform == {"op": "rotate90_clockwise"}:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if transform.get("op") == "resize" and set(transform) == {"op", "shape_hw"}:
        h, w = transform["shape_hw"]
        if not isinstance(h, int) or not isinstance(w, int) or min(h, w) <= 0 or max(h, w) > 2000:
            raise ValueError("Invalid fixed variant dimensions")
        return cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
    raise ValueError("Unrecognized fixed input transform")


def allowed_sources():
    loading = read_json(LOADING)
    return {s["image_id"]: s for s in loading["samples"] if s["split"] in ("train", "val")}


def materialize(case, sources):
    source = case["source"]
    sample = sources.get(source["image_id"])
    expected = ({"image_id": sample["image_id"], "path": sample["image_path"],
                 "sha256": sample["image_sha256"], "split": sample["split"]} if sample else None)
    if source != expected or source.get("split") not in ("train", "val"):
        raise ValueError("Only frozen non-test sources are allowed")
    path = ROOT / source["path"]
    if file_sha256(path) != source["sha256"]:
        raise ValueError("Frozen source image changed")
    frame = transform_frame(image_read(path), case["transform"])
    images, geometry = preprocess_bgr(frame)
    if "input_sha256" in case and (case["input_sha256"] != tensor_sha(images) or
            case["frame_sha256"] != tensor_sha(frame) or case["shape_hw"] != list(frame.shape[:2])):
        raise ValueError("Fixed variant / input tensor changed")
    return frame, images, geometry


def build_manifest():
    sources = allowed_sources()
    if Counter(s["split"] for s in sources.values()) != {"train": 20, "val": 5}:
        raise ValueError("Expected the frozen 20/5 development pool")
    specs = [(s, {"op": "identity"}) for s in sources.values()]
    val = [s for s in sources.values() if s["split"] == "val"]
    train = next(s for s in sources.values() if s["split"] == "train")
    specs += [(val[0], {"op": "resize", "shape_hw": [361, 641]}),
              (val[3], {"op": "rotate90_clockwise"}),
              (train, {"op": "resize", "shape_hw": [321, 321]}),
              (val[3], {"op": "resize", "shape_hw": [1001, 333]}),
              (val[4], {"op": "resize", "shape_hw": [17, 29]})]
    cases = []
    for index, (sample, transform) in enumerate(specs, 1):
        case = {"case_id": "c%02d" % index,
                "source": {"image_id": sample["image_id"], "path": sample["image_path"],
                           "sha256": sample["image_sha256"], "split": sample["split"]},
                "transform": transform, "negative_source": not sample["instances"]}
        frame, images, _ = materialize(case, sources)
        case.update(shape_hw=list(frame.shape[:2]), frame_sha256=tensor_sha(frame), input_sha256=tensor_sha(images))
        cases.append(case)
    return {"schema_version": 1, "purpose": "fixed_engineering_parity_not_independent_quality",
            "baseline_id": load_bundle().baseline_id, "source_loading": ref(LOADING),
            "cases": cases, "summary": {"inputs": 30, "unique_original_images": 25,
                "original_train": 20, "original_val": 5, "derived_variants": 5, "sealed_test": 0},
            "postprocess": load_bundle().config["postprocess"], "tolerance": TOLERANCE}


def validate_manifest(manifest):
    # Compare the entire deterministic recipe, not just input count or hashes.
    if manifest != build_manifest():
        raise ValueError("Fixed manifest differs from the accepted non-test input recipe")
    return manifest


def official_instances(raw, proto, geometry, frame, options):
    import torch
    from types import SimpleNamespace
    from ultralytics.utils import nms, ops
    from ultralytics.models.yolo.segment.predict import SegmentationPredictor
    predictions, indices = nms.non_max_suppression(torch.from_numpy(raw.copy()),
        conf_thres=options["conf"], iou_thres=options["iou"], max_det=options["max_det"],
        multi_label=options["multi_label"], nc=3, agnostic=False, return_idxs=True)
    prediction = predictions[0]
    boxes = ops.scale_boxes((640, 640), prediction[:, :4].clone(), geometry.original_shape_hw)
    masks = ops.process_mask_native(torch.from_numpy(proto[0].copy()), prediction[:, 6:], boxes,
                                    geometry.original_shape_hw).numpy()
    keep = masks.any(axis=(1, 2))
    rows = prediction.numpy()[keep]
    result = ProductInstances(boxes.numpy()[keep], rows[:, 4], rows[:, 5].astype(np.int64),
        masks[keep].astype(np.uint8), rows[:, 6:], indices[0].reshape(-1).numpy().astype(np.int64)[keep], {})
    # Also include the official result constructor, which applies the empty-mask filter.
    predictor = object.__new__(SegmentationPredictor)
    predictor.args = SimpleNamespace(retina_masks=True)
    predictor.model = SimpleNamespace(names={0: "sam_whole_milk", 1: "yili_shuhua", 2: "luckin_cup"})
    constructed = predictor.construct_result(prediction.clone(), torch.empty((1, 3, 640, 640)),
                                             frame, "fixed_parity_input", torch.from_numpy(proto[0].copy()))
    constructed_masks = (constructed.masks.data.numpy() if constructed.masks is not None else
                         np.empty((0, *geometry.original_shape_hw), np.uint8))
    if (not np.array_equal(constructed.boxes.data.numpy(), np.concatenate((result.boxes_xyxy, rows[:, 4:6]), axis=1))
            or not np.array_equal(constructed_masks, result.masks)):
        raise ValueError("Official result-construction mismatch")
    assert_binding(result, raw)
    return result


def require_comparison(record, case_id):
    if not record["passed"]:
        raise ValueError("Parity failure " + case_id + ": " + json.dumps(record))


def worker(route, scratch, profile_prefix=None):
    manifest = validate_manifest(read_json(MANIFEST))
    sources = allowed_sources()
    bundle = load_bundle()
    backend = create_backend(bundle, "torch" if route == "torch_cuda" else "onnx",
                             "cpu" if route == "ort_cpu" else "cuda:0", profile_prefix)
    records = []
    try:
        identity = backend.describe()
        for case in manifest["cases"]:
            frame, images, geometry = materialize(case, sources)
            raw = backend.run_raw(images)
            actual = postprocess_b1(*raw, geometry, **manifest["postprocess"])
            assert_binding(actual, raw[0])
            record = {"case_id": case["case_id"], "input_sha256": tensor_sha(images),
                      "instances": len(actual.scores), "class_ids": actual.class_ids.tolist(),
                      "score_ties_above_conf": score_ties(raw[0]), "counts": actual.counts}
            if route != "torch_cuda":
                with np.load(scratch / (case["case_id"] + "_torch_cuda.npz"), allow_pickle=False) as saved:
                    if str(saved["input_sha256"]) != tensor_sha(images):
                        raise ValueError("Backends did not receive the same normalized tensor")
                    reference_raw = (saved["output0"], saved["output1"])
                    record["raw_comparison"] = compare_raw(reference_raw, raw, TOLERANCE["raw"])
                    if not all(v["allclose"] for v in record["raw_comparison"]):
                        raise ValueError("Raw output exceeds fixed tolerance: " + case["case_id"])
                    reference_instances = postprocess_b1(*reference_raw, geometry, **manifest["postprocess"])
                    record["shared_postprocess_comparison"] = compare_instances(actual, reference_instances)
                    require_comparison(record["shared_postprocess_comparison"], case["case_id"])
            np.savez(scratch / (case["case_id"] + "_" + route + ".npz"), output0=raw[0], output1=raw[1],
                     input_sha256=np.array(tensor_sha(images)))
            records.append(record)
    finally:
        backend.close()
    result = {"route": route, "backend": identity, "cases": records, "backend_closed": backend.closed,
              "torch_imported": "torch" in sys.modules, "ultralytics_imported": "ultralytics" in sys.modules,
              "profile": ref(backend.profile_path) if getattr(backend, "profile_path", None) else None}
    if route.startswith("ort") and (result["torch_imported"] or result["ultralytics_imported"]):
        raise ValueError("Independent ORT worker imported a training framework")
    print("__PARITY_WORKER__=" + json.dumps(result, ensure_ascii=False))


def official_worker(scratch):
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    manifest = validate_manifest(read_json(MANIFEST))
    sources = allowed_sources()
    records = []
    for case in manifest["cases"]:
        frame, _, geometry = materialize(case, sources)
        for route in ROUTES:
            with np.load(scratch / (case["case_id"] + "_" + route + ".npz"), allow_pickle=False) as saved:
                raw, proto = saved["output0"], saved["output1"]
            actual = postprocess_b1(raw, proto, geometry, **manifest["postprocess"])
            assert_binding(actual, raw)
            reference = official_instances(raw, proto, geometry, frame, manifest["postprocess"])
            comparison = compare_instances(actual, reference)
            require_comparison(comparison, case["case_id"] + ":" + route + ":official")
            records.append({"case_id": case["case_id"], "route": route, "comparison": comparison,
                            "official_result_constructor_checked": True})
    print("__PARITY_OFFICIAL__=" + json.dumps({"cases": records, "model_inference": False,
                                              "reference_device": "cpu"}, ensure_ascii=False))


def child(command, prefix):
    result = subprocess.run([sys.executable, "-B", "-X", "utf8", *command], cwd=ROOT,
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode:
        raise RuntimeError("Worker failed:\n" + result.stdout[-2000:] + result.stderr[-6000:])
    lines = [s[len(prefix):] for s in result.stdout.splitlines() if s.startswith(prefix)]
    if len(lines) != 1:
        raise ValueError("Expected exactly one worker receipt")
    return json.loads(lines[0])


def check_hashes(guards):
    for path, sha in guards.items():
        if file_sha256(ROOT / path) != sha:
            raise ValueError("Protected file changed: " + path)


def summarize(runtime, official):
    cross = [c["shared_postprocess_comparison"] for r in ("ort_cpu", "ort_cuda") for c in runtime[r]["cases"]]
    own = [c["comparison"] for c in official["cases"]]
    raws = [v for r in ("ort_cpu", "ort_cuda") for c in runtime[r]["cases"] for v in c["raw_comparison"]]
    def summary(records):
        measured = [r["min_mask_iou"] for r in records if r["min_mask_iou"] is not None]
        return {"comparisons": len(records), "all_passed": all(r["passed"] for r in records),
                "matched_instances": sum(r["matched"] for r in records),
                "empty_pairs": sum(r["both_empty"] for r in records),
                "unmatched_instances": sum(len(r["unmatched_actual"]) + len(r["unmatched_reference"]) for r in records),
                "max_box_abs": max(r["max_box_abs"] for r in records),
                "max_score_abs": max(r["max_score_abs"] for r in records),
                "min_mask_iou": min(measured) if measured else None,
                "mask_disagreement_pixels": sum(r["mask_disagreement_pixels"] for r in records),
                "candidate_binding_differences": sum(not r["candidate_indices_equal_after_matching"] for r in records)}
    return {"inputs": 30, "real_originals": 25, "derived_variants": 5, "runtime_forward_passes": 90,
            "same_normalized_inputs": True, "raw_comparisons": len(raws),
            "all_raw_within_tolerance": all(r["allclose"] for r in raws),
            "raw_output0_max_abs": max(r["max_abs"] for r in raws if r["name"] == "output0"),
            "raw_output1_max_abs": max(r["max_abs"] for r in raws if r["name"] == "output1"),
            "backend_shared_postprocess": summary(cross), "independent_vs_official_same_raw": summary(own),
            "score_tie_cases": {r: [c["case_id"] for c in runtime[r]["cases"] if c["score_ties_above_conf"]] for r in ROUTES}}


def accept():
    from scripts.check_product_deployment import check_saved as check_previous
    if REPORT.exists():
        raise FileExistsError("M5-05 acceptance exists; use --check")
    manifest = validate_manifest(read_json(MANIFEST))
    with contextlib.redirect_stdout(sys.stderr):
        check_previous()
    previous = read_json(PREVIOUS)
    guards = {**previous["guarded_files"], **previous["implementation_sha256"],
              **{v["path"]: v["sha256"] for v in previous["artifacts"].values()},
              PREVIOUS.relative_to(ROOT).as_posix(): file_sha256(PREVIOUS)}
    check_hashes(guards)
    timestamps = {p: (ROOT / p).stat().st_mtime_ns for p in guards}
    prefix = ROOT / "artifacts/B1" / ("M5-05_cuda_profile_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    temp_parent = ROOT / "tmp"
    temp_parent.mkdir(exist_ok=True)
    runtime = {}
    with tempfile.TemporaryDirectory(prefix="M5-05_", dir=str(temp_parent)) as scratch_path:
        scratch = Path(scratch_path).resolve()
        if not scratch.is_relative_to(temp_parent.resolve()):
            raise ValueError("Temporary references escaped the intended workspace directory")
        for route in ROUTES:
            print("Checking " + route + " on 30 fixed inputs", flush=True)
            command = ["scripts/check_backend_parity.py", "--worker", route, "--scratch", str(scratch)]
            if route == "ort_cuda":
                command += ["--profile-prefix", str(prefix)]
            runtime[route] = child(command, "__PARITY_WORKER__=")
        print("Checking official postprocessing on all 90 saved temporary raw outputs", flush=True)
        official = child(["scripts/check_backend_parity.py", "--official-worker", "--scratch", str(scratch)],
                         "__PARITY_OFFICIAL__=")
    temporary_removed = not scratch.exists()
    events = Counter(v.get("args", {}).get("provider") for v in read_json(ROOT / runtime["ort_cuda"]["profile"]["path"])
                     if v.get("cat") == "Node" and v.get("args", {}).get("provider"))
    if not events["CUDAExecutionProvider"] or events["CPUExecutionProvider"]:
        raise ValueError("Missing actual GPU execution or unexpected CPU node fallback")
    check_hashes(guards)
    if timestamps != {p: (ROOT / p).stat().st_mtime_ns for p in guards}:
        raise ValueError("Historical file modification times changed")
    report = {"schema_version": 1, "task": "M5-05", "status": "completed",
              "completed_at": datetime.now().astimezone().isoformat(), "baseline_id": manifest["baseline_id"],
              "previous_receipt": ref(PREVIOUS), "manifest": ref(MANIFEST), "tolerance": TOLERANCE,
              "runtime_workers": runtime, "official_postprocess": official, "summary": summarize(runtime, official),
              "actual_cuda_node_events": dict(events), "guarded_files": guards, "guarded_files_count": len(guards),
              "historical_sha256_and_mtime_unchanged": True,
              "implementation_sha256": {p: file_sha256(ROOT / p) for p in IMPLEMENTATION},
              "artifacts": {"cuda_profile": runtime["ort_cuda"]["profile"]},
              "scope": {"new_photos": 0, "label_changes": 0, "training": 0, "new_model_exports": 0,
                        "sealed_test_inference": 0, "formal_mAP": False, "formal_benchmark": False,
                        "temporary_raw_references_removed": temporary_removed, "persistent_image_copies": 0,
                        "camera_switched_to_onnx": False},
              "progress": {"M5": "5/6", "main": "29/54", "complete_modules": "3/9", "next": "M5-06 fixed val5 deployment quality"}}
    write_json(REPORT, report)
    print(json.dumps({"status": "passed", "task": "M5-05", "summary": report["summary"],
                      "cuda_node_events": dict(events), "guarded_files": len(guards),
                      "temporary_raw_references_removed": temporary_removed}, ensure_ascii=False))


def check_saved():
    from scripts.check_product_deployment import check_saved as check_previous
    report = read_json(REPORT)
    if report["status"] != "completed" or report["task"] != "M5-05":
        raise ValueError("Expected completed M5-05 acceptance")
    check_hashes(report["guarded_files"])
    check_hashes(report["implementation_sha256"])
    check_hashes({v["path"]: v["sha256"] for v in report["artifacts"].values()})
    validate_manifest(read_json(MANIFEST))
    with contextlib.redirect_stdout(sys.stderr):
        check_previous()
    if summarize(report["runtime_workers"], report["official_postprocess"]) != report["summary"]:
        raise ValueError("Summary does not agree with saved per-input checks")
    print(json.dumps({"status": "passed", "task": "M5-05", "inputs": 30,
                      "guarded_files": len(report["guarded_files"]), "files_written": False, "model_inference": False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--init-manifest", action="store_true")
    mode.add_argument("--worker", choices=ROUTES)
    mode.add_argument("--official-worker", action="store_true")
    parser.add_argument("--scratch")
    parser.add_argument("--profile-prefix")
    args = parser.parse_args()
    if args.init_manifest:
        if MANIFEST.exists():
            raise FileExistsError("Do not overwrite fixed manifest")
        write_json(MANIFEST, build_manifest())
        print("Fixed 30-input manifest created: " + MANIFEST.name)
    elif args.worker or args.official_worker:
        scratch = Path(args.scratch).resolve()
        if not scratch.is_relative_to((ROOT / "tmp").resolve()) or not scratch.is_dir():
            raise ValueError("Worker references must be in the intended temporary directory")
        if args.worker:
            worker(args.worker, scratch, args.profile_prefix)
        else:
            official_worker(scratch)
    elif args.check:
        check_saved()
    else:
        accept()


if __name__ == "__main__":
    main()
