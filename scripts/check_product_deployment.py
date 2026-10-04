"""M5-04 file/backend acceptance; isolated ORT workers, saved-reference checks, no final test."""
import argparse
from collections import Counter
import contextlib
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from app.product_data import image_read
from deploy.base_backend import create_backend, file_sha256, load_bundle, read_json
from deploy.onnx_contract import compare_raw
from deploy.postprocess import postprocess_b1
from deploy.preprocess import preprocess_bgr


REPORT = ROOT / "reports/deployment/M5-04_B1_file_inference.json"
PREVIOUS = ROOT / "reports/deployment/M5-03_B1_postprocess.json"
DEMO = ROOT / "demo/deployment/B1"
IMPLEMENTATION = ("configs/infer_products.yaml", "deploy/results.py", "deploy/base_backend.py",
                  "deploy/onnx_backend.py", "deploy/torch_backend.py", "app/infer_products.py",
                  "scripts/check_product_deployment.py", "tests/test_product_deployment.py")
TOLERANCE = {"raw": {"atol": .001, "rtol": .0001}, "max_original_box_abs": .01,
             "max_score_abs": .0001, "min_mask_iou": .999}


def asset_ref(path):
    path = Path(path).resolve()
    return {"path": path.relative_to(ROOT.resolve()).as_posix(), "sha256": file_sha256(path)}


def check_hashes(guards):
    for path, sha in guards.items():
        if file_sha256(ROOT / path) != sha:
            raise ValueError("Protected file changed: " + path)


def compare_instances(actual, reference):
    if (not np.array_equal(actual.candidate_indices, reference.candidate_indices) or
            not np.array_equal(actual.class_ids, reference.class_ids) or actual.masks.shape != reference.masks.shape):
        raise ValueError("Full pipeline candidate / class / shape mismatch")
    box_error = float(np.abs(actual.boxes_xyxy - reference.boxes_xyxy).max()) if len(actual.scores) else 0.
    score_error = float(np.abs(actual.scores - reference.scores).max()) if len(actual.scores) else 0.
    intersection = (actual.masks.astype(bool) & reference.masks.astype(bool)).sum(axis=(1, 2))
    union = (actual.masks.astype(bool) | reference.masks.astype(bool)).sum(axis=(1, 2))
    ious = np.divide(intersection, union, out=np.ones(len(union), np.float64), where=union > 0)
    minimum = float(ious.min()) if len(ious) else None
    if (box_error > TOLERANCE["max_original_box_abs"] or score_error > TOLERANCE["max_score_abs"] or
            (minimum is not None and minimum < TOLERANCE["min_mask_iou"])):
        raise ValueError("Full pipeline exceeds box / score / mask parity tolerances")
    return {"candidate_indices_equal": True, "class_ids_equal": True, "max_original_box_abs": box_error,
            "max_score_abs": score_error, "min_mask_iou": minimum,
            "mask_disagreement_pixels": int(np.count_nonzero(actual.masks != reference.masks)), "passed": True}


def runtime_worker(kind, device, profile_prefix):
    bundle = load_bundle()
    backend = create_backend(bundle, kind, device, profile_prefix)
    cases = []
    try:
        identity = backend.describe()
        with np.load(ROOT / "artifacts/B1/export_reference.npz", allow_pickle=False) as saved:
            for case in bundle.metadata["smoke_cases"]:
                if case["split"] != "val":
                    raise ValueError("Runtime smoke checks use only accepted validation sources")
                source = ROOT / case["source"]["path"]
                if file_sha256(source) != case["source"]["sha256"]:
                    raise ValueError("Frozen validation image changed")
                frame = image_read(source)
                images, geometry = preprocess_bgr(frame)
                prefix = case["array_prefix"]
                if not np.array_equal(images, saved[prefix + "_images"]):
                    raise ValueError("Shared preprocessing differs from the saved export input")
                expected = [saved[prefix + "_output0"], saved[prefix + "_output1"]]
                comparisons = compare_raw(expected, backend.run_raw(images), TOLERANCE["raw"])
                if not all(r["allclose"] for r in comparisons):
                    raise ValueError("Backend raw output exceeds saved B1 tolerances")
                actual = backend.predict(frame)
                reference = postprocess_b1(*expected, geometry, **bundle.config["postprocess"])
                cases.append({"image_id": case["image_id"], "source": case["source"], "split": "val",
                              "raw_comparison": comparisons, "instances": actual.to_dict(),
                              "full_pipeline_comparison": compare_instances(actual.instances, reference)})
    finally:
        backend.close()
    profile = asset_ref(backend.profile_path) if getattr(backend, "profile_path", None) else None
    result = {"backend": identity, "cases": cases, "backend_closed": backend.closed, "profile": profile,
              "torch_imported": "torch" in sys.modules, "ultralytics_imported": "ultralytics" in sys.modules}
    if kind == "onnx" and (result["torch_imported"] or result["ultralytics_imported"]):
        raise ValueError("ORT worker unexpectedly imported a training framework")
    print("__DEPLOY_RUNTIME__=" + json.dumps(result, ensure_ascii=False))


def child(command, prefix):
    completed = subprocess.run([sys.executable, "-B", "-X", "utf8", *command], cwd=ROOT, check=True,
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
    lines = [s[len(prefix):] for s in completed.stdout.splitlines() if s.startswith(prefix)]
    if len(lines) != 1:
        raise ValueError("Child did not return exactly one runtime receipt")
    return json.loads(lines[0])


def inspect_video(path):
    capture = cv2.VideoCapture(str(path))
    shapes = []
    try:
        if not capture.isOpened():
            raise ValueError("Saved video cannot be reopened")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            shapes.append(list(frame.shape[:2]))
    finally:
        capture.release()
    if shapes != [[720, 1280]] * 3:
        raise ValueError("Expected exactly three 1280x720 output frames")
    return {"frames_decoded": len(shapes), "all_shape_hw": [720, 1280], "capture_released": True}


def build_video_fixture(cases, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"MJPG"), 5., (1280, 720))
    try:
        if not writer.isOpened():
            raise ValueError("Cannot create three-frame file-video fixture")
        for case in cases:
            if case["split"] != "val":
                raise ValueError("Fixture sources must remain in the frozen validation pool")
            writer.write(image_read(ROOT / case["source"]["path"]))
    finally:
        writer.release()
    return {"path": asset_ref(destination), "sources": [c["source"] for c in cases],
            "fps": 5., "codec": "MJPG", "role": "Three existing val frames in a file; not new independent data",
            "decoding": inspect_video(destination)}


def check_saved():
    from scripts.check_postprocess import run as check_previous
    report = read_json(REPORT)
    if report["status"] != "completed" or report["task"] != "M5-04":
        raise ValueError("Expected accepted M5-04 file inference")
    check_hashes(report["guarded_files"])
    check_hashes(report["implementation_sha256"])
    check_hashes({v["path"]: v["sha256"] for v in report["artifacts"].values()})
    with contextlib.redirect_stdout(sys.stderr):
        # Saved M5-03 acceptance is checked without running a model or writing files.
        check_previous(True)
    bundle = load_bundle()
    if bundle.baseline_id != report["baseline_id"] or report["tolerance"] != TOLERANCE:
        raise ValueError("Saved B1 or comparison configuration changed")
    print(json.dumps({"status": "passed", "task": "M5-04", "guarded_files": len(report["guarded_files"]),
                      "files_written": False, "model_inference": False}))


def accept():
    from scripts.check_postprocess import run as check_previous
    from app.product_data import write_json
    if REPORT.exists():
        raise FileExistsError("M5-04 acceptance exists; use --check")
    paths = {"image_cpu": DEMO / "M5-04_onnx_cpu.png", "image_cuda": DEMO / "M5-04_onnx_cuda.png",
             "video_input": DEMO / "M5-04_input_fixture.avi", "video_output": DEMO / "M5-04_onnx_cpu.avi"}
    if any(p.exists() for p in paths.values()):
        raise FileExistsError("Demo output exists; do not overwrite prior file-inference evidence")
    with contextlib.redirect_stdout(sys.stderr):
        check_previous(True)
    previous = read_json(PREVIOUS)
    guards = {**previous["guarded_files"], **previous["implementation_sha256"],
              PREVIOUS.relative_to(ROOT).as_posix(): file_sha256(PREVIOUS)}
    check_hashes(guards)
    timestamps = {p: (ROOT / p).stat().st_mtime_ns for p in guards}
    bundle = load_bundle()
    profiles = ROOT / "artifacts/B1" / ("M5-04_cuda_profile_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    runtime = {}
    for name, kind, device in (("ort_cpu", "onnx", "cpu"), ("ort_cuda", "onnx", "cuda:0"),
                              ("torch_cuda", "torch", "cuda:0")):
        command = ["scripts/check_product_deployment.py", "--worker", kind, "--device", device]
        if name == "ort_cuda":
            command.extend(["--profile-prefix", str(profiles)])
        runtime[name] = child(command, "__DEPLOY_RUNTIME__=")
    profile = runtime["ort_cuda"]["profile"]
    events = Counter(v.get("args", {}).get("provider") for v in read_json(ROOT / profile["path"])
                     if v.get("cat") == "Node" and v.get("args", {}).get("provider"))
    if not events["CUDAExecutionProvider"] or events["CPUExecutionProvider"]:
        raise ValueError("Actual CUDA node-execution evidence is absent or has CPU node fallback")
    source = bundle.metadata["smoke_cases"][0]["source"]["path"]
    cli = {}
    for name, device in (("image_cpu", "cpu"), ("image_cuda", "cuda:0")):
        cli[name] = child(["app/infer_products.py", "--source", source, "--output", str(paths[name]),
                           "--backend", "onnx", "--device", device], "__PRODUCT_RUN__=")
        if (cli[name]["execution"]["frame"]["counts"]["instances"] != 3 or
                cli[name]["torch_imported"] or cli[name]["ultralytics_imported"] or not cli[name]["backend_closed"]):
            raise ValueError("Actual image CLI did not complete standalone inference")
        if image_read(paths[name]).shape != (720, 1280, 3):
            raise ValueError("Annotated image has the wrong shape")
    fixture = build_video_fixture(bundle.metadata["smoke_cases"], paths["video_input"])
    cli["video_cpu"] = child(["app/infer_products.py", "--source", str(paths["video_input"]),
                             "--output", str(paths["video_output"]), "--backend", "onnx", "--device", "cpu"],
                            "__PRODUCT_RUN__=")
    video = cli["video_cpu"]["execution"]
    if (video["frames_inferred"] != 3 or video["exit_reason"] != "end_of_file" or
            not video["capture_released"] or not video["writer_released"] or
            cli["video_cpu"]["torch_imported"] or cli["video_cpu"]["ultralytics_imported"]):
        raise ValueError("Actual video CLI failed frame/standalone/resource checks")
    output_decode = inspect_video(paths["video_output"])
    check_hashes(guards)
    if timestamps != {p: (ROOT / p).stat().st_mtime_ns for p in guards}:
        raise ValueError("Protected inputs' modification times changed")
    report = {"schema_version": 1, "task": "M5-04", "status": "completed", "completed_at": datetime.now().astimezone().isoformat(),
        "baseline_id": bundle.baseline_id, "postprocessing_receipt": asset_ref(PREVIOUS),
        "configuration": asset_ref(ROOT / "configs/infer_products.yaml"), "tolerance": TOLERANCE,
        "runtime_workers": runtime, "actual_cuda_node_events": dict(events), "file_cli": cli,
        "video_fixture": fixture, "output_video_decoding": output_decode,
        "artifacts": {**{k: asset_ref(v) for k, v in paths.items()}, "cuda_profile": profile},
        "guarded_files": guards, "guarded_files_count": len(guards), "historical_sha256_and_mtime_unchanged": True,
        "implementation_sha256": {p: file_sha256(ROOT / p) for p in IMPLEMENTATION},
        "scope": {"new_photos": 0, "label_changes": 0, "training": 0, "sealed_test_inference": 0,
                  "new_model_exports": 0, "formal_mAP": False, "formal_benchmark": False,
                  "camera_switched_to_onnx": False, "fixed_30_input_pairing": False},
        "progress": {"M5": "4/6", "main": "28/54", "complete_modules": "3/9", "next": "M5-05 fixed development inputs"}}
    write_json(REPORT, report)
    print(json.dumps({"status": "passed", "task": "M5-04", "backends": list(runtime),
                      "cuda_node_events": dict(events), "video_frames": 3, "guarded_files": len(guards),
                      "files_written": True}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Verify saved evidence, no model execution/output writes")
    mode.add_argument("--worker", choices=("onnx", "torch"), help="Internal isolated runtime smoke worker")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--profile-prefix", help="Internal ORT profiling prefix")
    args = parser.parse_args()
    if args.worker:
        runtime_worker(args.worker, args.device, args.profile_prefix)
    elif args.check:
        check_saved()
    else:
        accept()


if __name__ == "__main__":
    main()
