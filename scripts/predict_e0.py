"""Run E0 image/video/camera inference and save inputs, overlays and JSON evidence."""

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SETTINGS = ROOT / "logs/ultralytics_settings"
SETTINGS.mkdir(parents=True, exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(SETTINGS)
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ["YOLO_OFFLINE"] = "true"

from deploy.paths import resolve_path
import yaml


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(args, output):
    import cv2
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO

    config = yaml.safe_load(resolve_path(args.config).read_text(encoding="utf-8"))
    weights = resolve_path(config["weights"]["path"])
    if not weights.is_file():
        raise FileNotFoundError("Run scripts/prepare_e0.py first")
    weight_hash = hashlib.sha256(weights.read_bytes()).hexdigest()
    if weight_hash != config["weights"]["sha256"]:
        raise ValueError("Pretrained weight SHA256 does not match configs/e0.yaml")
    lock = json.loads((ROOT / "configs/source-lock.json").read_text(encoding="utf-8"))
    actual_commit = subprocess.check_output(["git", "-C", str(resolve_path(lock["path"])),
                                             "rev-parse", "HEAD"], text=True).strip()
    if actual_commit != lock["commit"] or ultralytics.__version__ != lock["version"]:
        raise ValueError("Ultralytics source/version does not match the project lock")
    if Path(ultralytics.__file__).resolve() != resolve_path(lock["path"]) / "ultralytics/__init__.py":
        raise ValueError("Ultralytics did not load from the locked project source")
    model = YOLO(str(weights))
    if model.task != "segment" or len(model.names) != 80:
        raise ValueError("E0 requires the original 80-class segmentation model")
    mapping = []
    for target in config["targets"]:
        matches = [class_id for class_id, name in model.names.items() if name == target["name"]]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one pretrained class named {target['name']}")
        mapping.append({**target, "pretrained_id": matches[0]})
    project_ids = {item["pretrained_id"]: item["project_id"] for item in mapping}
    options = {**config["predict"], "save": False, "verbose": False}
    if not args.all_classes:
        options["classes"] = list(project_ids)
    if args.conf is not None:
        options["conf"] = args.conf
    if args.device is not None:
        options["device"] = args.device
    # Initialize the GPU/predictor before opening the camera or measuring the clip.
    model.predict(np.zeros((480, 640, 3), dtype=np.uint8), **options)
    backend = model.predictor.model
    report = {"task": "M0-04", "experiment": "E0", "status": "running",
              "started_at": datetime.now().astimezone().isoformat(),
              "source": "camera" if args.camera else str(resolve_path(args.source)),
              "output_dir": str(output), "python": sys.executable,
              "versions": {"ultralytics": ultralytics.__version__, "torch": torch.__version__,
                           "opencv": cv2.__version__}, "source_commit": actual_commit,
              "weights": {"path": str(weights), "sha256": weight_hash},
              "model": {"task": model.task, "class_count": len(model.names),
                        "execution_device": str(backend.device), "fp16": backend.fp16},
              "class_mapping": mapping, "all_classes": args.all_classes, "predict": options,
              "frames_processed": 0, "detections_by_name": {}, "frames_with_detections": 0,
              "capture_released": None, "camera_backend": None,
              "timing_note": "Demo timings include copying/plotting/saving; they are not a controlled inference benchmark.",
              "evaluation_note": "No project training or labeled precision/recall/mAP evaluation is performed."}
    write_json(output / "summary.json", report)
    capture = None
    writers = []
    counts = Counter()
    first_saved = False
    best_count = -1
    started = time.perf_counter()

    def process(frame, number, records):
        nonlocal first_saved, best_count
        prediction_started = time.perf_counter()
        result = model.predict(frame, **options)[0]
        if result.boxes is None or not torch.isfinite(result.boxes.data).all().item():
            raise ValueError("Invalid box output")
        if len(result.boxes) and (result.masks is None or len(result.masks) != len(result.boxes)):
            raise ValueError("Each detected instance must have a mask")
        detections = []
        boxes = result.boxes.data.cpu().tolist()
        if result.masks is not None:
            if not torch.isfinite(result.masks.data).all().item():
                raise ValueError("Invalid mask output")
            areas = result.masks.data.sum(dim=(1, 2)).cpu().tolist()
        else:
            areas = []
        for index, box in enumerate(boxes):
            class_id = int(box[-1])
            name = result.names[class_id]
            detections.append({"pretrained_id": class_id, "project_id": project_ids.get(class_id),
                               "name": name, "confidence": box[-2], "xyxy": box[:4],
                               "mask_area_pixels": areas[index]})
            counts[name] += 1
        annotated = result.plot(color_mode="instance", line_width=2)
        record = {"frame": number, "elapsed_seconds": round(time.perf_counter() - started, 4),
                  "shape": list(frame.shape), "detections": detections,
                  "box_device": str(result.boxes.data.device),
                  "mask_shape": list(result.masks.data.shape) if result.masks is not None else None,
                  "mask_device": str(result.masks.data.device) if result.masks is not None else None,
                  "predict_ms": round((time.perf_counter() - prediction_started) * 1000, 3)}
        records.write(json.dumps(record, ensure_ascii=False) + "\n")
        records.flush()
        report["frames_processed"] += 1
        report["frames_with_detections"] += int(bool(detections))
        if not first_saved:
            if not cv2.imwrite(str(output / "input_first.jpg"), frame):
                raise OSError("Could not save the input image")
            if not cv2.imwrite(str(output / "result_first.jpg"), annotated):
                raise OSError("Could not save the annotated image")
            first_saved = True
        if len(detections) > best_count:
            best_count = len(detections)
            cv2.imwrite(str(output / "input_best.jpg"), frame)
            cv2.imwrite(str(output / "result_best.jpg"), annotated)
            write_json(output / "best_frame.json", record)
        if args.show:
            cv2.imshow("YOLO11n-seg E0 - Q to stop", annotated)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                return annotated, False
        return annotated, True

    try:
        with (output / "frames.jsonl").open("w", encoding="utf-8") as records:
            if not args.camera and resolve_path(args.source).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                frame = cv2.imread(str(resolve_path(args.source)))
                if frame is None:
                    raise ValueError("Could not decode the input image")
                process(frame, 0, records)
            else:
                if args.camera:
                    api_name = config["camera"]["backend"]
                    api = {"DSHOW": cv2.CAP_DSHOW, "MSMF": cv2.CAP_MSMF, "ANY": cv2.CAP_ANY}[api_name]
                    capture = cv2.VideoCapture(args.camera_index if args.camera_index is not None else config["camera"]["index"], api)
                    report["camera_backend"] = api_name
                    seconds = args.seconds if args.seconds is not None else config["camera"]["seconds"]
                else:
                    capture = cv2.VideoCapture(str(resolve_path(args.source)))
                    seconds = args.seconds
                if not capture.isOpened():
                    raise RuntimeError("Could not open camera/video")
                report["actual_capture_backend"] = capture.getBackendName()
                reported_fps = capture.get(cv2.CAP_PROP_FPS)
                fps = reported_fps if 1 <= reported_fps <= 120 else config["video"]["fallback_fps"]
                report["video_fps"] = fps
                report["reported_capture_fps"] = reported_fps
                report["fallback_fps_used"] = fps != reported_fps
                started = time.perf_counter()
                while seconds is None or time.perf_counter() - started < seconds:
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        if args.camera:
                            raise RuntimeError("Camera frame read failed")
                        break
                    if not writers:
                        shape = (frame.shape[1], frame.shape[0])
                        for name in ("input.mp4", "result.mp4"):
                            writer = cv2.VideoWriter(str(output / name), cv2.VideoWriter_fourcc(*"mp4v"), fps, shape)
                            writers.append(writer)
                            if not writer.isOpened():
                                raise RuntimeError("Could not open MP4 writer")
                        report["video_codec"] = "mp4v"
                    annotated, keep_running = process(frame, report["frames_processed"], records)
                    writers[0].write(frame)
                    writers[1].write(annotated)
                    if not keep_running or (args.max_frames and report["frames_processed"] >= args.max_frames):
                        break
        if report["frames_processed"] == 0:
            raise RuntimeError("No valid frames were processed")
        report["status"] = "passed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        report["detections_by_name"] = dict(counts)
        report["processing_seconds"] = round(time.perf_counter() - started, 3)
        # Save diagnostics and finish files before a camera driver can block in release().
        for writer in writers:
            writer.release()
        write_json(output / "summary.json", report)
        if capture is not None:
            print("stage=releasing_capture", flush=True)
            releasing = time.perf_counter()
            capture.release()
            report["capture_released"] = True
            report["release_seconds"] = round(time.perf_counter() - releasing, 3)
        if args.show:
            cv2.destroyAllWindows()
        report["finished_at"] = datetime.now().astimezone().isoformat()
        write_json(output / "summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source", help="Image/video path, relative to the project root or absolute")
    source.add_argument("--camera", action="store_true", help="Use the configured DirectShow camera")
    parser.add_argument("--config", default="configs/e0.yaml")
    parser.add_argument("--output", help="New output directory; existing directories are never overwritten")
    parser.add_argument("--all-classes", action="store_true", help="Show all 80 pretrained classes")
    parser.add_argument("--device")
    parser.add_argument("--conf", type=float)
    parser.add_argument("--seconds", type=float, help="Clip duration limit; camera defaults to 10 seconds")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--camera-index", type=int)
    parser.add_argument("--show", action="store_true", help="Display a preview window; Q/Escape stops it")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.seconds is not None and args.seconds <= 0:
        parser.error("seconds must be positive")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("max-frames must be positive")
    if args.conf is not None and not 0 <= args.conf <= 1:
        parser.error("conf must be in [0,1]")
    if args.camera_index is not None and args.camera_index < 0:
        parser.error("camera-index must be nonnegative")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = resolve_path(args.output or f"runs/E0/{stamp}")
    if not args.worker:
        output.mkdir(parents=True, exist_ok=False)
    if args.camera and not args.worker:
        config = yaml.safe_load(resolve_path(args.config).read_text(encoding="utf-8"))
        duration = args.seconds if args.seconds is not None else config["camera"]["seconds"]
        timeout = max(config["camera"]["timeout"], duration + 30)
        command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker", "--output", str(output)]
        try:
            worker = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                                    errors="replace", timeout=timeout, env={**os.environ, "PYTHONUTF8": "1"})
            (output / "worker.log").write_text(worker.stdout + "\n" + worker.stderr, encoding="utf-8")
            if worker.returncode:
                print(worker.stdout + worker.stderr, file=sys.stderr)
                return worker.returncode
        except subprocess.TimeoutExpired as error:
            report_file = output / "summary.json"
            report = json.loads(report_file.read_text(encoding="utf-8")) if report_file.exists() else {}
            report.update({"status": "failed", "error": f"Camera worker timeout after {timeout}s",
                           "capture_released": False})
            write_json(report_file, report)
            for name, value in (("stdout", error.stdout), ("stderr", error.stderr)):
                if isinstance(value, bytes):
                    value = value.decode("utf-8", errors="replace")
                (output / f"worker_{name}.log").write_text(value or "", encoding="utf-8")
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 1
        report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    else:
        try:
            report = run(args, output)
        except Exception as error:
            if not (output / "summary.json").exists():
                write_json(output / "summary.json", {"task": "M0-04", "status": "failed", "error": str(error)})
            print(f"{type(error).__name__}: {error}", file=sys.stderr)
            return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
