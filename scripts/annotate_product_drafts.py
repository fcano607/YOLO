"""Capture/import original product images and generate unreviewed COCO mask proposals."""

import argparse
from datetime import datetime
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.product_data import RAW, catalog, digest, identifier, image_read, image_write, now, read_json, write_json

EVENT = "__CAPTURE_EVENT__="
WINDOW = "Product capture | SPACE save | Q/Esc quit"


def plan(full=False):
    shots = []
    for background in range(2 if full else 1):
        for cls in range(3):
            for pose in ("front", "angled"):
                shots.append({"expected_class_ids": [cls], "pose": pose,
                              "background": background + 1})
    if full:
        shots.extend({"expected_class_ids": [0, 1, 2], "pose": pose, "background": 2}
                     for pose in ("mixed-clear-1", "mixed-clear-2", "mixed-light-occlusion-1",
                                  "mixed-light-occlusion-2"))
        shots.extend({"expected_class_ids": [], "pose": pose, "background": 2}
                     for pose in ("empty-1", "empty-2", "non-target-1", "non-target-2"))
    return shots


def new_manifest(group_id, mode):
    group_id = identifier(group_id)
    path = RAW / "sessions" / (group_id + ".json")
    if path.exists():
        raise FileExistsError("Group already exists; use another --group: " + group_id)
    return {"schema_version": 1, "mapping_version": "products-v1", "group_id": group_id,
            "created_at": now(), "mode": mode, "status": "collecting", "samples": []}


def add_sample(manifest, frame, hint, source=None):
    index = len(manifest["samples"]) + 1
    image_id = manifest["group_id"] + "_" + f"{index:03d}"
    path = RAW / "frames" / manifest["group_id"] / (image_id + ".png")
    if path.exists():
        raise FileExistsError("Will not overwrite an original image")
    image_write(path, frame)
    sha = digest(path)
    if any(sample["image_sha256"] == sha for sample in manifest["samples"]):
        path.unlink()
        raise ValueError("Exact duplicate image; change position/angle before saving")
    entries = catalog()["classes"]
    sample = {"image_id": image_id, "group_id": manifest["group_id"],
              "image_path": path.relative_to(RAW).as_posix(), "image_sha256": sha,
              "width": frame.shape[1], "height": frame.shape[0], "captured_at": now(),
              "capture_hint": hint,
              "hint_note": "Capture prompt is not a verified label; human review must check every instance.",
              "physical_ids": [entries[c]["physical_id"] for c in hint["expected_class_ids"]],
              "physical_ids_status": "capture_prompt_hint; actual visible instances must be reviewed",
              "source_path": str(source) if source else None}
    manifest["samples"].append(sample)
    write_json(RAW / "sessions" / (manifest["group_id"] + ".json"), manifest)
    return sample


def capture_worker(args):
    import cv2
    import yaml
    config = yaml.safe_load((ROOT / "configs/e0.yaml").read_text("utf-8"))
    entries = catalog()["classes"]
    manifest = new_manifest(args.group, "camera_manual_snapshots")
    shots = plan(args.plan in {"full", "remaining"})
    if args.plan == "remaining":
        shots = shots[6:]
    manifest["capture_plan"] = args.plan
    manifest["planned_images"] = len(shots)
    capture = None
    created = False
    heartbeat = time.monotonic()
    try:
        backend = {"DSHOW": cv2.CAP_DSHOW, "MSMF": cv2.CAP_MSMF, "ANY": cv2.CAP_ANY}[config["camera"]["backend"]]
        capture = cv2.VideoCapture(args.camera_index, backend)
        if not capture.isOpened():
            raise RuntimeError("Cannot open camera")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        created = True
        print("Camera opened. Only SPACE saves a frame. Follow the prompt; Q exits.", flush=True)
        print(EVENT + "ready", flush=True)
        while len(manifest["samples"]) < len(shots):
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError("Camera read failed")
            i = len(manifest["samples"])
            shot = shots[i]
            names = " + ".join(entries[c]["name"] for c in shot["expected_class_ids"]) or "NO target products"
            scale = min(1.0, 1100 / frame.shape[1], 620 / frame.shape[0])
            preview = cv2.resize(frame, None, fx=scale, fy=scale)
            import numpy as np
            header = np.zeros((100, preview.shape[1], 3), dtype=np.uint8)
            for row, text in enumerate([
                    f"{i + 1}/{len(shots)}: {names}",
                    f"Pose: {shot['pose']} | Background {shot['background']} | Original {frame.shape[1]}x{frame.shape[0]}",
                    "SPACE: save original + next | Q/Esc: quit (saved photos retained)"]):
                cv2.putText(header, text, (12, 27 + row * 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.62, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(WINDOW, cv2.vconcat([header, preview]))
            key = cv2.waitKey(15) & 0xFF
            if key in (27, ord("q"), ord("Q")) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == 32:
                try:
                    sample = add_sample(manifest, frame.copy(), shot)
                    print("Saved " + sample["image_id"], flush=True)
                except ValueError as error:
                    print(str(error), flush=True)
            if time.monotonic() - heartbeat >= 1:
                print(EVENT + "heartbeat", flush=True)
                heartbeat = time.monotonic()
        manifest["status"] = "captured" if len(manifest["samples"]) == len(shots) else "partial"
        manifest["ended_at"] = now()
        write_json(RAW / "sessions" / (args.group + ".json"), manifest)
        print(f"Collection {manifest['status']}: {len(manifest['samples'])} originals", flush=True)
    except Exception as error:
        manifest["status"] = "error"
        manifest["error"] = str(error)
        manifest["ended_at"] = now()
        write_json(RAW / "sessions" / (args.group + ".json"), manifest)
        raise
    finally:
        print(EVENT + "closing", flush=True)
        if created:
            cv2.destroyAllWindows()
        if capture is not None:
            capture.release()


def capture_supervised(args):
    child = subprocess.Popen([sys.executable, "-B", "-X", "utf8", str(Path(__file__).resolve()),
                              *sys.argv[1:], "--worker"], cwd=str(ROOT),
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace")
    messages = queue.Queue()
    def reader():
        for line in child.stdout:
            messages.put(line.rstrip())
    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    activity = time.monotonic()
    closing = None
    try:
        while True:
            try:
                line = messages.get(timeout=0.2)
                activity = time.monotonic()
                if line == EVENT + "closing":
                    closing = activity
                elif not line.startswith(EVENT):
                    print(line, flush=True)
            except queue.Empty:
                pass
            if child.poll() is not None:
                thread.join(timeout=1)
                while not messages.empty():
                    line = messages.get_nowait()
                    if not line.startswith(EVENT):
                        print(line, flush=True)
                return child.returncode
            elapsed = time.monotonic()
            if closing is not None and elapsed - closing > 10:
                print("Camera release timed out. Closing only this capture worker; saved originals retained.", flush=True)
                session = RAW / "sessions" / (args.group + ".json")
                if session.exists() and read_json(session)["status"] in {"captured", "partial"}:
                    return 0
                return 1
            if elapsed - activity > 60:
                raise TimeoutError("Camera worker has stopped responding")
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)
        thread.join(timeout=1)
        child.stdout.close()


def import_images(args):
    source = Path(args.input).expanduser().resolve()
    if not source.is_dir():
        raise ValueError("--input must be a directory")
    entries = catalog()["classes"]
    manifest = new_manifest(args.group, "import_original_photos")
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)
    if not files:
        raise ValueError("No supported image files found")
    # Decode untouched photos to lossless PNG; source path/hash kept for provenance.
    for path in files:
        hint_ids = [c["id"] for c in entries if path.parent.name == c["name"]]
        sample = add_sample(manifest, image_read(path),
                            {"expected_class_ids": hint_ids, "pose": "user_photo",
                             "background": "unknown"}, path)
        sample["source_sha256"] = digest(path)
    manifest["status"] = "imported"
    write_json(RAW / "sessions" / (args.group + ".json"), manifest)
    print(f"Imported {len(files)} originals; no product labels created.")


def bundle_groups(args):
    """Combine review lists without copying photos or erasing original capture groups."""
    if not args.groups:
        raise ValueError("--groups is required for bundle")
    manifest = new_manifest(args.group, "review_bundle")
    seen_ids, seen_hashes = set(), set()
    manifest["source_group_ids"] = []
    for group in args.groups:
        source = read_json(RAW / "sessions" / (identifier(group) + ".json"))
        if source["mapping_version"] != manifest["mapping_version"]:
            raise ValueError("Mappings differ")
        manifest["source_group_ids"].append(group)
        for sample in source["samples"]:
            if sample["image_id"] in seen_ids or sample["image_sha256"] in seen_hashes:
                raise ValueError("Repeated image in bundle")
            manifest["samples"].append(sample)
            seen_ids.add(sample["image_id"])
            seen_hashes.add(sample["image_sha256"])
    if not manifest["samples"]:
        raise ValueError("No photos to bundle")
    manifest["status"] = "review_list_ready"
    manifest["group_note"] = "Review list only. Original per-image group_id is preserved. These related pilot sessions are not an independent train/val split."
    write_json(RAW / "sessions" / (args.group + ".json"), manifest)
    print(f"Bundled {len(manifest['samples'])} image references; originals and labels not copied.")


def drafts(args):
    import yaml
    from app.live_camera import load_model
    manifest = read_json(RAW / "sessions" / (identifier(args.group) + ".json"))
    if not manifest["samples"]:
        raise ValueError("No captured images")
    config = yaml.safe_load((ROOT / "configs/e0.yaml").read_text("utf-8"))
    model, options = load_model(config, device=args.device, all_classes=True)
    options.update(conf=args.conf, classes=None)
    lock = read_json(ROOT / "configs/source-lock.json")
    for sample in manifest["samples"]:
        path = RAW / "annotations/drafts" / (sample["image_id"] + ".json")
        if path.exists():
            old = read_json(path)
            if old["image_sha256"] != sample["image_sha256"] or old["predict_conf"] != args.conf:
                raise ValueError("Existing draft differs; use a new group instead of overwriting")
            print("Existing draft retained: " + sample["image_id"], flush=True)
            continue
        original = RAW / sample["image_path"]
        if digest(original) != sample["image_sha256"]:
            raise ValueError("Original image changed")
        result = model.predict(image_read(original), **options)[0]
        candidates = []
        if result.masks is not None:
            for i, (points, box) in enumerate(zip(result.masks.xyn, result.boxes)):
                # A closed contour may repeat its first vertex; the TXT needs unique vertices.
                polygon = []
                for point in points.tolist():
                    p = [min(1., max(0., float(v))) for v in point]
                    if not polygon or p != polygon[-1]:
                        polygon.append(p)
                if len(polygon) > 1 and polygon[0] == polygon[-1]:
                    polygon.pop()
                if len(polygon) < 3:
                    continue
                cls = int(box.cls.item())
                candidates.append({"id": f"c{i:03d}", "coco_class_id": cls,
                                   "coco_name": model.names[cls], "score": float(box.conf.item()),
                                   "points": polygon, "class_id": None,
                                   "box_xyxy": box.xyxy[0].cpu().tolist()})
        write_json(path, {"schema_version": 1, "mapping_version": "products-v1",
                          "status": "draft_unreviewed", "image_id": sample["image_id"],
                          "image_sha256": sample["image_sha256"], "generated_at": now(),
                          "predict_conf": args.conf, "all_coco_classes": True,
                          "model": "yolo11n-seg", "weights_sha256": config["weights"]["sha256"],
                          "source_commit": lock["commit"], "candidates": candidates})
        print(f"{sample['image_id']}: {len(candidates)} COCO proposals; human review required.", flush=True)


def check_reviews(args):
    import os
    os.environ["YOLO_CONFIG_DIR"] = str(ROOT / "logs/ultralytics_settings")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"
    import numpy as np
    import ultralytics
    from ultralytics.data.utils import verify_image_label, polygons2masks
    from app.product_data import group_summary, validate_polygon
    lock = read_json(ROOT / "configs/source-lock.json")
    source = ROOT / lock["path"]
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != lock["commit"] or ultralytics.__version__ != lock["version"] or Path(ultralytics.__file__).resolve() != source / "ultralytics/__init__.py":
        raise ValueError("Label checker must use the locked source")
    manifest = read_json(RAW / "sessions" / (identifier(args.group) + ".json"))
    checks = []
    class_counts = [0, 0, 0]
    for sample in manifest["samples"]:
        original = RAW / sample["image_path"]
        if digest(original) != sample["image_sha256"]:
            raise ValueError("Original image changed: " + sample["image_id"])
        path = RAW / "annotations/reviewed" / (sample["image_id"] + ".json")
        if not path.exists():
            continue
        review = read_json(path)
        label = path.with_suffix(".txt")
        if review["status"] == "excluded":
            if label.exists():
                raise ValueError("Excluded image has a training label")
            checks.append({"image_id": sample["image_id"], "status": "excluded",
                           "reason": review["reason"], "review_sha256": digest(path)})
            continue
        for poly in review["polygons"]:
            validate_polygon(poly["points"])
        result = verify_image_label((str(original), str(label), "M2-02: ", False, 3, 0, 2, False))
        if result[5] != 0 or result[6] != 1 or result[8] != 0 or result[9]:
            raise ValueError("Label loading failed: " + result[9])
        if result[2] != (sample["height"], sample["width"]):
            raise ValueError("Image dimensions changed")
        if len(result[1]) != len(review["polygons"]):
            raise ValueError("TXT instance count differs from canonical review")
        for box, segment, poly in zip(result[1], result[3], review["polygons"]):
            if int(box[0]) != poly["class_id"] or not np.allclose(segment, poly["points"], rtol=0, atol=1e-7):
                raise ValueError("TXT class/geometry differs from canonical review")
        if review["status"] == "negative" and len(result[1]):
            raise ValueError("Negative label is not empty")
        segments = [points * np.array([sample["width"], sample["height"]]) for points in result[3]]
        masks = polygons2masks(result[2], segments, 1, 4) if segments else []
        if any(int(mask.sum()) == 0 for mask in masks):
            raise ValueError("Polygon produces an empty mask")
        for poly in review["polygons"]:
            class_counts[poly["class_id"]] += 1
        checks.append({"image_id": sample["image_id"], "status": review["status"],
                       "instances": len(result[1]), "mask_nonzero_pixels": [int(m.sum()) for m in masks],
                       "physical_ids": [catalog()["classes"][cls]["physical_id"]
                                        for cls in sorted({p["class_id"] for p in review["polygons"]})],
                       "image_sha256": sample["image_sha256"], "label_sha256": digest(label),
                       "review_sha256": digest(path)})
    summary = group_summary(RAW, manifest)
    summary.update({"label_load_checks": checks, "class_instance_counts": class_counts,
                    "loader_source_commit": lock["commit"],
                    "label_check_status": "saved_labels_passed" if summary["unreviewed_images"] else "all_images_reviewed_and_labels_passed"})
    output = ROOT / "reports/data" / ("M2-02_" + args.group + ".json")
    write_json(output, summary)
    print(f"Checked {len(checks)} reviewed images. Pending: {summary['unreviewed_images']}. "
          f"Instances by product 0/1/2: {class_counts}. Report: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["capture", "import", "draft", "bundle", "check"])
    parser.add_argument("--group", default="pilot_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--plan", choices=["six", "remaining", "full"], default="six",
                        help="six: 6 originals; remaining: another background/mixed/negative, 14 images; full: 20")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--input")
    parser.add_argument("--groups", nargs="+", help="Existing group IDs to combine into a review list")
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--device", default="0")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    identifier(args.group)
    if not 0 < args.conf < 1 or args.camera_index < 0:
        parser.error("Invalid threshold or camera index")
    if args.mode == "import" and not args.input:
        parser.error("--input is required for import")
    catalog()
    if args.mode == "capture":
        if args.worker:
            capture_worker(args)
            return 0
        return capture_supervised(args)
    if args.mode == "check":
        check_reviews(args)
    elif args.mode == "bundle":
        bundle_groups(args)
    elif args.mode == "import":
        import_images(args)
    else:
        drafts(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
