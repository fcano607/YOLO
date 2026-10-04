"""Independent B1 inference on local images/videos; no camera or automatic overwrite."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from time import perf_counter

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from app.product_data import image_read, image_write
from deploy.base_backend import create_backend, file_sha256, load_bundle, read_json
from deploy.paths import resolve_path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".m4v"}
COLORS = ((40, 140, 255), (255, 160, 40), (190, 70, 180))


def check_io(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_file():
        raise ValueError("Source must be an existing local image/video file")
    if output.exists():
        raise FileExistsError("Output already exists; choose a new output path")
    if source.suffix.lower() not in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
        raise ValueError("Unsupported image/video source extension")
    if source.suffix.lower() in IMAGE_EXTENSIONS and output.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError("Image output needs an image extension")
    if source.suffix.lower() in VIDEO_EXTENSIONS and output.suffix.lower() not in {".avi", ".mp4"}:
        raise ValueError("Video output needs .avi or .mp4")
    # Preserve the project's sealed final-test pool, including byte-identical image copies.
    loading = read_json(ROOT / "data/desktop/metadata/products-v1_loading.json")
    sealed = [s for s in loading["samples"] if s["split"] == "test"]
    sealed_paths = {(ROOT / s[k]).resolve() for s in sealed for k in ("image_path", "staged_image_path")}
    if (source in sealed_paths or (source.suffix.lower() in IMAGE_EXTENSIONS and
            file_sha256(source) in {s["image_sha256"] for s in sealed})):
        raise ValueError("Final-test images remain sealed; use a training/validation or external development input")
    return source, output


def annotate(frame, result):
    if frame.shape[:2] != result.geometry.original_shape_hw:
        raise ValueError("Result geometry does not match the frame")
    image = frame.copy()
    instances = result.instances
    for mask, cls in zip(instances.masks, instances.class_ids):
        selected = mask.astype(bool)
        color = np.asarray(COLORS[int(cls)], dtype=np.float32)
        image[selected] = np.rint(image[selected].astype(np.float32) * .65 + color * .35).astype(np.uint8)
    for box, score, cls in zip(instances.boxes_xyxy, instances.scores, instances.class_ids):
        x1, y1, x2, y2 = np.rint(box).astype(int)
        color = COLORS[int(cls)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = result.class_names[int(cls)] + " %.2f" % score
        cv2.putText(image, label, (max(0, x1), max(18, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
    return image


def process_image(backend, source, output, show=False):
    frame = image_read(source)
    result = backend.predict(frame)
    annotated = annotate(frame, result)
    image_write(output, annotated)
    if show:
        try:
            cv2.imshow("Product segmentation", annotated)
            cv2.waitKey(0)
        finally:
            cv2.destroyAllWindows()
    return {"kind": "image", "frames_inferred": 1, "frame": result.to_dict(), "windows_closed": bool(show)}


def process_video(backend, source, output, show=False, max_frames=None):
    if max_frames is not None and (not isinstance(max_frames, int) or max_frames <= 0):
        raise ValueError("max_frames must be a positive integer")
    capture, writer = cv2.VideoCapture(str(source)), None
    count, detections, class_frames = 0, [], Counter()
    frame_shape, fps, reliable_fps, reason = None, None, False, "end_of_file"
    try:
        if not capture.isOpened():
            raise ValueError("Cannot open video source")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        reliable_fps = bool(np.isfinite(fps) and fps > 0)
        if not reliable_fps:
            fps = 30.
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_shape is None:
                frame_shape = frame.shape[:2]
                output = Path(output)
                output.parent.mkdir(parents=True, exist_ok=True)
                codec = "MJPG" if output.suffix.lower() == ".avi" else "mp4v"
                writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*codec), fps, frame_shape[::-1])
                if not writer.isOpened():
                    raise ValueError("Cannot open output video writer")
            elif frame.shape[:2] != frame_shape:
                raise ValueError("Video changed frame size")
            result = backend.predict(frame)  # current frame only, no cached result or second model call
            annotated = annotate(frame, result)
            writer.write(annotated)
            count += 1
            detections.append(len(result.instances.scores))
            class_frames.update({result.class_names[int(v)] for v in result.instances.class_ids})
            if show:
                cv2.imshow("Product segmentation", annotated)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    reason = "key_exit"
                    break
            if max_frames is not None and count >= max_frames:
                reason = "frame_limit"
                break
        if count == 0:
            raise ValueError("Video contains no decodable frames")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()
    return {"kind": "video", "frames_inferred": count, "frame_shape_hw": list(frame_shape),
            "source_fps": fps, "source_fps_reliable": reliable_fps, "exit_reason": reason,
            "detections_per_frame": detections, "frames_with_class": dict(class_frames),
            "capture_released": True, "writer_released": True, "windows_closed": bool(show)}


def run(args):
    source, output = check_io(resolve_path(args.source), resolve_path(args.output))
    bundle = load_bundle(args.config)
    start = perf_counter()
    backend = create_backend(bundle, args.backend, args.device)
    try:
        identity = backend.describe()
        if source.suffix.lower() in IMAGE_EXTENSIONS:
            execution = process_image(backend, source, output, args.show)
        else:
            execution = process_video(backend, source, output, args.show, args.max_frames)
    finally:
        backend.close()
    return {"status": "completed", "baseline_id": bundle.baseline_id, "source": str(source),
            "source_sha256": file_sha256(source), "output": str(output), "output_sha256": file_sha256(output),
            "backend": identity, "execution": execution, "backend_closed": backend.closed,
            "elapsed_seconds_including_startup_io": perf_counter() - start,
            "torch_imported": "torch" in sys.modules, "ultralytics_imported": "ultralytics" in sys.modules,
            "scope": "File inference; elapsed/core timings are not a formal benchmark or accuracy evaluation."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/infer_products.yaml")
    parser.add_argument("--source", required=True, help="Existing local image or video; no camera")
    parser.add_argument("--output", required=True, help="New annotated image/video path; existing output is rejected")
    parser.add_argument("--backend", choices=("onnx", "torch"))
    parser.add_argument("--device", choices=("cpu", "cuda:0"))
    parser.add_argument("--max-frames", type=int, help="Stop file-video inference after this many frames")
    parser.add_argument("--show", action="store_true", help="Show results; Q/Esc exits a video window")
    args = parser.parse_args()
    print("__PRODUCT_RUN__=" + json.dumps(run(args), ensure_ascii=False))


if __name__ == "__main__":
    main()
