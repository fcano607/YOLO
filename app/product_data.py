"""Product annotation contracts, Unicode image I/O and checked label export."""

import hashlib
import json
import math
from pathlib import Path
import re
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/camera"


def now():
    return datetime.now().astimezone().isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identifier(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", value):
        raise ValueError("IDs must contain only letters, digits, underscores or hyphens")
    return value


def catalog():
    value = read_json(ROOT / "data/desktop/metadata/classes.json")
    if value["status"] != "frozen" or value["mapping_version"] != "products-v1":
        raise ValueError("Expected frozen products-v1 mapping")
    if [item["id"] for item in value["classes"]] != [0, 1, 2]:
        raise ValueError("Product class IDs must be 0, 1, 2")
    return value


def image_read(path):
    import cv2
    import numpy as np
    result = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if result is None:
        raise ValueError("Cannot decode image: " + str(path))
    return result


def image_write(path, image):
    import cv2
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError("Image encoding failed")
    encoded.tofile(str(path))


def validate_polygon(points):
    if not isinstance(points, list) or len(points) < 3:
        raise ValueError("Each polygon needs at least three vertices")
    clean = []
    for point in points:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError("Vertices must be [x, y]")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
               not math.isfinite(v) or not 0 <= v <= 1 for v in point):
            raise ValueError("Coordinates must be finite normalized values in [0, 1]")
        clean.append([float(v) for v in point])
    if len({tuple(p) for p in clean}) != len(clean):
        raise ValueError("Repeated vertices are not accepted")
    area = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(clean, clean[1:] + clean[:1]))
    if abs(area) < 1e-9:
        raise ValueError("Polygon area is zero")

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def on_segment(a, b, p):
        return (abs(cross(a, b, p)) < 1e-12 and
                min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and
                min(a[1], b[1]) <= p[1] <= max(a[1], b[1]))

    edges = list(zip(clean, clean[1:] + clean[:1]))
    for i, (a, b) in enumerate(edges):
        for j in range(i + 1, len(edges)):
            if j == i + 1 or (i == 0 and j == len(edges) - 1):
                continue
            c, d = edges[j]
            products = cross(a, b, c) * cross(a, b, d), cross(c, d, a) * cross(c, d, b)
            if (products[0] < 0 and products[1] < 0) or any(
                    (on_segment(a, b, c), on_segment(a, b, d),
                     on_segment(c, d, a), on_segment(c, d, b))):
                raise ValueError(f"Polygon intersects itself: edges {i + 1} and {j + 1}")
    return clean


def save_review(raw, sample, draft, payload):
    """Only explicit human review creates labels; excluded images have no label."""
    raw = Path(raw)
    image_id = identifier(sample["image_id"])
    if not isinstance(payload, dict):
        raise ValueError("Review must be a JSON object")
    if payload.get("confirmed") is not True:
        raise ValueError("Explicit human confirmation is required")
    status = payload.get("status")
    if status not in {"reviewed", "negative", "excluded"}:
        raise ValueError("Review status must be reviewed, negative or excluded")
    seconds = payload.get("active_seconds", 0)
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or not 0 <= seconds <= 86400:
        raise ValueError("Invalid active review duration")
    if digest(raw / sample["image_path"]) != sample["image_sha256"]:
        raise ValueError("Original image has changed")
    polygons = payload.get("polygons", [])
    if not isinstance(polygons, list):
        raise ValueError("Polygons must be a list")
    if status == "reviewed" and not polygons:
        raise ValueError("Use explicit negative status for an empty image")
    if status in {"negative", "excluded"} and polygons:
        raise ValueError("Negative/excluded images must not export polygons")
    reason = str(payload.get("reason", "")).strip()
    if status == "excluded" and not reason:
        raise ValueError("Excluded images need a reason")
    candidates = {item["id"]: item for item in draft["candidates"]}
    saved = []
    used = set()
    unchanged = edited = manual = 0
    for polygon_index, poly in enumerate(polygons):
        cls = poly.get("class_id")
        if isinstance(cls, bool) or not isinstance(cls, int) or cls not in {0, 1, 2}:
            raise ValueError("Every retained polygon needs a product class 0, 1 or 2")
        source_id = poly.get("source_candidate_id")
        raw_points = poly.get("points")
        cleanup = None
        try:
            points = validate_polygon(raw_points)
        except ValueError as original_error:
            # A manual last click near the start often crosses the closing edge.
            # Remove ONLY that redundant closing click, and only if the result validates.
            repaired = False
            if source_id is None and isinstance(raw_points, list) and len(raw_points) >= 4 and (
                    str(original_error).startswith("Polygon intersects itself") or
                    str(original_error) == "Repeated vertices are not accepted"):
                first, last = raw_points[0], raw_points[-1]
                distance = math.hypot((first[0] - last[0]) * sample["width"],
                                      (first[1] - last[1]) * sample["height"])
                if distance <= 5:
                    try:
                        points = validate_polygon(raw_points[:-1])
                        cleanup = "removed_near_duplicate_closing_vertex_within_5_original_pixels"
                        repaired = True
                    except ValueError:
                        pass
            if not repaired:
                name = catalog()["classes"][cls]["display_name"]
                raise ValueError(f"第 {polygon_index + 1} 条轮廓（{name}）：{original_error}。"
                                 "请沿外边缘依次画点；可简化、拖动交叉顶点或重画这一条。") from original_error
        if source_id is not None:
            if source_id not in candidates or source_id in used:
                raise ValueError("Unknown or reused source candidate")
            used.add(source_id)
            original = candidates[source_id]["points"]
            same = len(points) == len(original) and all(
                abs(a - b) <= 1e-7 for p, q in zip(points, original) for a, b in zip(p, q))
            unchanged += int(same)
            edited += int(not same)
        else:
            manual += 1
        saved.append({"class_id": cls, "points": points, "source_candidate_id": source_id,
                      "geometry_cleanup": cleanup})
    record = {
        "schema_version": 1, "mapping_version": "products-v1", "image_id": image_id,
        "group_id": sample["group_id"], "image_sha256": sample["image_sha256"],
        "draft_sha256": digest(raw / "annotations/drafts" / (image_id + ".json")),
        "status": status, "reviewed_at": now(), "reviewer": "human_local_ui",
        "confirmed": True, "reason": reason, "active_seconds": round(seconds, 3),
        "physical_ids": [catalog()["classes"][cls]["physical_id"]
                         for cls in sorted({p["class_id"] for p in saved})],
        "physical_id_basis": "One known physical item per class in this pilot; based on reviewed class IDs.",
        "timing_basis": "Browser active time; pauses on hidden tab or 60s inactivity; not a controlled benchmark",
        "polygons": saved, "stats": {
            "draft_candidates": len(candidates), "retained_unchanged": unchanged,
            "retained_boundary_edited": edited, "manual_polygons": manual,
            "discarded_candidates": len(candidates) - len(used),
        },
    }
    reviewed = raw / "annotations/reviewed"
    # JSON is the canonical review record; TXT is reproducible from it.
    write_json(reviewed / (image_id + ".json"), record)
    label = reviewed / (image_id + ".txt")
    if status == "excluded":
        label.unlink(missing_ok=True)
    else:
        text = "".join(str(p["class_id"]) + " " + " ".join(
            f"{v:.8f}" for point in p["points"] for v in point) + "\n" for p in saved)
        temporary = label.with_suffix(".txt.part")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(label)
    return record


def group_summary(raw, manifest):
    raw = Path(raw)
    records = []
    draft_details = []
    for sample in manifest["samples"]:
        draft_path = raw / "annotations/drafts" / (sample["image_id"] + ".json")
        if draft_path.exists():
            draft = read_json(draft_path)
            draft_details.append({"image_id": sample["image_id"],
                                  "predict_conf": draft.get("predict_conf"),
                                  "candidates": [{"coco_name": c.get("coco_name"),
                                                  "score": c.get("score"),
                                                  "vertices": len(c["points"])}
                                                 for c in draft["candidates"]]})
        path = raw / "annotations/reviewed" / (sample["image_id"] + ".json")
        if path.exists():
            record = read_json(path)
            if record["image_sha256"] != sample["image_sha256"]:
                raise ValueError("Review is for a different image")
            records.append(record)
    accepted = [r for r in records if r["status"] in {"reviewed", "negative"}]
    objects = sum(len(r["polygons"]) for r in accepted)
    manual = sum(r["stats"]["manual_polygons"] for r in accepted)
    return {
        "schema_version": 1, "task": "M2-02", "group_id": manifest["group_id"],
        "updated_at": now(), "mapping_version": "products-v1",
        "raw_images": len(manifest["samples"]),
        "draft_images": len(draft_details), "draft_per_image": draft_details,
        "reviewed_target_images": sum(r["status"] == "reviewed" for r in records),
        "confirmed_negative_images": sum(r["status"] == "negative" for r in records),
        "excluded_images": sum(r["status"] == "excluded" for r in records),
        "unreviewed_images": len(manifest["samples"]) - len(records),
        "instances": objects,
        "retained_unchanged": sum(r["stats"]["retained_unchanged"] for r in accepted),
        "retained_boundary_edited": sum(r["stats"]["retained_boundary_edited"] for r in accepted),
        "manual_polygons": manual,
        "manual_polygon_fraction": manual / objects if objects else None,
        "active_review_seconds": sum(r["active_seconds"] for r in records),
        "per_image": [{"image_id": r["image_id"], "status": r["status"],
                       "active_seconds": r["active_seconds"], **r["stats"]} for r in records],
        "limits": ["Manual polygons include both missed targets and replacement of unusable drafts.",
                   "Discarded COCO candidates are not automatically false positives of the new product model.",
                   "Pilot data are not a formal test set or evidence of product model training."],
    }
