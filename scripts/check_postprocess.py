"""M5-03: compare independent postprocessing on saved raw outputs and in-memory fixtures."""
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
from deploy.postprocess import postprocess_b1, scale_mask_logits, select_candidates, validate_postprocessing_contract
from deploy.preprocess import preprocess_bgr
from scripts.check_preprocess import run as check_preprocessing
from scripts.export_onnx import now, reference


REPORT = ROOT / "reports/deployment/M5-03_B1_postprocess.json"
PREVIOUS = ROOT / "reports/deployment/M5-02_B1_preprocess.json"
IMPLEMENTATION = ("deploy/postprocess.py", "scripts/check_postprocess.py", "tests/test_postprocess.py")
TOLERANCE = {"box_atol": .0001, "logit_atol": .0002, "logit_rtol": .00002,
             "min_mask_iou": .999, "max_mask_disagreement_fraction": .00001}


def independent_import_check():
    code = """import json, sys, numpy as np
from deploy.preprocess import preprocess_bgr
from deploy.postprocess import postprocess_b1
_, geometry = preprocess_bgr(np.zeros((17,29,3), np.uint8))
raw = np.zeros((1,39,1), np.float32)
raw[0,:4,0], raw[0,4,0], raw[0,7,0] = [320,320,640,640], .9, 1
result = postprocess_b1(raw, np.ones((1,32,160,160), np.float32), geometry)
print(json.dumps({'torch_imported': 'torch' in sys.modules, 'ultralytics_imported': 'ultralytics' in sys.modules,
                  'boxes_shape': list(result.boxes_xyxy.shape), 'mask_shape': list(result.masks.shape),
                  'mask_dtype': str(result.masks.dtype), 'mask_nonempty': bool(result.masks.any())}))
"""
    process = subprocess.run([sys.executable, "-B", "-X", "utf8", "-c", code], cwd=ROOT,
                             check=True, capture_output=True, text=True, encoding="utf-8")
    result = json.loads(process.stdout)
    if result != {"torch_imported": False, "ultralytics_imported": False, "boxes_shape": [1, 4],
                  "mask_shape": [1, 17, 29], "mask_dtype": "uint8", "mask_nonempty": True}:
        raise ValueError("Standalone postprocessing failed or imported a training framework")
    return result


def compare_case(raw, proto, geometry, case_id, options, original_image=None):
    import torch
    from ultralytics.utils import nms, ops

    raw_hash = hashlib.sha256(raw.tobytes()).hexdigest()
    proto_hash = hashlib.sha256(proto.tobytes()).hexdigest()
    selected = select_candidates(raw, **options)
    # Separate actual candidate/coeff correspondence from mask floating-point differences.
    predictions, indices = nms.non_max_suppression(torch.from_numpy(raw.copy()),
        conf_thres=options["conf"], iou_thres=options["iou"], max_det=options["max_det"],
        multi_label=options["multi_label"], nc=3, agnostic=False, return_idxs=True)
    prediction, indices = predictions[0], indices[0].reshape(-1).numpy().astype(np.int64)
    rows = prediction.numpy()
    if (not np.array_equal(selected.candidate_indices, indices) or
            not np.array_equal(selected.class_ids, rows[:, 5].astype(np.int64)) or
            not np.array_equal(selected.scores, rows[:, 4]) or
            not np.array_equal(selected.mask_coefficients, rows[:, 6:]) or
            not np.allclose(selected.boxes_input_xyxy, rows[:, :4], rtol=0, atol=TOLERANCE["box_atol"])):
        raise ValueError("NMS indices / classes / scores / coefficients / boxes differ: " + case_id)
    reference_boxes = ops.scale_boxes((640, 640), prediction[:, :4].clone(), geometry.original_shape_hw)
    reference_masks = ops.process_mask_native(torch.from_numpy(proto[0].copy()), prediction[:, 6:],
                                               reference_boxes, geometry.original_shape_hw).numpy()
    reference_keep = reference_masks.any(axis=(1, 2))
    actual = postprocess_b1(raw, proto, geometry, **options)
    if (not np.array_equal(actual.candidate_indices, indices[reference_keep]) or
            not np.array_equal(actual.class_ids, rows[reference_keep, 5].astype(np.int64)) or
            not np.array_equal(actual.scores, rows[reference_keep, 4]) or
            not np.array_equal(actual.mask_coefficients, rows[reference_keep, 6:])):
        raise ValueError("Final instance binding or empty-mask filter differs: " + case_id)
    reference_boxes = reference_boxes.numpy()[reference_keep]
    reference_masks = reference_masks[reference_keep]
    if actual.masks.shape != reference_masks.shape or actual.masks.dtype != np.uint8:
        raise ValueError("Original mask shape / dtype differs: " + case_id)
    predictor_checked = False
    if original_image is not None and not options["multi_label"]:
        from ultralytics.models.yolo.segment.predict import SegmentationPredictor
        predictor = object.__new__(SegmentationPredictor)
        predictor.args = SimpleNamespace(retina_masks=True)
        predictor.model = SimpleNamespace(names={0: "sam_whole_milk", 1: "yili_shuhua", 2: "luckin_cup"})
        result = predictor.construct_result(prediction.clone(), torch.empty((1, 3, 640, 640)),
                                            original_image, case_id, torch.from_numpy(proto[0].copy()))
        masks = result.masks.data.numpy() if result.masks is not None else np.empty(
            (0, *geometry.original_shape_hw), np.uint8)
        if (not np.array_equal(result.boxes.data.numpy(), np.concatenate(
                (reference_boxes, rows[reference_keep, 4:6]), axis=1)) or
                not np.array_equal(masks, reference_masks)):
            raise ValueError("Official predictor construction or empty-mask rule differs: " + case_id)
        predictor_checked = True
    box_error = float(np.abs(actual.boxes_xyxy - reference_boxes).max()) if len(reference_boxes) else 0.
    if box_error > TOLERANCE["box_atol"]:
        raise ValueError("Original box coordinates exceed tolerance: " + case_id)
    intersections = (actual.masks.astype(bool) & reference_masks.astype(bool)).sum(axis=(1, 2))
    unions = (actual.masks.astype(bool) | reference_masks.astype(bool)).sum(axis=(1, 2))
    ious = np.divide(intersections, unions, out=np.ones(len(unions), np.float64), where=unions > 0)
    disagreement = (actual.masks != reference_masks).sum(axis=(1, 2))
    fractions = disagreement / np.prod(geometry.original_shape_hw)
    if ((ious < TOLERANCE["min_mask_iou"]).any() or
            (fractions > TOLERANCE["max_mask_disagreement_fraction"]).any()):
        raise ValueError("Binary mask parity exceeds tolerance: " + case_id + " " + str(ious.tolist()))
    # Compare prototype combination + scale_masks numerically, in small chunks to bound memory.
    max_combined_error, max_scaled_error = 0., 0.
    for start in range(0, len(selected.scores), 4):
        coeff = selected.mask_coefficients[start:start + 4]
        own_logits = (coeff @ proto[0].reshape(32, -1)).reshape(-1, 160, 160)
        torch_logits = (torch.from_numpy(coeff.copy()) @ torch.from_numpy(proto[0].copy()).reshape(32, -1))
        torch_logits = torch_logits.reshape(-1, 160, 160)
        own_scaled = scale_mask_logits(own_logits, geometry.original_shape_hw)
        official_scaled = ops.scale_masks(torch_logits[None], geometry.original_shape_hw)[0].numpy()
        max_combined_error = max(max_combined_error, float(np.abs(own_logits - torch_logits.numpy()).max()))
        max_scaled_error = max(max_scaled_error, float(np.abs(own_scaled - official_scaled).max()))
        if not np.allclose(own_scaled, official_scaled, atol=TOLERANCE["logit_atol"], rtol=TOLERANCE["logit_rtol"]):
            raise ValueError("Combined / resized mask logits exceed tolerance: " + case_id)
    if raw_hash != hashlib.sha256(raw.tobytes()).hexdigest() or proto_hash != hashlib.sha256(proto.tobytes()).hexdigest():
        raise ValueError("Postprocessing mutated a raw input")
    return {"case_id": case_id, "options": options, "geometry": geometry.to_dict(), "counts": actual.counts,
            "candidate_indices": actual.candidate_indices.tolist(), "class_ids": actual.class_ids.tolist(),
            "scores": actual.scores.tolist(), "boxes_xyxy": actual.boxes_xyxy.tolist(),
            "raw_sha256": {"output0": raw_hash, "output1": proto_hash},
            "official_predictor_construction_checked": predictor_checked,
            "nms_indices_classes_scores_coefficients_equal": True, "source_unchanged": True,
            "max_box_abs": box_error, "max_combined_logit_abs": max_combined_error,
            "max_scaled_logit_abs": max_scaled_error,
            "mask_iou_min": float(ious.min()) if len(ious) else None,
            "mask_iou_mean": float(ious.mean()) if len(ious) else None,
            "mask_disagreement_pixels_max": int(disagreement.max()) if len(disagreement) else 0,
            "mask_disagreement_fraction_max": float(fractions.max()) if len(fractions) else 0.,
            "binary_masks_equal": bool(np.array_equal(actual.masks, reference_masks)), "passed": True}


def synthetic_inputs():
    """Fixtures for geometry and selection rules, never written as image/annotation data."""
    y, x = np.indices((160, 160), dtype=np.float32)
    proto = np.zeros((1, 32, 160, 160), np.float32)
    proto[0, 0] = np.sin(x / 13) + .137
    proto[0, 1] = np.cos(y / 17) - .231
    proto[0, 2] = .67
    # The official helper treats a last dimension of 6 as an end-to-end layout, so use N=7.
    raw = np.zeros((1, 39, 7), np.float32)
    raw[0, :4, :6] = np.array([[320, 322, 320, 40, 100, 300], [320, 321, 320, 605, 100, 300],
                          [400, 400, 400, 190, 70, 90], [420, 420, 420, 200, 60, 85]], np.float32)
    raw[0, 4, :2], raw[0, 5, 2], raw[0, 6, 3:6] = [.91, .83], .72, [.55, .25, .07]
    raw[0, 7:10, :6] = np.array([[1, .9, -.8, .6, 1, 1], [.3, .3, .7, -.4, .1, .1],
                           [.2, .2, .1, .5, .1, .1]], np.float32)
    display = {"conf": .25, "iou": .7, "max_det": 300, "multi_label": False}
    for h, w in ((720, 1280), (1280, 720), (481, 640), (640, 481), (333, 1001), (1001, 333)):
        _, geometry = preprocess_bgr(np.zeros((h, w, 3), np.uint8))
        yield "geometry_%dx%d" % (h, w), raw.copy(), proto.copy(), geometry, display
    _, geometry = preprocess_bgr(np.zeros((640, 640, 3), np.uint8))
    yield "empty_candidates", raw[:, :, :0].copy(), proto.copy(), geometry, display
    yield "below_confidence", np.zeros_like(raw), proto.copy(), geometry, display
    yield "all_empty_masks", raw.copy(), np.zeros_like(proto), geometry, display
    multiple = raw.copy()
    multiple[0, 5, 0], multiple[0, 6, 0] = .81, .71
    yield "multi_label_binding", multiple, proto.copy(), geometry, dict(display, multi_label=True)
    yield "max_det_before_mask_filter", raw.copy(), np.zeros_like(proto), geometry, dict(display, max_det=1)
    _, small_geometry = preprocess_bgr(np.zeros((17, 29, 3), np.uint8))
    yield "small_upscaled_input", raw.copy(), proto.copy(), small_geometry, display


def verify_outputs():
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    metadata = read_json(ROOT / "artifacts/B1/best_fp32.metadata.json")
    validate_postprocessing_contract(metadata)
    records = []
    with np.load(ROOT / "artifacts/B1/export_reference.npz", allow_pickle=False) as saved:
        for case in metadata["smoke_cases"]:
            source = ROOT / case["source"]["path"]
            if case["split"] != "val" or digest(source) != case["source"]["sha256"]:
                raise ValueError("Only unchanged saved validation sources are allowed")
            original_image = image_read(source)
            _, geometry = preprocess_bgr(original_image)
            prefix = case["array_prefix"]
            for mode, conf, multi_label in (("display", .25, False), ("metric_rules_only", .001, True)):
                options = {"conf": conf, "iou": .7, "max_det": 300, "multi_label": multi_label}
                record = compare_case(saved[prefix + "_output0"], saved[prefix + "_output1"], geometry,
                                      case["image_id"] + ":" + mode, options, original_image)
                record.update(source=case["source"], split="val", mode=mode)
                records.append(record)
    for case_id, raw, proto, geometry, options in synthetic_inputs():
        record = compare_case(raw, proto, geometry, case_id, options)
        record["source"] = "in-memory raw-output fixture; not independent quality evidence"
        records.append(record)
    nonempty = [r for r in records if r["mask_iou_min"] is not None]
    return {"cases": records, "independent_import": independent_import_check(),
            "summary": {"saved_raw_images": 3, "saved_raw_modes": 2, "synthetic_cases": 12,
                "comparisons": len(records), "all_passed": all(r["passed"] for r in records),
                "all_nms_bindings_equal": True, "max_box_abs": max(r["max_box_abs"] for r in records),
                "official_predictor_constructions_checked": sum(
                    r["official_predictor_construction_checked"] for r in records),
                "max_scaled_logit_abs": max(r["max_scaled_logit_abs"] for r in records),
                "min_mask_iou": min(r["mask_iou_min"] for r in nonempty),
                "all_binary_masks_equal": all(r["binary_masks_equal"] for r in records),
                "max_mask_disagreement_pixels": max(r["mask_disagreement_pixels_max"] for r in records),
                "model_inference": False, "ground_truth_evaluation": False}}


def run(check_only=False, dry_run=False):
    if not check_only and not dry_run and REPORT.exists():
        raise FileExistsError("M5-03 receipt exists; use --check rather than overwrite it")
    saved = read_json(REPORT) if check_only else None
    if saved:
        if saved["status"] != "completed" or saved["task"] != "M5-03":
            raise ValueError("Expected the accepted M5-03 receipt")
        check_guards(saved["guarded_files"])
        check_guards(saved["implementation_sha256"])
    with contextlib.redirect_stdout(sys.stderr):
        check_preprocessing(True)
    previous = read_json(PREVIOUS)
    guards = {**previous["guarded_files"], **previous["implementation_sha256"],
              PREVIOUS.relative_to(ROOT).as_posix(): digest(PREVIOUS)}
    check_guards(guards)
    timestamps = {p: (ROOT / p).stat().st_mtime_ns for p in guards}
    checks = verify_outputs()
    check_guards(guards)
    if timestamps != {p: (ROOT / p).stat().st_mtime_ns for p in guards}:
        raise ValueError("Historical inputs' modification timestamps changed")
    if check_only:
        if saved["guarded_files"] != guards or saved["checks"] != checks or saved["tolerance"] != TOLERANCE:
            raise ValueError("Current results differ from accepted postprocessing checks")
    elif not dry_run:
        report = {"schema_version": 1, "task": "M5-03", "status": "completed", "completed_at": now(),
            "baseline_id": previous["baseline_id"], "preprocessing_receipt": reference(PREVIOUS),
            "model_metadata": reference(ROOT / "artifacts/B1/best_fp32.metadata.json"),
            "tolerance": TOLERANCE, "checks": checks,
            "reference": {"device": "CPU", "dtype": "float32", "source": previous["reference"]["source"],
                "methods": ["non_max_suppression(return_idxs=True,nc=3)", "scale_boxes",
                            "process_mask_native", "scale_masks", "SegmentationPredictor empty-mask rule"],
                "code": [reference(ROOT / "third_party/ultralytics/ultralytics" / path) for path in
                         ("utils/nms.py", "utils/ops.py", "models/yolo/segment/predict.py")],
                "tie_policy": "NumPy stable ascending candidate order; official equal-score ties can be backend-dependent"},
            "guarded_files": guards, "guarded_files_count": len(guards),
            "historical_sha256_and_mtime_unchanged": True,
            "implementation_sha256": {p: digest(ROOT / p) for p in IMPLEMENTATION},
            "scope": {"new_photos": 0, "label_changes": 0, "training": 0, "model_inference": 0,
                      "sealed_test_inference": 0, "new_saved_arrays": 0, "ground_truth_mAP_computed": False,
                      "full_ort_runner_implemented": False},
            "progress": {"M5": "3/6", "main": "27/54", "complete_modules": "3/9",
                         "next": "M5-04 ORT and PyTorch raw reference runners"}}
        write_json(REPORT, report)
    print(json.dumps({"status": "passed", "task": "M5-03", "checks": checks["summary"],
                      "guarded_files": len(guards), "files_written": not (check_only or dry_run)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Recheck accepted files without writing")
    mode.add_argument("--dry-run", action="store_true", help="Compare without creating acceptance receipt")
    args = parser.parse_args()
    run(args.check, args.dry_run)


if __name__ == "__main__":
    main()
