"""Export frozen product B1 once; --check verifies the saved interface and provenance."""
import argparse
from collections import Counter
import contextlib
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from app.product_data import digest, image_read, read_json, write_json
from app.product_dataset import locked_source
from app.product_training import check_guards
from deploy.onnx_contract import compare_raw, inspect_graph, select_smoke_sources
from scripts.freeze_product_baseline import check_baseline


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def relative(path):
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def reference(path):
    return {"path": relative(path), "sha256": digest(path)}


def configuration(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    expected = {"imgsz": 640, "batch": 1, "dynamic": False, "quantize": 32, "opset": 17,
                "simplify": False, "nms": False, "device": "0"}
    if (config["schema_version"] != 1 or config["purpose"] != "frozen_product_onnx_export" or
            config["baseline"] != "configs/baseline_products_v1.json" or
            config["baseline_receipt"] != "reports/experiments/M3-06_products_v1_baseline.json" or
            config["export"] != expected or config["comparison"] != {"atol": .001, "rtol": .0001}):
        raise ValueError("Expected the B1 static FP32 raw-output export configuration")
    with contextlib.redirect_stdout(sys.stderr):
        baseline_check = check_baseline()
    if baseline_check["reference_code_drift"] or baseline_check["reference_environment_drift"]:
        raise ValueError("Review B1 code/package drift before this first export")
    baseline = read_json(ROOT / config["baseline"])
    loading = read_json(ROOT / baseline["references"]["loading"]["path"])
    samples = select_smoke_sources(loading, config["smoke_image_ids"])
    paths = {k: (ROOT / config[k]).resolve() for k in
             ("onnx", "metadata", "reference_arrays", "report", "log", "cuda_profile")}
    if len(set(paths.values())) != len(paths):
        raise ValueError("Each export artifact needs its own path")
    for key, destination in paths.items():
        destination.relative_to(ROOT.resolve())
        if not destination.is_relative_to((ROOT / ("artifacts" if key in {
                "onnx", "metadata", "reference_arrays", "cuda_profile"} else
                "reports/deployment" if key == "report" else "logs/deployment")).resolve()):
            raise ValueError("Export output must stay in its declared artifact/report/log directory")
    return config, baseline, samples, paths


def reference_network(weights):
    from ultralytics import YOLO
    from ultralytics.nn.modules import Detect, C2f
    from app.product_experiment import verify_product_model
    network = YOLO(str(weights), task="segment").model.to("cuda:0").float().eval()
    verify_product_model(network)
    network.fuse(imgsz=(640, 640), verbose=False)
    for module in network.modules():
        if isinstance(module, Detect):
            module.dynamic, module.export, module.format = False, True, "onnx"
            module.xyxy, module.shape = False, None
        elif isinstance(module, C2f):
            module.forward = module.forward_split
    return network


def create_references(network, samples, destination):
    """Official square preprocessing is only a reference here, not the M5-02 implementation."""
    import numpy as np
    import torch
    from ultralytics.data.augment import LetterBox
    arrays, records = {}, []
    letterbox = LetterBox(new_shape=(640, 640), auto=False, scale_fill=False,
                         scaleup=True, center=True, stride=32, padding_value=114)
    for index, sample in enumerate(samples):
        source = ROOT / sample["image_path"]
        if digest(source) != sample["image_sha256"]:
            raise ValueError("Frozen smoke image changed")
        image = image_read(source)
        padded = letterbox(image=image)
        images = np.ascontiguousarray(padded[..., ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        with torch.inference_mode():
            raw = network(torch.from_numpy(images).to("cuda:0"))
        outputs = [v.cpu().numpy() for v in raw]
        if [list(v.shape) for v in outputs] != [[1, 39, 8400], [1, 32, 160, 160]]:
            raise ValueError("PyTorch export-head output contract differs")
        compare_raw(outputs, outputs, {"atol": 0., "rtol": 0.})
        prefix = "case" + str(index)
        arrays.update({prefix + "_images": images, prefix + "_output0": outputs[0], prefix + "_output1": outputs[1]})
        records.append({"image_id": sample["image_id"], "source": reference(source), "split": "val",
                        "status": sample["status"], "original_shape": list(image.shape), "array_prefix": prefix,
                        "input_shape": list(images.shape), "input_sha256": hashlib.sha256(images.tobytes()).hexdigest()})
    np.savez_compressed(destination, **arrays)
    return records


def ort_smoke(paths, cases, tolerance, cuda):
    import numpy as np
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 4, 1
    if cuda:
        # torch has already loaded CUDA/cuDNN DLLs; no PyTorch model is executed by this session.
        options.enable_profiling = True
        options.profile_file_prefix = str(paths["cuda_profile"])
    providers = [("CUDAExecutionProvider", {"device_id": 0, "use_tf32": 0}), "CPUExecutionProvider"] if cuda else ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(paths["onnx"]), sess_options=options, providers=providers)
    if cuda and "CUDAExecutionProvider" not in session.get_providers():
        raise ValueError("CPU fallback alone does not count as a CUDA pass")
    records = []
    with np.load(paths["reference_arrays"], allow_pickle=False) as saved:
        for case in cases:
            prefix = case["array_prefix"]
            images = saved[prefix + "_images"]
            if hashlib.sha256(images.tobytes()).hexdigest() != case["input_sha256"]:
                raise ValueError("Saved input differs from the PyTorch reference")
            actual = session.run(["output0", "output1"], {"images": images})
            comparisons = compare_raw([saved[prefix + "_output0"], saved[prefix + "_output1"]], actual, tolerance)
            records.append({"image_id": case["image_id"], "comparisons": comparisons,
                            "passed": all(v["allclose"] for v in comparisons)})
    result = {"passed": all(v["passed"] for v in records), "version": ort.__version__,
              "session_providers": session.get_providers(), "provider_options": session.get_provider_options(),
              "cases": records, "scope": "Same saved tensors, raw-output smoke only; no NMS, masks, mAP or benchmark."}
    if cuda:
        profile = Path(session.end_profiling())
        counts = Counter(v.get("args", {}).get("provider") for v in read_json(profile)
                         if v.get("cat") == "Node" and v.get("args", {}).get("provider"))
        if not counts["CUDAExecutionProvider"]:
            raise ValueError("No actual CUDA node execution found in the profile")
        result.update(profile=reference(profile), profile_node_events_by_provider=dict(counts))
    if not result["passed"]:
        raise ValueError("Raw output differences exceed the configured tolerance: " + json.dumps(result))
    return result


def check_export(config_path):
    config, baseline, samples, paths = configuration(config_path)
    report, metadata = read_json(paths["report"]), read_json(paths["metadata"])
    if report["status"] != "completed" or report["baseline_id"] != baseline["baseline_id"]:
        raise ValueError("Expected the accepted B1 export report")
    check_guards(report["guarded_files"])
    check_guards(report["implementation_sha256"])
    check_guards({v["path"]: v["sha256"] for v in report["artifacts"].values()})
    if metadata["source_weights"] != baseline["selected_model"]["checkpoint"] or metadata["class_names"] != baseline["selected_model"]["names"]:
        raise ValueError("Saved export provenance differs from B1")
    actual = inspect_graph(paths["onnx"], metadata["class_names"])
    if actual != metadata["graph"]:
        raise ValueError("Saved ONNX graph interface changed")
    if (metadata["export_options"] != config["export"] or metadata["baseline"] != reference(ROOT / config["baseline"]) or
            report["config"] != reference(config_path) or report["cases"] != metadata["smoke_cases"] or
            [s["image_id"] for s in samples] != [s["image_id"] for s in report["cases"]]):
        raise ValueError("Saved export configuration or smoke assignment changed")
    for sample, case in zip(samples, report["cases"]):
        if case["source"] != reference(ROOT / sample["image_path"]) or case["split"] != "val":
            raise ValueError("Saved smoke source differs from the frozen validation pool")
    print({"status": "passed", "baseline_id": baseline["baseline_id"], "onnx": metadata["onnx"],
           "interface": actual["outputs"], "files_written": False, "model_inference": False})
    return report


def export(config_path):
    config, baseline, samples, paths = configuration(config_path)
    if any(paths[k].exists() for k in ("onnx", "metadata", "reference_arrays", "report", "log")):
        raise FileExistsError("Export outputs already exist; use --check or a new output configuration")
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    from app.product_runner import capture_run_log
    from scripts.train import environment_fingerprint
    report = {"schema_version": 1, "task": "M5-01", "status": "running", "started_at": now(),
              "baseline_id": baseline["baseline_id"], "baseline": reference(ROOT / config["baseline"]),
              "config": reference(config_path), "source": locked_source(),
              "environment_fingerprint_before": environment_fingerprint(),
              "guarded_files": {**baseline["guarded_files"], config["baseline"]: digest(ROOT / config["baseline"]),
                                config["baseline_receipt"]: digest(ROOT / config["baseline_receipt"]),
                                relative(config_path): digest(config_path)},
              "implementation_sha256": {p: digest(ROOT / p) for p in
                  ("scripts/export_onnx.py", "deploy/onnx_contract.py", "tests/test_onnx_export.py")},
              "new_training_runs": 0, "optimizer_updates": 0, "new_photos": 0,
              "final_test_images_inferred": 0, "new_pt_copies": 0,
              "log": relative(paths["log"])}
    write_json(paths["report"], report)
    try:
        with paths["log"].open("x", encoding="utf-8") as log, capture_run_log(log), patch(
                "ultralytics.utils.callbacks.add_integration_callbacks", lambda instance: None):
            import onnx
            import torch
            import ultralytics
            from ultralytics import YOLO
            from app.product_experiment import verify_product_model
            torch.set_num_threads(4)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            if not torch.cuda.is_available():
                raise ValueError("Configured CUDA device is unavailable")
            weights = ROOT / baseline["selected_model"]["checkpoint"]["path"]
            wrapper = YOLO(str(weights), task="segment")
            verify_product_model(wrapper.model)
            if wrapper.model.model[-1].nm != 32 or wrapper.model.model[-1].end2end:
                raise ValueError("Expected the non-end2end three-class / 32-prototype B1 head")
            # The locked exporter derives its destination from pt_path. Change only this
            # in-memory naming attribute; never copy, save or edit the original checkpoint.
            wrapper.model.pt_path = str(paths["onnx"].with_suffix(".pt"))
            exported = Path(wrapper.export(format="onnx", **config["export"])).resolve()
            if exported != paths["onnx"]:
                raise ValueError("Exporter returned an unexpected path")
            graph = inspect_graph(exported, baseline["selected_model"]["names"])
            network = reference_network(weights)
            cases = create_references(network, samples, paths["reference_arrays"])
            cpu = ort_smoke(paths, cases, config["comparison"], cuda=False)
            cuda = ort_smoke(paths, cases, config["comparison"], cuda=True)
            metadata = {
                "schema_version": 1, "status": "accepted_raw_interface", "baseline_id": baseline["baseline_id"],
                "baseline": reference(ROOT / config["baseline"]), "source_weights": baseline["selected_model"]["checkpoint"],
                "source": report["source"], "class_names": baseline["selected_model"]["names"], "task": "segment",
                "onnx": reference(exported), "export_options": config["export"], "graph": graph,
                "output0_layout": {"axis_order": "batch, channels, candidate_locations", "boxes": {
                    "channels": [0, 4], "format": "cx,cy,w,h", "units": "pixels in letterboxed 640x640 input"},
                    "class_scores": {"channels": [4, 7], "activation": "sigmoid already applied; do not apply again"},
                    "mask_coefficients": {"channels": [7, 39], "activation": "none"},
                    "separate_objectness": False, "candidates_are_object_count": False},
                "output1_layout": {"axis_order": "batch, prototype_channels, height, width", "activation": "none",
                    "meaning": "32 shared learned mask bases; not final per-instance binary masks"},
                "preprocessing_contract": {"status": "reference_specification_only; M5-02 implementation pending",
                    "input": "BGR uint8 HWC", "model_input": "RGB float32 NCHW / 255, contiguous",
                    "letterbox": {"new_shape": [640, 640], "auto": False, "scale_fill": False, "scaleup": True,
                                  "center": True, "stride": 32, "padding_value": 114, "interpolation": "cv2.INTER_LINEAR"},
                    "geometry": "r=min(640/h,640/w); resize to round(w*r),round(h*r); round(half_pad-0.1)/round(half_pad+0.1). Keep original shape, resize ratio and integer padding."},
                "postprocessing_contract": {"status": "reference_specification_only; M5-03 implementation pending",
                    "display_conf": .25, "metric_conf": .001, "nms_iou": .7, "max_det": 300, "classes": None,
                    "display_multi_label": False, "metric_multi_label": True, "agnostic_nms": False,
                    "boxes": "Decode output is xywh; convert to xyxy before class-aware NMS. Keep candidate indices and matching mask coefficients.",
                    "masks": "Coefficients @ flattened prototypes gives logits. Locked process_mask_native scales logits to original shape, thresholds >0, then crops using original-coordinate boxes; no per-prototype sigmoid.",
                    "retina_masks": True, "empty_mask_filter": "Discard detections whose binary mask is all zero, as the locked predictor does."},
                "smoke_cases": cases, "smoke_tolerance": config["comparison"],
                "reference_role": "Fused PyTorch CUDA FP32 export head and ORT receive identical saved square tensors.",
                "limitations": ["No independent preprocessing/NMS/mask decoding implementation yet.",
                    "No new mAP, camera execution, controlled FPS benchmark or final test evaluation.",
                    "Existing rectangular B1 val/camera settings are not this square deployment profile."]}
            write_json(paths["metadata"], metadata)
            check_guards(report["guarded_files"])
            after = environment_fingerprint()
            if after != report["environment_fingerprint_before"] or locked_source() != report["source"]:
                raise ValueError("Source or package versions changed during export")
            report.update(status="completed", completed_at=now(), environment_fingerprint_after=after,
                          guarded_files_unchanged=True, export_options=config["export"], graph=graph, cases=cases,
                          ort_cpu=cpu, ort_cuda=cuda, raw_comparison_tolerance=config["comparison"],
                          versions={"torch": torch.__version__, "onnx": onnx.__version__, "ultralytics": ultralytics.__version__},
                          reference_device="cuda:0", reference_gpu=torch.cuda.get_device_name(0), reference_tf32=False,
                          artifacts={"onnx": reference(exported), "metadata": reference(paths["metadata"]),
                                     "reference_arrays": reference(paths["reference_arrays"]), "cuda_profile": cuda["profile"]},
                          scope="M5-01 export/interface and raw-output smoke only; M5-02..06 remain pending.")
            write_json(paths["report"], report)
    except Exception as error:
        report.update(status="failed", completed_at=now(), error=repr(error))
        write_json(paths["report"], report)
        raise
    check_export(config_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/export_products.yaml")
    parser.add_argument("--check", action="store_true", help="Read-only artifact/interface verification, no inference")
    args = parser.parse_args()
    path = (ROOT / args.config).resolve()
    check_export(path) if args.check else export(path)
