"""B1 provenance and a single shared pre/raw/post pipeline; framework imports are lazy."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import yaml

from deploy.paths import PROJECT_ROOT, resolve_path
from deploy.postprocess import postprocess_b1, validate_postprocessing_contract
from deploy.preprocess import preprocess_bgr, validate_preprocessing_contract
from deploy.results import InferenceResult


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass
class DeploymentBundle:
    config: dict
    metadata: dict
    baseline_id: str
    onnx: Path
    weights: Path
    class_names: dict


def load_bundle(config_path="configs/infer_products.yaml"):
    config = yaml.safe_load(resolve_path(config_path).read_text(encoding="utf-8"))
    if (config.get("schema_version") != 1 or config.get("purpose") != "independent_b1_product_inference" or
            config.get("baseline") != "configs/baseline_products_v1.json" or
            config.get("export_receipt") != "reports/deployment/M5-01_B1_onnx_export.json" or
            config.get("postprocess") != {"conf": .25, "iou": .7, "max_det": 300, "multi_label": False} or
            config.get("runtime") != {"cpu_threads": 4, "cuda_device": 0, "tf32": False} or
            config.get("defaults") != {"backend": "onnx", "device": "cuda:0"}):
        raise ValueError("Expected the frozen B1 independent-inference configuration")
    receipt = read_json(resolve_path(config["export_receipt"]))
    baseline = read_json(resolve_path(config["baseline"]))
    metadata = read_json(resolve_path(config["metadata"]))
    if receipt["status"] != "completed" or receipt["baseline_id"] != baseline["baseline_id"]:
        raise ValueError("B1/export receipt mismatch")
    references = [receipt["baseline"], receipt["artifacts"]["metadata"], receipt["artifacts"]["onnx"],
                  baseline["selected_model"]["checkpoint"]]
    expected_paths = [config[k] for k in ("baseline", "metadata", "onnx", "weights")]
    if [r["path"] for r in references] != expected_paths:
        raise ValueError("Deployment must use the accepted B1 model, metadata and weight paths")
    for reference in references:
        if file_sha256(resolve_path(reference["path"])) != reference["sha256"]:
            raise ValueError("Deployment asset changed: " + reference["path"])
    if (metadata["source_weights"] != references[-1] or metadata["baseline_id"] != baseline["baseline_id"] or
            metadata["class_names"] != baseline["selected_model"]["names"]):
        raise ValueError("Model provenance / class order differs from B1")
    validate_preprocessing_contract(metadata)
    validate_postprocessing_contract(metadata)
    return DeploymentBundle(config, metadata, baseline["baseline_id"], resolve_path(config["onnx"]),
                            resolve_path(config["weights"]), {int(k): v for k, v in metadata["class_names"].items()})


def validate_input_tensor(images):
    if (not isinstance(images, np.ndarray) or images.shape != (1, 3, 640, 640) or
            images.dtype != np.float32 or not images.flags.c_contiguous or not np.isfinite(images).all() or
            images.min() < 0 or images.max() > 1):
        raise ValueError("Expected contiguous normalized RGB FP32 [1,3,640,640]")


def validate_raw_outputs(outputs):
    if not isinstance(outputs, (list, tuple)) or len(outputs) != 2:
        raise ValueError("Expected both B1 raw candidate and prototype outputs")
    for value, shape in zip(outputs, ((1, 39, 8400), (1, 32, 160, 160))):
        if (not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != np.float32 or
                not np.isfinite(value).all()):
            raise ValueError("Raw model output violates the B1 static FP32 contract")
    return tuple(outputs)


class ProductBackend(ABC):
    def __init__(self, bundle, device):
        if device not in ("cpu", "cuda:0"):
            raise ValueError("Supported devices: cpu or cuda:0")
        self.bundle, self.device, self.closed = bundle, device, False

    @abstractmethod
    def run_raw(self, images):
        """Return NumPy FP32 candidates/prototypes, including device transfer completion."""

    @abstractmethod
    def describe(self):
        """Runtime identity; provider registration alone is not node-execution evidence."""

    def predict(self, frame):
        if self.closed:
            raise RuntimeError("Backend is closed")
        start = perf_counter()
        images, geometry = preprocess_bgr(frame)
        pre_end = perf_counter()
        raw = validate_raw_outputs(self.run_raw(images))
        raw_end = perf_counter()
        instances = postprocess_b1(*raw, geometry, **self.bundle.config["postprocess"])
        post_end = perf_counter()
        times = {"preprocess": (pre_end - start) * 1000, "raw_with_transfers": (raw_end - pre_end) * 1000,
                 "postprocess": (post_end - raw_end) * 1000, "core_total": (post_end - start) * 1000}
        return InferenceResult(instances, geometry, self.bundle.class_names, self.describe(), times)

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def create_backend(bundle, kind=None, device=None, profile_prefix=None):
    kind = kind or bundle.config["defaults"]["backend"]
    device = device or bundle.config["defaults"]["device"]
    if kind == "onnx":
        from deploy.onnx_backend import OnnxBackend
        return OnnxBackend(bundle, device, profile_prefix)
    if kind == "torch":
        if profile_prefix is not None:
            raise ValueError("ORT node profiling belongs to the ONNX backend")
        from deploy.torch_backend import TorchBackend
        return TorchBackend(bundle, device)
    raise ValueError("Supported backends: onnx or torch")
