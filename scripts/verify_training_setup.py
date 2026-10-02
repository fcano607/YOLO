"""Check the M0-02 training installation without weights, datasets or a camera."""

import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIG_DIR = ROOT / "logs/ultralytics_settings"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(CONFIG_DIR)
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ["YOLO_OFFLINE"] = "true"


def check_training_setup():
    """Return the verified training setup; raise if a required check fails."""
    import cv2
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO
    from packaging.requirements import Requirement

    from deploy.paths import load_paths

    paths = load_paths()
    ultralytics.settings.update({"datasets_dir": str(paths["data"]), "weights_dir": str(paths["pretrained"]),
                                 "runs_dir": str(paths["runs"]), "sync": False})
    lock = json.loads((ROOT / "configs/source-lock.json").read_text(encoding="utf-8"))
    source = paths["source"]
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    assert commit == lock["commit"], (commit, lock["commit"])
    assert ultralytics.__version__ == lock["version"]
    assert Path(ultralytics.__file__).resolve() == source / "ultralytics/__init__.py"
    distribution = metadata.distribution("ultralytics")
    direct_url = json.loads(distribution.read_text("direct_url.json"))
    assert direct_url["dir_info"]["editable"] is True
    assert all(path.exists() for path in paths.values()), "A configured directory is missing"
    pinned_packages = {}
    for line in (ROOT / "requirements-train.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "-")):
            continue
        requirement = Requirement(line)
        installed = metadata.version(requirement.name)
        assert installed in requirement.specifier, f"{requirement}: installed {installed}"
        pinned_packages[requirement.name] = {"required": str(requirement.specifier), "installed": installed}
    pip_check = subprocess.run([sys.executable, "-m", "pip", "check"], text=True, capture_output=True)
    assert pip_check.returncode == 0, pip_check.stdout + pip_check.stderr

    image = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
    encoded_ok, encoded = cv2.imencode(".png", image)
    assert encoded_ok
    assert np.array_equal(image, cv2.imdecode(encoded, cv2.IMREAD_COLOR))
    assert torch.cuda.is_available(), "This workstation's CUDA validation requires a GPU"

    # A YAML creates a randomly initialized model; it does not download pretrained weights.
    wrapper = YOLO(str(source / "ultralytics/cfg/models/11/yolo11n-seg.yaml"), task="segment")
    model = wrapper.model.to("cuda:0").eval()
    assert model.yaml["scale"] == "n"
    assert model.model[-1].__class__.__name__ == "Segment"
    assert model.model[-1].nc == 80
    torch.manual_seed(42)
    with torch.inference_mode():
        output = model(torch.rand(1, 3, 640, 640, device="cuda:0"))
    torch.cuda.synchronize()
    tensors = {}

    def collect(value, key="output"):
        if isinstance(value, torch.Tensor):
            assert value.is_cuda and torch.isfinite(value).all().item(), key
            tensors[key] = {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
        elif isinstance(value, dict):
            for name, child in value.items():
                collect(child, f"{key}.{name}")
        elif isinstance(value, (tuple, list)):
            for index, child in enumerate(value):
                collect(child, f"{key}[{index}]")

    collect(output)
    assert tensors
    packages = {dist.metadata["Name"]: dist.version for dist in metadata.distributions()}
    source_files = ["pyproject.toml", "ultralytics/cfg/models/11/yolo11-seg.yaml", "ultralytics/nn/modules/head.py"]
    report = {
        "task": "M0-02", "date": "2026-10-02", "passed": True,
        "python": sys.version, "executable": sys.executable, "prefix": sys.prefix,
        "source": {**lock, "actual_commit": commit, "loaded_file": ultralytics.__file__, "editable": True,
                   "sha256": {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in source_files}},
        "packages": dict(sorted(packages.items(), key=lambda item: item[0].lower())),
        "pip_check": pip_check.stdout.strip(), "opencv_png_round_trip": True,
        "pinned_packages": pinned_packages,
        "gpu": {"name": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)),
                "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()},
        "model": {"initialization": "random", "scale": model.yaml["scale"], "head": "Segment",
                  "classes": 80, "parameters": sum(p.numel() for p in model.parameters()),
                  "input_shape": [1, 3, 640, 640], "outputs": tensors},
        "paths": {name: str(path) for name, path in paths.items()},
        "settings_file": str(CONFIG_DIR / "Ultralytics/settings.json"),
        "not_checked": ["pretrained E0", "training/backward", "camera", "ONNX", "TensorRT"],
    }
    return report


def main():
    report = check_training_setup()
    report_path = ROOT / "logs/environment/M0-02_20261002_install_snapshot.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "report": str(report_path), "source": report["source"]["actual_commit"],
                      "gpu": report["gpu"]["name"], "model": report["model"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
