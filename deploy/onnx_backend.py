"""Independent ORT execution; neither PyTorch nor Ultralytics is imported."""
import importlib.util
import os
from pathlib import Path
import sys

from deploy.base_backend import ProductBackend, validate_input_tensor, validate_raw_outputs


def cuda_dll_directories():
    """Find existing Windows runtime DLLs without importing the torch Python package."""
    if os.name != "nt":
        return [], []
    candidates = [Path(sys.prefix) / "Library/bin"]
    spec = importlib.util.find_spec("torch")
    if spec and spec.origin:
        candidates.insert(0, Path(spec.origin).parent / "lib")
    if os.environ.get("CUDA_PATH"):
        candidates.append(Path(os.environ["CUDA_PATH"]) / "bin")
    directories = [p.resolve() for p in candidates if p.is_dir()]
    # ORT 1.19's provider loader also needs the process PATH, not only Python's DLL directories.
    old_path = os.environ.get("PATH", "")
    prefixes = [str(p) for p in directories if str(p).lower() not in old_path.lower().split(os.pathsep)]
    if prefixes:
        os.environ["PATH"] = os.pathsep.join(prefixes + [old_path])
    handles = [os.add_dll_directory(str(p)) for p in directories]
    return [str(p) for p in directories], handles


class OnnxBackend(ProductBackend):
    def __init__(self, bundle, device="cpu", profile_prefix=None):
        super().__init__(bundle, device)
        import onnxruntime as ort
        self.version, self.profile_path = ort.__version__, None
        self.dll_directories, self._dll_handles = [], []
        if device == "cuda:0":
            self.dll_directories, self._dll_handles = cuda_dll_directories()
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("CUDA provider is unavailable; use --device cpu explicitly")
        options = ort.SessionOptions()
        options.intra_op_num_threads = bundle.config["runtime"]["cpu_threads"]
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        self.profiling = profile_prefix is not None
        if self.profiling:
            prefix = Path(profile_prefix)
            prefix.parent.mkdir(parents=True, exist_ok=True)
            options.enable_profiling, options.profile_file_prefix = True, str(prefix)
        providers = [("CUDAExecutionProvider", {"device_id": 0, "use_tf32": 0}), "CPUExecutionProvider"] if (
            device == "cuda:0") else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(bundle.onnx), sess_options=options, providers=providers)
        if device == "cuda:0" and "CUDAExecutionProvider" not in self.session.get_providers():
            self.close()
            raise RuntimeError("Requested CUDA but ORT fell back to CPU; CUDA deployment is not accepted")
        self.session.disable_fallback()
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if ([(v.name, v.shape, v.type) for v in inputs] != [("images", [1, 3, 640, 640], "tensor(float)")] or
                [(v.name, v.shape, v.type) for v in outputs] != [
                    ("output0", [1, 39, 8400], "tensor(float)"),
                    ("output1", [1, 32, 160, 160], "tensor(float)")]):
            self.close()
            raise ValueError("ORT session interface differs from B1")

    def run_raw(self, images):
        if self.closed:
            raise RuntimeError("Backend is closed")
        validate_input_tensor(images)
        return validate_raw_outputs(self.session.run(["output0", "output1"], {"images": images}))

    def describe(self):
        return {"kind": "onnx", "device": self.device, "version": self.version,
                "baseline_id": self.bundle.baseline_id, "session_providers": self.session.get_providers(),
                "provider_options": self.session.get_provider_options(), "cuda_dll_directories": self.dll_directories}

    def close(self):
        if self.closed:
            return
        if getattr(self, "session", None) is not None:
            if self.profiling:
                self.profile_path = self.session.end_profiling()
            self.session = None
        for handle in self._dll_handles:
            handle.close()
        super().close()
