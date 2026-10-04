"""B1 PyTorch raw reference, sharing all NumPy/OpenCV pre/postprocessing with ORT."""
from deploy.base_backend import ProductBackend, validate_input_tensor, validate_raw_outputs


class TorchBackend(ProductBackend):
    def __init__(self, bundle, device="cuda:0"):
        super().__init__(bundle, device)
        import torch
        from app.product_dataset import locked_source
        from app.product_experiment import verify_product_model
        from ultralytics import YOLO
        from ultralytics.nn.modules import C2f, Detect
        self.torch, self.source = torch, locked_source()
        if device == "cuda:0" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use --device cpu explicitly")
        self._old_tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        self.network = YOLO(str(bundle.weights), task="segment").model.to(device).float().eval()
        verify_product_model(self.network)
        self.network.fuse(imgsz=(640, 640), verbose=False)
        for module in self.network.modules():
            if isinstance(module, Detect):
                module.dynamic, module.export, module.format = False, True, "onnx"
                module.xyxy, module.shape = False, None
            elif isinstance(module, C2f):
                module.forward = module.forward_split

    def run_raw(self, images):
        if self.closed:
            raise RuntimeError("Backend is closed")
        validate_input_tensor(images)
        with self.torch.inference_mode():
            outputs = self.network(self.torch.from_numpy(images).to(self.device))
        return validate_raw_outputs(tuple(v.detach().cpu().numpy() for v in outputs))

    def describe(self):
        return {"kind": "torch", "device": self.device, "version": self.torch.__version__,
                "baseline_id": self.bundle.baseline_id, "source": self.source, "fp32": True, "tf32": False,
                "gpu": self.torch.cuda.get_device_name(0) if self.device == "cuda:0" else None}

    def close(self):
        if self.closed:
            return
        self.network = None
        self.torch.backends.cuda.matmul.allow_tf32, self.torch.backends.cudnn.allow_tf32 = self._old_tf32
        super().close()
