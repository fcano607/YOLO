"""Catch backend input drift, sealed inputs, cached frames and video cleanup failures."""
from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import yaml

from app import infer_products as app
from deploy.base_backend import ProductBackend, load_bundle, validate_input_tensor, validate_raw_outputs
from deploy.onnx_backend import OnnxBackend


class ToyBackend(ProductBackend):
    def __init__(self):
        bundle = SimpleNamespace(config={"postprocess": {"conf": .25, "iou": .7, "max_det": 300,
                                                        "multi_label": False}},
                                 class_names={0: "sam_whole_milk", 1: "yili_shuhua", 2: "luckin_cup"})
        super().__init__(bundle, "cpu")
        self.calls, self.last_images = 0, None

    def run_raw(self, images):
        validate_input_tensor(images)
        self.calls += 1
        self.last_images = images.copy()
        raw = np.zeros((1, 39, 8400), np.float32)
        raw[0, :4, 0], raw[0, 4, 0], raw[0, 7, 0] = [320, 320, 400, 400], .9, 1
        return raw, np.ones((1, 32, 160, 160), np.float32)

    def describe(self):
        return {"kind": "fixture", "device": "cpu"}


class ProductDeploymentTests(unittest.TestCase):
    def test_both_frames_use_shared_prepost_and_complete_result_interface(self):
        backend = ToyBackend()
        first = backend.predict(np.full((17, 29, 3), (10, 80, 240), np.uint8))
        np.testing.assert_array_equal(backend.last_images[0, :, 320, 320], np.array([240, 80, 10], np.float32) / 255)
        second = backend.predict(np.zeros((29, 17, 3), np.uint8))
        self.assertEqual(backend.calls, 2)
        self.assertEqual(first.instances.masks.shape, (1, 17, 29))
        self.assertEqual(second.instances.masks.shape, (1, 29, 17))
        self.assertEqual(first.to_dict()["instances"][0]["label"], "sam_whole_milk")
        self.assertTrue(all(v >= 0 for v in first.timings_ms.values()))
        backend.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            backend.predict(np.zeros((17, 29, 3), np.uint8))

    def test_input_and_raw_interfaces_reject_wrong_dtype_shape_and_values(self):
        good = np.zeros((1, 3, 640, 640), np.float32)
        validate_input_tensor(good)
        for bad in (good.astype(np.float16), good[:, :, :, ::-1], np.full(good.shape, np.nan, np.float32)):
            with self.assertRaises(ValueError):
                validate_input_tensor(bad)
        with self.assertRaises(ValueError):
            validate_raw_outputs([np.zeros((1, 116, 8400), np.float32), np.zeros((1, 32, 160, 160), np.float32)])

    def test_config_cannot_redirect_to_coco_weights_or_enable_tf32(self):
        from deploy.paths import resolve_path
        original = yaml.safe_load(resolve_path("configs/infer_products.yaml").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.yaml"
            for field in ("weights", "tf32"):
                wrong = deepcopy(original)
                if field == "weights":
                    wrong["weights"] = "artifacts/precheck/E0.pt"
                else:
                    wrong["runtime"]["tf32"] = True
                path.write_text(yaml.safe_dump(wrong), encoding="utf-8")
                with self.subTest(field=field), self.assertRaises(ValueError):
                    load_bundle(path)

    def test_sealed_image_and_identical_copy_are_rejected_before_model_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, copied, external, output = [root / name for name in ("test.png", "copy.png", "external.png", "out.png")]
            source.write_bytes(b"sealed image")
            copied.write_bytes(source.read_bytes())
            external.write_bytes(b"development image")
            sample = {"split": "test", "image_path": "test.png", "staged_image_path": "staged.png",
                      "image_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
            with patch.object(app, "ROOT", root), patch.object(app, "read_json", return_value={"samples": [sample]}):
                for value in (source, copied):
                    with self.assertRaisesRegex(ValueError, "sealed"):
                        app.check_io(value, output)
                self.assertEqual(app.check_io(external, output), (external.resolve(), output.resolve()))

    def test_existing_output_blocks_before_backend_initialization(self):
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / "in.png", Path(temp) / "out.png"
            source.write_bytes(b"input")
            output.write_bytes(b"existing result")
            args = SimpleNamespace(source=str(source), output=str(output))
            with patch.object(app, "create_backend") as create:
                with self.assertRaises(FileExistsError):
                    app.run(args)
                create.assert_not_called()
            self.assertEqual(output.read_bytes(), b"existing result")

    def test_video_uses_each_frame_once_and_releases_resources_at_eof(self):
        frame = np.zeros((17, 29, 3), np.uint8)
        capture = Mock()
        capture.isOpened.return_value, capture.get.return_value = True, 5
        capture.read.side_effect = [(True, frame.copy())] * 3 + [(False, None)]
        writer = Mock()
        writer.isOpened.return_value = True
        backend = ToyBackend()
        with tempfile.TemporaryDirectory() as temp, patch.object(app.cv2, "VideoCapture", return_value=capture), \
                patch.object(app.cv2, "VideoWriter", return_value=writer):
            result = app.process_video(backend, "input.avi", Path(temp) / "output.avi")
        self.assertEqual(backend.calls, 3)
        self.assertEqual(writer.write.call_count, 3)
        self.assertEqual(result["detections_per_frame"], [1, 1, 1])
        self.assertEqual(result["exit_reason"], "end_of_file")
        capture.release.assert_called_once()
        writer.release.assert_called_once()

    def test_video_inference_exception_still_releases_capture_and_writer(self):
        capture = Mock()
        capture.isOpened.return_value, capture.get.return_value = True, 5
        capture.read.return_value = (True, np.zeros((17, 29, 3), np.uint8))
        writer = Mock()
        writer.isOpened.return_value = True
        backend = Mock()
        backend.predict.side_effect = RuntimeError("inference failed")
        with tempfile.TemporaryDirectory() as temp, patch.object(app.cv2, "VideoCapture", return_value=capture), \
                patch.object(app.cv2, "VideoWriter", return_value=writer):
            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                app.process_video(backend, "input.avi", Path(temp) / "output.avi")
        capture.release.assert_called_once()
        writer.release.assert_called_once()

    def test_registered_cuda_provider_does_not_hide_cpu_fallback(self):
        session = Mock()
        session.get_providers.return_value = ["CPUExecutionProvider"]
        ort = SimpleNamespace(__version__="fixture", SessionOptions=SimpleNamespace,
                              get_available_providers=lambda: ["CUDAExecutionProvider"],
                              InferenceSession=Mock(return_value=session))
        bundle = SimpleNamespace(config={"runtime": {"cpu_threads": 4}}, onnx=Path("model.onnx"))
        with patch.dict("sys.modules", {"onnxruntime": ort}), \
                patch("deploy.onnx_backend.cuda_dll_directories", return_value=([], [])):
            with self.assertRaisesRegex(RuntimeError, "fell back to CPU"):
                OnnxBackend(bundle, "cuda:0")
        session.disable_fallback.assert_not_called()


if __name__ == "__main__":
    unittest.main()
