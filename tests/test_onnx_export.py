"""Catch COCO/dynamic/FP16 export mistakes, sealed inputs and raw comparison errors."""
from copy import deepcopy
import unittest

import numpy as np

from deploy.onnx_contract import compare_raw, select_smoke_sources, validate_interface


class OnnxExportTests(unittest.TestCase):
    def setUp(self):
        self.inputs = [{"name": "images", "shape": [1, 3, 640, 640], "onnx_dtype": 1}]
        self.outputs = [{"name": "output0", "shape": [1, 39, 8400], "onnx_dtype": 1},
                        {"name": "output1", "shape": [1, 32, 160, 160], "onnx_dtype": 1}]
        self.names = {"0": "sam_whole_milk", "1": "yili_shuhua", "2": "luckin_cup"}
        self.embedded = {"task": "segment", "names": str({int(k): v for k, v in self.names.items()}),
                         "imgsz": "[640, 640]", "batch": "1", "end2end": "False"}

    def test_raw_contract_rejects_coco_dynamic_and_fp16(self):
        validate_interface(self.inputs, self.outputs, self.embedded, ["Conv"], self.names)
        for field in ("coco", "dynamic", "fp16"):
            outputs = deepcopy(self.outputs)
            if field == "coco":
                outputs[0]["shape"][1] = 116
            elif field == "dynamic":
                outputs[0]["shape"][2] = "anchors"
            else:
                outputs[0]["onnx_dtype"] = 10
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "static"):
                validate_interface(self.inputs, outputs, self.embedded, ["Conv"], self.names)

    def test_nms_and_wrong_embedded_class_order_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "NMS"):
            validate_interface(self.inputs, self.outputs, self.embedded, ["NonMaxSuppression"], self.names)
        embedded = dict(self.embedded, names=str({0: "yili_shuhua", 1: "sam_whole_milk", 2: "luckin_cup"}))
        with self.assertRaisesRegex(ValueError, "mapping"):
            validate_interface(self.inputs, self.outputs, embedded, ["Conv"], self.names)

    def test_smoke_selector_rejects_test_training_and_duplicate_sources(self):
        loading = {"samples": [{"image_id": role, "split": role} for role in ("train", "val", "test")]}
        self.assertEqual(select_smoke_sources(loading, ["val"]), [loading["samples"][1]])
        for ids in (["test"], ["train"], ["val", "val"], []):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                select_smoke_sources(loading, ids)

    def test_raw_comparison_counts_drift_and_rejects_nonfinite_or_wrong_precision(self):
        ref = [np.ones((1, 2), dtype=np.float32), np.zeros((1, 3), dtype=np.float32)]
        tolerance = {"atol": .001, "rtol": .0001}
        self.assertTrue(all(r["allclose"] for r in compare_raw(ref, ref, tolerance)))
        actual = [v.copy() for v in ref]
        actual[1][0, 0] = .01
        self.assertEqual(compare_raw(ref, actual, tolerance)[1]["outside_tolerance"], 1)
        for bad in (np.full((1, 2), np.nan, dtype=np.float32), ref[0].astype(np.float16)):
            with self.assertRaises(ValueError):
                compare_raw(ref, [bad, ref[1]], tolerance)


if __name__ == "__main__":
    unittest.main()
