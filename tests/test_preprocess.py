"""Check color, asymmetric padding, geometry and input-contract mistakes."""
from copy import deepcopy
import unittest

import numpy as np

from deploy.preprocess import LETTERBOX_OPTIONS, preprocess_bgr, validate_preprocessing_contract


class PreprocessTests(unittest.TestCase):
    def test_bgr_to_rgb_normalized_once_and_no_input_mutation(self):
        image = np.full((640, 640, 3), (10, 80, 240), dtype=np.uint8)
        original = image.copy()
        tensor, geometry = preprocess_bgr(image)
        np.testing.assert_array_equal(tensor[0, :, 123, 234], np.array([240, 80, 10], np.float32) / 255)
        np.testing.assert_array_equal(image, original)
        self.assertEqual(tensor.shape, (1, 3, 640, 640))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(tensor.flags.c_contiguous)
        self.assertEqual(geometry.padding_ltrb, (0, 0, 0, 0))

    def test_landscape_portrait_and_upscale_geometry(self):
        for shape, content, padding, ratio in (
                ((720, 1280), (360, 640), (0, 140, 0, 140), .5),
                ((1280, 720), (640, 360), (140, 0, 140, 0), .5),
                ((1, 1), (640, 640), (0, 0, 0, 0), 640.)):
            with self.subTest(shape=shape):
                tensor, geometry = preprocess_bgr(np.zeros((*shape, 3), np.uint8))
                self.assertEqual(geometry.original_shape_hw, shape)
                self.assertEqual(geometry.resized_shape_hw, content)
                self.assertEqual(geometry.padding_ltrb, padding)
                self.assertEqual(geometry.ratio_xy, (ratio, ratio))
                self.assertEqual(tensor.shape, (1, 3, 640, 640))
                self.assertEqual(geometry.to_dict()["original_shape_hw"], list(shape))

    def test_odd_padding_on_correct_sides_and_ideal_gain_preserved(self):
        for shape, padding in (((481, 640), (0, 79, 0, 80)), ((640, 481), (79, 0, 80, 0))):
            with self.subTest(shape=shape):
                tensor, geometry = preprocess_bgr(np.zeros((*shape, 3), np.uint8))
                self.assertEqual(geometry.padding_ltrb, padding)
                left, top, right, bottom = padding
                self.assertTrue(np.all(tensor[:, :, top:640-bottom, left:640-right] == 0))
                self.assertEqual(tensor[0, 0, 0, 0], np.float32(114) / np.float32(255))
        _, geometry = preprocess_bgr(np.zeros((333, 1001, 3), np.uint8))
        self.assertEqual(geometry.ratio_xy, (640 / 1001, 640 / 1001))
        self.assertNotEqual(geometry.ratio_xy[1], geometry.resized_shape_hw[0] / 333)

    def test_noncontiguous_readonly_input_is_supported(self):
        image = np.arange(19 * 31 * 3, dtype=np.uint8).reshape(19, 31, 3)[:, ::-1]
        image.setflags(write=False)
        original = image.copy()
        tensor, geometry = preprocess_bgr(image)
        expected, reference_geometry = preprocess_bgr(original)
        np.testing.assert_array_equal(tensor, expected)
        np.testing.assert_array_equal(image, original)
        self.assertEqual(geometry, reference_geometry)
        self.assertTrue(tensor.flags.c_contiguous)

    def test_invalid_image_and_incompatible_metadata_are_rejected(self):
        for image in (None, np.zeros((3, 3), np.uint8), np.zeros((3, 3, 4), np.uint8),
                      np.zeros((3, 3, 3), np.float32), np.zeros((0, 3, 3), np.uint8),
                      np.zeros((1, 2000, 3), np.uint8)):
            with self.subTest(shape=getattr(image, "shape", None)), self.assertRaises(ValueError):
                preprocess_bgr(image)
        metadata = {"graph": {"inputs": [{"name": "images", "shape": [1, 3, 640, 640], "onnx_dtype": 1}]},
                    "preprocessing_contract": {"input": "BGR uint8 HWC",
                        "model_input": "RGB float32 NCHW / 255, contiguous", "letterbox": deepcopy(LETTERBOX_OPTIONS)}}
        validate_preprocessing_contract(metadata)
        for field in ("auto", "shape", "dtype"):
            wrong = deepcopy(metadata)
            if field == "auto":
                wrong["preprocessing_contract"]["letterbox"]["auto"] = True
            elif field == "shape":
                wrong["graph"]["inputs"][0]["shape"][-1] = "width"
            else:
                wrong["graph"]["inputs"][0]["onnx_dtype"] = 10
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_preprocessing_contract(wrong)


if __name__ == "__main__":
    unittest.main()
