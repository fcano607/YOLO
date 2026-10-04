"""Guard NMS class isolation, coefficient binding and original-coordinate mask boundaries."""
from dataclasses import replace
import unittest

import numpy as np

from deploy.postprocess import native_masks, postprocess_b1, restore_boxes, select_candidates
from deploy.preprocess import preprocess_bgr


def fixtures(n=4, shape=(640, 640)):
    raw = np.zeros((1, 39, n), np.float32)
    proto = np.ones((1, 32, 160, 160), np.float32)
    _, geometry = preprocess_bgr(np.zeros((*shape, 3), np.uint8))
    return raw, proto, geometry


class PostprocessTests(unittest.TestCase):
    def test_nms_keeps_other_classes_and_matching_coefficients(self):
        raw, proto, geometry = fixtures()
        raw[0, :4] = np.array([[100, 101, 100, 500], [100, 100, 100, 500],
                              [80, 80, 80, 50], [80, 80, 80, 50]], np.float32)
        raw[0, 4, :2], raw[0, 5, 2], raw[0, 6, 3] = [.9, .8], .7, .1
        raw[0, 7] = [1, 2, 3, 4]
        original, original_proto = raw.copy(), proto.copy()
        result = postprocess_b1(raw, proto, geometry)
        np.testing.assert_array_equal(result.candidate_indices, [0, 2])
        np.testing.assert_array_equal(result.class_ids, [0, 1])
        np.testing.assert_array_equal(result.mask_coefficients[:, 0], [1, 3])
        self.assertEqual(result.counts['score_passed_locations'], 3)
        self.assertEqual(result.counts['after_nms'], 2)
        np.testing.assert_array_equal(raw, original)
        np.testing.assert_array_equal(proto, original_proto)

    def test_confidence_is_strict_and_iou_equal_to_threshold_is_kept(self):
        raw, _, _ = fixtures(3)
        raw[0, :4, :2] = np.array([[15, 25], [5, 5], [30, 30], [10, 10]], np.float32)
        raw[0, 4] = [.9, .8, .25]
        np.testing.assert_array_equal(select_candidates(raw, iou=.5).candidate_indices, [0, 1])
        np.testing.assert_array_equal(select_candidates(raw, iou=.4999).candidate_indices, [0])
        self.assertEqual(select_candidates(raw).counts['score_passed_locations'], 2)

    def test_equal_scores_use_deterministic_candidate_order(self):
        raw, _, _ = fixtures(3)
        raw[0, :4] = np.array([[100, 100, 400], [100, 100, 400], [80, 80, 80], [80, 80, 80]], np.float32)
        raw[0, 4] = .8
        np.testing.assert_array_equal(select_candidates(raw).candidate_indices, [0, 2])
        np.testing.assert_array_equal(select_candidates(raw, max_det=1).candidate_indices, [0])

    def test_multi_label_repeats_candidate_but_preserves_its_coefficients(self):
        raw, _, _ = fixtures(1)
        raw[0, :4, 0], raw[0, 4:7, 0], raw[0, 7:39, 0] = [200, 200, 80, 80], [.8, .7, .1], np.arange(32)
        single = select_candidates(raw)
        multiple = select_candidates(raw, multi_label=True)
        np.testing.assert_array_equal(single.class_ids, [0])
        np.testing.assert_array_equal(multiple.class_ids, [0, 1])
        np.testing.assert_array_equal(multiple.candidate_indices, [0, 0])
        np.testing.assert_array_equal(multiple.mask_coefficients[0], multiple.mask_coefficients[1])

    def test_empty_candidates_and_empty_masks_have_complete_shapes(self):
        raw, proto, geometry = fixtures(0, (481, 640))
        result = postprocess_b1(raw, proto, geometry)
        self.assertEqual(result.boxes_xyxy.shape, (0, 4))
        self.assertEqual(result.mask_coefficients.shape, (0, 32))
        self.assertEqual(result.masks.shape, (0, 481, 640))
        self.assertEqual(result.masks.dtype, np.uint8)
        raw, proto, geometry = fixtures(1)
        raw[0, :4, 0], raw[0, 4, 0], raw[0, 7, 0] = [320, 320, 100, 100], .8, -1
        result = postprocess_b1(raw, proto, geometry)
        self.assertEqual(result.counts['after_nms'], 1)
        self.assertEqual(result.counts['empty_masks_removed'], 1)
        self.assertEqual(result.masks.shape, (0, 640, 640))

    def test_max_det_precedes_empty_mask_removal_without_refilling(self):
        raw, proto, geometry = fixtures(2)
        raw[0, :4] = np.array([[100, 400], [100, 400], [50, 50], [50, 50]], np.float32)
        raw[0, 4], raw[0, 7] = [.9, .8], [-1, 1]
        result = postprocess_b1(raw, proto, geometry, max_det=1)
        self.assertEqual(result.counts['after_nms'], 1)
        self.assertEqual(result.counts['instances'], 0)

    def test_original_boxes_use_correct_odd_padding_and_clip(self):
        for shape, boxes in (((481, 640), [[-1, 78, 641, 561]]), ((640, 481), [[78, -1, 561, 641]])):
            with self.subTest(shape=shape):
                _, _, geometry = fixtures(0, shape)
                restored = restore_boxes(np.array(boxes, np.float32), geometry)
                np.testing.assert_array_equal(restored, [[0, 0, shape[1], shape[0]]])
        with self.assertRaises(ValueError):
            restore_boxes(np.zeros((1, 4), np.float32), replace(geometry, padding_ltrb=(0, 0, 0, 0)))

    def test_fractional_boxes_crop_pixels_without_rounding_box_coordinates(self):
        _, proto, _ = fixtures(0)
        coeff = np.zeros((1, 32), np.float32)
        coeff[0, 0] = 1
        masks = native_masks(proto, coeff, np.array([[.2, 1.2, 3.8, 4.8]], np.float32), (5, 5))
        expected = np.zeros((1, 5, 5), np.uint8)
        expected[0, 2:5, 1:4] = 1
        np.testing.assert_array_equal(masks, expected)

    def test_zero_logit_is_background_and_coefficients_can_combine_negative_prototypes(self):
        _, proto, _ = fixtures(0)
        proto.fill(0)
        proto[0, 0], proto[0, 1] = -1, 2
        coefficients = np.zeros((2, 32), np.float32)
        coefficients[:, :2] = [[2, 1], [1, 1]]
        masks = native_masks(proto, coefficients, np.array([[0, 0, 5, 5]] * 2, np.float32), (5, 5))
        self.assertFalse(masks[0].any())
        self.assertTrue(masks[1].all())

    def test_invalid_raw_precision_values_and_options_are_rejected(self):
        raw, proto, geometry = fixtures(1)
        for invalid in (raw.astype(np.float16), raw[:, :38], np.full(raw.shape, np.nan, np.float32)):
            with self.assertRaises(ValueError):
                postprocess_b1(invalid, proto, geometry)
        for options in ({'conf': -1}, {'iou': 2}, {'max_det': 0}, {'multi_label': 1}):
            with self.assertRaises(ValueError):
                postprocess_b1(raw, proto, geometry, **options)
        with self.assertRaises(ValueError):
            postprocess_b1(raw, proto[:, :31], geometry)


if __name__ == '__main__':
    unittest.main()
