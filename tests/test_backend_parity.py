"""Prevent false parity from reordered outputs, missing instances, masks or sealed inputs."""
from copy import deepcopy
import unittest
import numpy as np

from deploy.parity import assert_binding, compare_instances, score_ties
from deploy.postprocess import ProductInstances
from scripts import check_backend_parity as check


def instances(classes=(0, 1), order=None):
    n = len(classes)
    boxes = np.array([[2 + 10*i, 2, 8 + 10*i, 8] for i in range(n)], np.float32).reshape(n, 4)
    masks = np.zeros((n, 20, 30), np.uint8)
    for i in range(n):
        masks[i, 2:8, 2 + 10*i:8 + 10*i] = 1
    result = ProductInstances(boxes, np.full(n, .8, np.float32), np.array(classes, np.int64),
                              masks, np.zeros((n, 32), np.float32), np.arange(n), {})
    if order is not None:
        for name in ("boxes_xyxy", "scores", "class_ids", "masks", "mask_coefficients", "candidate_indices"):
            setattr(result, name, getattr(result, name)[order])
    return result


class BackendParityTests(unittest.TestCase):
    def test_output_order_does_not_create_false_unmatched_instances(self):
        comparison = compare_instances(instances(order=[1, 0]), instances())
        self.assertTrue(comparison["passed"])
        self.assertEqual(comparison["matched"], 2)
        self.assertEqual(comparison["min_mask_iou"], 1.)

    def test_missing_and_wrong_class_are_not_disguised_by_box_overlap(self):
        for actual in (instances(classes=(0,)), instances(classes=(2, 1))):
            comparison = compare_instances(actual, instances())
            self.assertFalse(comparison["passed"])
            self.assertTrue(comparison["unmatched_reference"])

    def test_box_score_and_mask_tolerances_each_reject_large_errors(self):
        for field in ("boxes_xyxy", "scores", "masks"):
            actual = instances()
            if field == "boxes_xyxy": actual.boxes_xyxy[0, 0] += .02
            elif field == "scores": actual.scores[0] += .001
            else: actual.masks[0, 2, 2] = 0
            with self.subTest(field=field):
                comparison = compare_instances(actual, instances())
                self.assertFalse(comparison["passed"])
        self.assertEqual(comparison["mask_disagreement_pixels"], 1)

    def test_empty_pair_has_no_invented_measured_mask_iou(self):
        result = compare_instances(instances(classes=()), instances(classes=()))
        self.assertTrue(result["passed"])
        self.assertTrue(result["both_empty"])
        self.assertIsNone(result["min_mask_iou"])
        self.assertFalse(compare_instances(instances(classes=()), instances())["passed"])

    def test_equal_score_and_candidate_binding_are_recorded_separately(self):
        raw = np.zeros((1, 39, 3), np.float32)
        raw[0, 4, :2] = .8
        self.assertEqual(score_ties(raw), [{"score": float(np.float32(.8)), "candidates": 2}])
        actual = instances()
        raw[0, 5, 1] = .8
        assert_binding(actual, raw)
        actual.mask_coefficients[0, 0] = 1
        with self.assertRaisesRegex(ValueError, "binding"):
            assert_binding(actual, raw)

    def test_manifest_rejects_sealed_test_even_with_valid_shape(self):
        case = {"source": {"image_id": "sealed", "split": "test", "path": "unused.png", "sha256": "unused"},
                "transform": {"op": "identity"}}
        with self.assertRaisesRegex(ValueError, "non-test"):
            check.materialize(case, {})

    def test_manifest_rejects_duplicate_cases_and_recipe_changes(self):
        manifest = check.read_json(check.MANIFEST)
        for change in ("duplicate", "transform"):
            bad = deepcopy(manifest)
            if change == "duplicate": bad["cases"][1] = deepcopy(bad["cases"][0])
            else: bad["cases"][-1]["transform"]["shape_hw"] = [18, 29]
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "recipe"):
                check.validate_manifest(bad)

    def test_fixed_tensor_hash_rejects_source_or_variant_drift(self):
        case = deepcopy(check.read_json(check.MANIFEST)["cases"][0])
        case["input_sha256"] = "incorrect"
        with self.assertRaisesRegex(ValueError, "input tensor"):
            check.materialize(case, check.allowed_sources())


if __name__ == "__main__":
    unittest.main()
