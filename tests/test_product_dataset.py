"""Temporary engineering fixtures for provenance, sampling budget and joint transforms."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from app.product_data import ROOT, digest, image_write, write_json
from app.product_dataset import (apply_preview, inspect_sample, inspect_transformed, load_policy,
                                 locked_source, make_pipeline, sampling_budget)


class DatasetInventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="yolo_dataset_fixture_")
        self.raw = Path(self.temp.name)
        self.image = self.raw / "frames/group/sample_001.png"
        image_write(self.image, np.zeros((100, 200, 3), np.uint8))
        self.sample = {"image_id": "sample_001", "group_id": "group",
                       "image_path": "frames/group/sample_001.png",
                       "width": 200, "height": 100, "image_sha256": digest(self.image)}
        draft = self.raw / "annotations/drafts/sample_001.json"
        write_json(draft, {"candidates": []})
        self.review = {"image_id": "sample_001", "group_id": "group", "mapping_version": "products-v1",
                       "confirmed": True, "status": "reviewed", "image_sha256": digest(self.image),
                       "draft_sha256": digest(draft), "physical_ids": ["fixture"],
                       "polygons": [{"class_id": 0, "points": [[0.2, 0.2], [0.8, 0.2],
                                                                 [0.8, 0.8], [0.2, 0.8]]}]}
        self.review_path = self.raw / "annotations/reviewed/sample_001.json"
        write_json(self.review_path, self.review)
        self.review_path.with_suffix(".txt").write_text("0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8\n")

    def tearDown(self):
        self.temp.cleanup()

    def test_changed_original_is_rejected(self):
        image_write(self.image, np.full((100, 200, 3), 50, np.uint8))
        with self.assertRaisesRegex(ValueError, "Original image changed"):
            inspect_sample(self.raw, self.sample)

    def test_txt_disagrees_with_human_record(self):
        self.review_path.with_suffix(".txt").write_text("1 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8\n")
        with self.assertRaisesRegex(ValueError, "TXT class/geometry"):
            inspect_sample(self.raw, self.sample)

    def test_unconfirmed_and_false_negative_are_rejected(self):
        for patch in ({"confirmed": False}, {"status": "negative"}):
            write_json(self.review_path, {**self.review, **patch})
            with self.assertRaises(ValueError):
                inspect_sample(self.raw, self.sample)

    def test_legacy_review_metadata_is_derived_without_rewriting_truth(self):
        review_hash = digest(self.review_path)
        self.review.pop("physical_ids")
        write_json(self.review_path, self.review)
        legacy_hash = digest(self.review_path)
        checked = inspect_sample(self.raw, self.sample)
        self.assertEqual(checked["physical_ids"], ["sam_milk_001"])
        self.assertEqual(digest(self.review_path), legacy_hash)
        self.assertNotEqual(review_hash, legacy_hash)

    def test_budget_counts_originals_once_and_cannot_reset_baseline(self):
        for group, ids in (("group", ["sample_001"]), ("bundle", ["sample_001"]), ("new", ["new_001"])):
            write_json(self.raw / "sessions" / (group + ".json"),
                       {"samples": [{"image_id": image_id} for image_id in ids]})
        result = sampling_budget(self.raw, ["sample_001"])
        self.assertEqual(result["additional_real_images_used"], 1)
        self.assertEqual(result["remaining_additional_real_images"], 29)
        with self.assertRaisesRegex(ValueError, "Do not reset"):
            sampling_budget(self.raw, ["sample_001", "new_001"], result)
        write_json(self.raw / "sessions/too_many.json",
                   {"samples": [{"image_id": "more_" + str(i)} for i in range(30)]})
        with self.assertRaisesRegex(ValueError, "budget exceeded"):
            sampling_budget(self.raw, ["sample_001"], result)


class JointTransformTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        locked_source()
        cls.policy = load_policy(ROOT / "configs/augment_products.yaml")

    def test_image_marker_and_polygon_receive_the_same_geometry(self):
        policy = deepcopy(self.policy)
        policy["train_args"].update(hsv_s=0.0, hsv_v=0.0, degrees=0.0, translate=0.0, scale=0.0)
        pipeline = make_pipeline(policy)
        image = np.zeros((360, 640, 3), np.uint8)
        points = np.array([[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]])
        cv2.fillPoly(image, [np.rint(points * [640, 360]).astype(np.int32)], (0, 255, 0))
        result = apply_preview(pipeline, image, [{"class_id": 1, "points": points.tolist()}], 640, 42)
        # A non-square input is centered in the square output. Check pixels against
        # the resulting polygon, not against the transform's own matrix.
        expected_box = [160, 230, 480, 410]
        np.testing.assert_allclose(result["instances"].bboxes[0], expected_box, atol=1e-4)
        marker = (result["img"][..., 1] > 200) & (result["img"][..., 0] < 20)
        mask = np.zeros((640, 640), np.uint8)
        cv2.fillPoly(mask, [np.rint(result["instances"].segments[0]).astype(np.int32)], 1)
        intersection = np.logical_and(marker, mask).sum()
        union = np.logical_or(marker, mask).sum()
        self.assertGreater(intersection / union, 0.99)

    def test_fixed_seed_repeats_but_different_seed_changes_input(self):
        pipeline = make_pipeline(self.policy)
        image = np.full((360, 640, 3), (30, 140, 210), np.uint8)
        polygon = [{"class_id": 0, "points": [[0.3, 0.2], [0.7, 0.2], [0.7, 0.8], [0.3, 0.8]]}]
        a = apply_preview(pipeline, image, polygon, 640, 42)
        b = apply_preview(pipeline, image, polygon, 640, 42)
        c = apply_preview(pipeline, image, polygon, 640, 43)
        self.assertTrue(np.array_equal(a["img"], b["img"]))
        self.assertTrue(np.array_equal(a["instances"].segments, b["instances"].segments))
        self.assertFalse(np.array_equal(a["img"], c["img"]))
        self.assertEqual(inspect_transformed(a, [0])["retained_class_ids"], [0])

    def test_negative_remains_negative_with_new_image_geometry(self):
        pipeline = make_pipeline(self.policy)
        result = apply_preview(pipeline, np.full((360, 640, 3), 80, np.uint8), [], 640, 42)
        check = inspect_transformed(result, [])
        self.assertTrue(check["negative_still_empty"])
        self.assertEqual(check["retained_instances"], 0)
        self.assertEqual(check["output_shape_hw"], [640, 640])


if __name__ == "__main__":
    unittest.main()
