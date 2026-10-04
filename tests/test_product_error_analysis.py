"""Guard diagnostic instance counts against duplicates, class confusion and test leakage."""
import unittest
import numpy as np

from app.product_error_analysis import analysis_samples, box_iou, instance_counts, match_instances


class ErrorAnalysisTests(unittest.TestCase):
    def test_each_truth_matches_once_and_low_threshold_adds_duplicate_fp(self):
        ious = np.array([[.9], [.8]])
        high = instance_counts(ious, [0, 0], [.8, .1], [0], .25)["per_class"][0]
        low = instance_counts(ious, [0, 0], [.8, .1], [0], .05)["per_class"][0]
        self.assertEqual((high["tp"], high["fp"], high["fn"]), (1, 0, 0))
        self.assertEqual((low["tp"], low["fp"], low["fn"]), (1, 1, 0))

    def test_wrong_class_is_not_a_true_positive_but_spatial_match_can_find_it(self):
        counted = instance_counts([[.9]], [1], [.8], [0], .25)["per_class"]
        self.assertEqual((counted[0]["tp"], counted[0]["fn"], counted[1]["fp"]), (0, 1, 1))
        self.assertEqual(len(match_instances([[.9]], [1], [.8], [0], .25, False)), 1)

    def test_box_success_does_not_imply_mask_success(self):
        b = instance_counts([[.8]], [1], [.8], [1], .25)["per_class"][1]
        m = instance_counts([[.2]], [1], [.8], [1], .25)["per_class"][1]
        self.assertEqual((b["tp"], m["tp"], m["fp"], m["fn"]), (1, 0, 1, 1))

    def test_empty_targets_and_empty_predictions_are_supported(self):
        self.assertEqual(box_iou([], []).shape, (0, 0))
        self.assertEqual(box_iou([[0, 0, 1, 1]], []).shape, (1, 0))
        neg = instance_counts(np.empty((1, 0)), [2], [.6], [], .25)["per_class"][2]
        missed = instance_counts(np.empty((0, 1)), [], [], [2], .25)["per_class"][2]
        self.assertEqual((neg["fp"], missed["fn"]), (1, 1))

    def test_only_declared_train_val_are_selected_even_from_historical_folders(self):
        samples = [{"image_id": str(i), "split": "train", "image_path": "old/val/" + str(i)} for i in range(20)]
        samples += [{"image_id": "v" + str(i), "split": "val"} for i in range(5)]
        samples += [{"image_id": "t" + str(i), "split": "test"} for i in range(5)]
        pools = analysis_samples({"samples": samples})
        self.assertEqual({k: len(v) for k, v in pools.items()}, {"train": 20, "val": 5})
        self.assertFalse(any(s["split"] == "test" for p in pools.values() for s in p))
        samples[0]["split"] = "test"
        with self.assertRaises(ValueError):
            analysis_samples({"samples": samples})
