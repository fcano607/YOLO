"""Meaningful guardrails for independent deployment metrics, negative images and labels."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import numpy as np
import yaml

from app import deployment_evaluation as ev


class DeploymentEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.label="0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8"
        self.truth=ev.parse_labels(self.label,(20,30))

    def test_full_resolution_polygon_mask_agrees_with_pinned_official_rasterizer(self):
        from ultralytics.data.utils import polygon2mask
        points=np.asarray([.2,.2,.8,.2,.8,.8,.2,.8],np.float32).reshape(-1,2)*[30,20]
        official=polygon2mask((20,30),[points.reshape(-1)],color=1,downsample_ratio=1)
        np.testing.assert_array_equal(self.truth.masks[0],official)

    def test_invalid_class_polygon_shape_and_nonfinite_labels_are_rejected(self):
        for label in ("0.5 0 0 1 0 1 1","3 0 0 1 0 1 1","0 0 0 1 0 1",
                      "0 nan 0 1 0 1 1","0 0 0 1.1 0 1 1"):
            with self.subTest(label=label),self.assertRaises(ValueError):ev.parse_labels(label,(20,30))

    def test_policy_rejects_test_split_and_changed_metric_threshold(self):
        with tempfile.TemporaryDirectory() as temporary:
            p=Path(temporary)/"config.yaml"
            for kind in ("split","threshold"):
                config=deepcopy(ev.POLICY)
                if kind=="split":config["split"]="test"
                else:config["metric_postprocess"]["conf"] = .25
                p.write_text(yaml.safe_dump(config),encoding="utf-8")
                with self.assertRaisesRegex(ValueError,"policy"):ev.configuration(p)

    def test_validation_selector_rejects_duplicate_or_new_images(self):
        loading=ev.read_json(ev.ROOT/ev.POLICY["loading_contract"])
        baseline=ev.read_json(ev.ROOT/ev.POLICY["baseline"])
        for kind in ("duplicate","test"):
            bad=deepcopy(loading)
            vals=[s for s in bad["samples"] if s["split"]=="val"]
            if kind=="duplicate":vals[1]["image_id"]=vals[0]["image_id"]
            else:next(s for s in bad["samples"] if s["split"]=="test")["split"]="val"
            with self.assertRaisesRegex(ValueError,"val5"):ev.validation_samples(bad,baseline)

    def test_box_success_does_not_hide_bad_mask_ap(self):
        prediction=deepcopy(self.truth);prediction.masks[:]=0;prediction.masks[:,0:2,0:2]=1
        result=ev.evaluate_instances([("fixture",self.truth,prediction,prediction)],{0:"a",1:"b",2:"c"})
        self.assertGreater(result["metrics"]["metrics/mAP50-95(B)"],.9)
        self.assertEqual(result["metrics"]["metrics/mAP50-95(M)"],0.)
        self.assertEqual(result["fixed_display_threshold"]["overall"]["mask"]["fn"],1)

    def test_negative_image_false_positive_is_kept_in_evaluation_counts(self):
        negative=ev.parse_labels("",(20,30))
        result=ev.evaluate_instances([("positive",self.truth,self.truth,self.truth),
                                     ("negative",negative,self.truth,self.truth)],{0:"a",1:"b",2:"c"})
        counts=result["fixed_display_threshold"]["overall"]["box"]
        self.assertEqual(result["images"],2);self.assertEqual(counts["fp"],1)
        self.assertEqual(counts["precision"],.5);self.assertEqual(counts["recall"],1.)

    def test_packed_masks_preserve_pixels_and_empty_original_shape(self):
        for item in (self.truth,ev.parse_labels("",(20,30))):
            restored=ev.unpack_instances(ev.pack_instances(item,"test_"),"test_")
            np.testing.assert_array_equal(restored.masks,item.masks)
            np.testing.assert_array_equal(restored.boxes_xyxy,item.boxes_xyxy)
            self.assertEqual(restored.masks.shape,item.masks.shape)

    def test_predefined_deployment_drop_limit_rejects_loss_and_nonfinite_metrics(self):
        reference={"metrics/mAP50-95(B)":.8,"metrics/mAP50-95(M)":.7}
        self.assertTrue(ev.metric_deltas(reference,reference)["passed"])
        changed={**reference,"metrics/mAP50-95(M)":.694}
        self.assertFalse(ev.metric_deltas(reference,changed)["passed"])
        with self.assertRaises(ValueError):ev.metric_deltas(reference,{**reference,"metrics/mAP50-95(M)":float('nan')})


if __name__=="__main__":unittest.main()
