"""Reject namespace/provenance mistakes and check same-frame camera cleanup."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch
from ultralytics.engine.results import Masks

from app import live_camera as live


class ProductPreviewTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.names = {0: 'sam_whole_milk', 1: 'yili_shuhua', 2: 'luckin_cup'}
        self.model = SimpleNamespace(task='segment', names=self.names,
                                     model=SimpleNamespace(model=[SimpleNamespace(nc=3, proto=True)]))
        metadata = {'status': 'frozen', 'mapping_version': 'products-v1',
                    'classes': [{'id': k, 'name': v} for k, v in self.names.items()]}
        weights = {'path': 'best.pt', 'sha256': 'saved-best-hash'}
        report = {'status': 'completed', 'experiment': 'E1-A',
                  'training': {'checkpoints': {'best': weights}}}
        self.config = {'model_kind': 'products', 'experiment': 'E1-A', 'weights': weights,
                       'camera': {'index': 0, 'backend': 'DSHOW', 'timeout': 60},
                       'predict': {'conf': .1}}
        for key, value in [('class_metadata', metadata), ('training_report', report)]:
            path = self.root / (key + '.json')
            path.write_text(json.dumps(value), encoding='utf-8')
            self.config[key] = {'path': str(path), 'sha256': live.hashlib.sha256(path.read_bytes()).hexdigest()}

    def test_products_use_entire_three_class_network(self):
        self.assertIsNone(live.verify_model_contract(self.config, self.model))
        self.model.names = {0: 'yili_shuhua', 1: 'sam_whole_milk', 2: 'luckin_cup'}
        with self.assertRaisesRegex(ValueError, 'three-product'):
            live.verify_model_contract(self.config, self.model)

    def test_stale_metadata_and_wrong_best_checkpoint_are_rejected(self):
        for key in ('class_metadata', 'training_report'):
            with self.subTest(reference=key):
                wrong = deepcopy(self.config)
                wrong[key]['sha256'] = 'stale'
                with self.assertRaisesRegex(ValueError, 'changed'):
                    live.verify_model_contract(wrong, self.model)
        wrong = deepcopy(self.config)
        wrong['weights']['path'] = 'last-other-run.pt'
        with self.assertRaisesRegex(ValueError, 'best checkpoint'):
            live.verify_model_contract(wrong, self.model)

    def test_coco_namespace_is_preserved_and_cannot_accept_product_model(self):
        config = {'targets': [{'name': 'cup'}, {'name': 'bottle'}, {'name': 'cell phone'}]}
        names = {i: 'class_' + str(i) for i in range(80)}
        names.update({41: 'cup', 39: 'bottle', 67: 'cell phone'})
        self.assertEqual(live.verify_model_contract(config, SimpleNamespace(task='segment', names=names)), [41, 39, 67])
        with self.assertRaisesRegex(ValueError, '80-class'):
            live.verify_model_contract(config, self.model)

    def test_both_panes_share_current_frame_and_release_on_exit(self):
        frame = np.full((24, 32, 3), 40, dtype=np.uint8)
        annotated = frame.copy()
        annotated[4:12, 5:15] = [0, 255, 0]
        result = SimpleNamespace(names=self.names, boxes=SimpleNamespace(data=torch.tensor([[1., 2., 9., 10., .8, 0.]]),
                                 cls=torch.tensor([0.])), masks=Masks(torch.ones(1, 24, 32), (24, 32)))
        result.plot = lambda **kwargs: annotated.copy()
        self.model.predict = lambda source, **kwargs: [result]
        args = SimpleNamespace(device=None, all_classes=False, camera_index=None, seconds=None, view='both',
                               source=None, report=str(self.root / 'session.json'))
        capture = SimpleNamespace(isOpened=lambda: True, read=lambda: (True, frame.copy()),
                                  release=unittest.mock.Mock())
        shown = []
        with patch.object(live, 'load_model', return_value=(self.model, {'conf': .1})), \
                patch.object(cv2, 'VideoCapture', return_value=capture), \
                patch.object(cv2, 'namedWindow'), patch.object(cv2, 'resizeWindow'), \
                patch.object(cv2, 'imshow', side_effect=lambda name, image: shown.append(image.copy())), \
                patch.object(cv2, 'waitKey', return_value=ord('q')), patch.object(cv2, 'destroyAllWindows') as destroy:
            self.assertEqual(live.camera_worker(args, self.config), 0)
        capture.release.assert_called_once()
        destroy.assert_called_once()
        self.assertTrue(np.array_equal(shown[0][40:, :32], frame))
        self.assertTrue(np.array_equal(shown[0][40:, 32:], annotated))
        report = json.loads(Path(args.report).read_text(encoding='utf-8'))
        self.assertEqual(report['frames_inferred'], 1)
        self.assertEqual(report['frames_with_class'], {'sam_whole_milk': 1})
        self.assertFalse(report['camera_images_or_videos_saved'])
        self.assertEqual(report['exit_reason'], 'key_exit')

    def test_inference_failure_still_closes_window_and_releases_camera(self):
        self.model.predict = unittest.mock.Mock(side_effect=RuntimeError('inference failed'))
        capture = SimpleNamespace(isOpened=lambda: True, read=lambda: (True, np.zeros((24, 32, 3), np.uint8)),
                                  release=unittest.mock.Mock())
        args = SimpleNamespace(device=None, all_classes=False, camera_index=None, seconds=None, view='both',
                               source=None, report=str(self.root / 'failed.json'))
        with patch.object(live, 'load_model', return_value=(self.model, {'conf': .1})), \
                patch.object(cv2, 'VideoCapture', return_value=capture), \
                patch.object(cv2, 'namedWindow'), patch.object(cv2, 'destroyAllWindows') as destroy:
            with self.assertRaisesRegex(RuntimeError, 'inference failed'):
                live.camera_worker(args, self.config)
        capture.release.assert_called_once()
        destroy.assert_called_once()
        self.assertEqual(json.loads(Path(args.report).read_text())['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
