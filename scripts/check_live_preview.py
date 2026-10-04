"""Exercise real GPU segmentation and simulated camera/GUI lifecycle without opening a camera."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from collections import Counter
from datetime import datetime, timezone, timedelta
import hashlib
import io
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import live_camera as live
import cv2
import numpy as np
import torch
import yaml

config = yaml.safe_load((ROOT / 'configs/e0.yaml').read_text('utf-8'))
source = ROOT / 'demo/input/E0/bus.jpg'
image = cv2.imdecode(np.frombuffer(source.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
assert image is not None
original_hash = hashlib.sha256(image.tobytes()).hexdigest()
model, options = live.load_model(config)
assert options['classes'] == [41, 39, 67]
result = model.predict(image.copy(), **{**options, 'classes': None})[0]
counts = Counter(result.names[int(x)] for x in result.boxes.cls.tolist())
assert counts == {'person': 4, 'bus': 1}
assert result.masks is not None and len(result.masks) == 5
assert result.boxes.data.is_cuda and result.masks.data.is_cuda
annotated = result.plot(color_mode='instance', line_width=2)
canvas = live.make_preview(image, annotated, 'both')
height, width = image.shape[:2]
assert canvas.shape == (height + 40, width * 2, 3)
assert np.array_equal(canvas[40:, :width], image)
assert np.array_equal(canvas[40:, width:], annotated)
assert not np.array_equal(annotated, image)
assert hashlib.sha256(image.tobytes()).hexdigest() == original_hash

class Capture:
    def __init__(self, opened=True):
        self.opened = opened
        self.released = 0
        self.frames = []
    def isOpened(self):
        return self.opened
    def read(self):
        frame = image.copy()
        frame[0, 0] = [len(self.frames) + 1, 77, 99]
        self.frames.append(frame.copy())
        return True, frame
    def release(self):
        self.released += 1

args = SimpleNamespace(device=None, all_classes=True, camera_index=None, seconds=None, view='both')
capture = Capture()
shown = []
keys = iter([ord('1'), ord('2'), ord('3'), ord('q')])
with patch.object(live, 'load_model', return_value=(model, {**options, 'classes': None})), \
     patch.object(cv2, 'VideoCapture', return_value=capture), \
     patch.object(cv2, 'namedWindow'), patch.object(cv2, 'resizeWindow'), \
     patch.object(cv2, 'imshow', side_effect=lambda name, frame: shown.append(frame.copy())), \
     patch.object(cv2, 'waitKey', side_effect=lambda delay: next(keys)), \
     patch.object(cv2, 'getWindowProperty', return_value=1), patch.object(cv2, 'destroyAllWindows') as destroy:
    assert live.camera_worker(args, config) == 0
    assert destroy.call_count == 1
assert len(shown) == 4 and capture.released == 1
assert [x.shape[1] for x in shown] == [2 * width, width, width, 2 * width]
assert np.array_equal(shown[0][40:, :width], capture.frames[0])
assert np.array_equal(shown[1][40:], capture.frames[1])
assert np.array_equal(shown[3][40:, :width], capture.frames[3])

closed = Capture()
with patch.object(live, 'load_model', return_value=(model, options)), \
     patch.object(cv2, 'VideoCapture', return_value=closed), patch.object(cv2, 'namedWindow'), \
     patch.object(cv2, 'resizeWindow'), patch.object(cv2, 'imshow'), \
     patch.object(cv2, 'waitKey', return_value=-1), patch.object(cv2, 'getWindowProperty', return_value=0), \
     patch.object(cv2, 'destroyAllWindows'):
    assert live.camera_worker(args, config) == 0
assert len(closed.frames) == 1 and closed.released == 1

unavailable = Capture(opened=False)
with patch.object(live, 'load_model', return_value=(model, options)), \
     patch.object(cv2, 'VideoCapture', return_value=unavailable):
    try:
        live.camera_worker(args, config)
        raise AssertionError('An unavailable camera must fail')
    except RuntimeError:
        pass
assert unavailable.released == 1

class StuckWorker:
    def __init__(self):
        self.stdout = io.StringIO('')
        self.killed = False
        self.returncode = None
    def poll(self):
        return self.returncode
    def kill(self):
        self.killed = True
        self.returncode = 1
    def wait(self, timeout=None):
        return self.returncode

stuck = StuckWorker()
clock = iter([0, 61, 62, 63, 64])
with patch.object(live.subprocess, 'Popen', return_value=stuck), \
     patch.object(live.time, 'monotonic', side_effect=lambda: next(clock)):
    try:
        live.supervise(args, config)
        raise AssertionError('An unresponsive camera worker must time out')
    except TimeoutError:
        pass
assert stuck.killed and stuck.stdout.closed

report = {'checked_at': datetime.now(timezone(timedelta(hours=8))).isoformat(), 'passed': True,
          'entry': 'app/live_camera.py', 'default_target_ids': options['classes'],
          'public_image_gpu_prediction': {'class_counts': dict(counts), 'nonempty_masks': 5,
                                          'device': str(result.boxes.data.device)},
          'same_frame_panes_verified': True, 'source_image_unchanged': True,
          'simulated_checks': ['1/2/3 mode switching', 'Q exit and release', 'window close and release',
                               'camera unavailable and release', 'unresponsive worker termination'],
          'real_camera_or_visible_window_tested': False, 'camera_images_or_videos_saved': False,
          'scope': 'Real GPU image prediction and simulated camera/GUI lifecycle; live desktop window and camera await manual verification.'}
destination = ROOT / 'logs/application/live_preview_check.json'
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(report, ensure_ascii=False))
