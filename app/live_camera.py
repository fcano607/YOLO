"""Live E0 camera preview: original image and instance segmentation, no recording."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['YOLO_CONFIG_DIR'] = str(ROOT / 'logs/ultralytics_settings')
os.environ['YOLO_AUTOINSTALL'] = 'false'
os.environ['YOLO_OFFLINE'] = 'true'

from deploy.paths import resolve_path
import yaml

WINDOW = 'Camera + YOLO | 1: camera  2: YOLO  3: both  Q/Esc: quit'
EVENT = '__LIVE_EVENT__='


def event(name):
    print(EVENT + name, flush=True)


def load_model(config, device=None, all_classes=False):
    import numpy as np
    import ultralytics
    from ultralytics import YOLO

    weights = resolve_path(config['weights']['path'])
    if not weights.is_file():
        raise FileNotFoundError('Run python scripts/prepare_e0.py first')
    if hashlib.sha256(weights.read_bytes()).hexdigest() != config['weights']['sha256']:
        raise ValueError('Weight SHA256 differs from the project configuration')
    lock = json.loads((ROOT / 'configs/source-lock.json').read_text('utf-8'))
    source = resolve_path(lock['path'])
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != lock['commit'] or ultralytics.__version__ != lock['version']:
        raise ValueError('Ultralytics version/commit differs from the locked source')
    if Path(ultralytics.__file__).resolve() != source / 'ultralytics/__init__.py':
        raise ValueError('Ultralytics must load from the locked project source')
    model = YOLO(str(weights))
    if model.task != 'segment' or len(model.names) != 80:
        raise ValueError('This preview uses the original COCO 80-class E0 segmentation model')
    options = {**config['predict'], 'save': False, 'save_txt': False, 'show': False, 'verbose': False}
    if device is not None:
        options['device'] = device
    if not all_classes:
        ids = []
        for target in config['targets']:
            matches = [i for i, name in model.names.items() if name == target['name']]
            if len(matches) != 1:
                raise ValueError(f"Invalid target class: {target['name']}")
            ids.append(matches[0])
        options['classes'] = ids
    # Initialize the GPU before acquiring the camera, as in the existing E0 entry.
    model.predict(np.zeros((480, 640, 3), dtype=np.uint8), **options)
    return model, options


def make_preview(frame, annotated, view):
    """Keep the camera image intact; both panes correspond to this same frame."""
    import cv2
    import numpy as np

    if view not in {'original', 'result', 'both'}:
        raise ValueError('Unknown preview view')
    if view != 'original' and (annotated is None or annotated.shape != frame.shape):
        raise ValueError('The result must have the same image shape as the camera frame')
    if view == 'both':
        picture = cv2.hconcat([frame, annotated])
        labels = [('Original camera', 12), ('YOLO11n-seg | masks + boxes', frame.shape[1] + 12)]
    else:
        picture = frame if view == 'original' else annotated
        labels = [('Original camera' if view == 'original' else 'YOLO11n-seg | masks + boxes', 12)]
    title = np.zeros((40, picture.shape[1], 3), dtype=np.uint8)
    for label, x in labels:
        cv2.putText(title, label, (x, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.vconcat([title, picture])


def camera_worker(args, config):
    import cv2

    print('Initializing YOLO. Camera preview does not record or save images.', flush=True)
    model, options = load_model(config, args.device, args.all_classes)
    backend = {'DSHOW': cv2.CAP_DSHOW, 'MSMF': cv2.CAP_MSMF, 'ANY': cv2.CAP_ANY}[config['camera']['backend']]
    index = config['camera']['index'] if args.camera_index is None else args.camera_index
    capture = None
    window_created = False
    processed = 0
    started = time.monotonic()
    heartbeat = started
    view = args.view
    try:
        capture = cv2.VideoCapture(index, backend)
        if not capture.isOpened():
            raise RuntimeError('Could not open the camera; check its index and other camera applications')
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        window_created = True
        print('Camera ready. 1: camera / 2: YOLO / 3: both. Q, Esc or closing the window exits.', flush=True)
        event('ready')
        started = time.monotonic()
        while args.seconds is None or time.monotonic() - started < args.seconds:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError('Could not read a camera frame')
            annotated = None
            if view != 'original':
                result = model.predict(frame.copy(), **options)[0]
                annotated = result.plot(color_mode='instance', line_width=2)
            preview = make_preview(frame, annotated, view)
            if processed == 0:
                width = min(1280, preview.shape[1])
                cv2.resizeWindow(WINDOW, width, round(preview.shape[0] * width / preview.shape[1]))
            cv2.imshow(WINDOW, preview)
            processed += 1
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break
            if key in (ord('1'), ord('2'), ord('3')):
                view = {ord('1'): 'original', ord('2'): 'result', ord('3'): 'both'}[key]
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if time.monotonic() - heartbeat >= 1:
                event('heartbeat')
                heartbeat = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        event('closing')
        # Close the visible window first; the parent also bounds a blocked release.
        if window_created:
            cv2.destroyAllWindows()
        if capture is not None:
            capture.release()
    print(f'Camera closed. Frames displayed: {processed}. No recordings saved.', flush=True)
    return 0


def supervise(args, config):
    """Bound initialization/read/release hangs without storing worker logs."""
    child = subprocess.Popen([sys.executable, '-B', '-X', 'utf8', str(Path(__file__).resolve()),
                              *sys.argv[1:], '--worker'], cwd=ROOT,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding='utf-8', errors='replace',
                             env=dict(os.environ, PYTHONIOENCODING='utf-8'))
    messages = queue.Queue()

    def read_output():
        for line in child.stdout:
            messages.put(line.rstrip())

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    last_activity = time.monotonic()
    closing = None
    timeout = max(60, config['camera']['timeout'])
    try:
        while True:
            try:
                line = messages.get(timeout=0.2)
                last_activity = time.monotonic()
                if line == EVENT + 'closing':
                    closing = last_activity
                elif not line.startswith(EVENT):
                    print(line, flush=True)
            except queue.Empty:
                pass
            if child.poll() is not None:
                reader.join(timeout=1)
                while not messages.empty():
                    line = messages.get_nowait()
                    if not line.startswith(EVENT):
                        print(line, flush=True)
                return child.returncode
            now = time.monotonic()
            if (closing is not None and now - closing > 10) or now - last_activity > timeout:
                raise TimeoutError('Camera worker stopped responding; its process will be closed')
    except KeyboardInterrupt:
        print('Stopping camera preview...', flush=True)
        try:
            return child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            return 1
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)
        reader.join(timeout=1)
        child.stdout.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--view', choices=['both', 'original', 'result'], default='both')
    parser.add_argument('--config', default='configs/e0.yaml')
    parser.add_argument('--camera-index', type=int)
    parser.add_argument('--device')
    parser.add_argument('--all-classes', action='store_true')
    parser.add_argument('--seconds', type=float, help='Optional duration; default runs until Q/Esc/window close')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.seconds is not None and args.seconds <= 0:
        parser.error('seconds must be positive')
    if args.camera_index is not None and args.camera_index < 0:
        parser.error('camera-index must be nonnegative')
    config = yaml.safe_load(resolve_path(args.config).read_text('utf-8'))
    try:
        return camera_worker(args, config) if args.worker else supervise(args, config)
    except Exception as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
