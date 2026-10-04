"""Same-frame E0/product camera, image or video preview; no image/video recording."""

import argparse
from collections import Counter
from datetime import datetime
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


def verify_model_contract(config, model):
    """Use the checkpoint's own class namespace, never COCO IDs for products."""
    kind = config.get('model_kind', 'coco_e0')
    if model.task != 'segment':
        raise ValueError('Preview requires an instance segmentation model')
    if kind == 'coco_e0':
        if len(model.names) != 80:
            raise ValueError('E0 requires the original COCO 80-class segmentation model')
        ids = []
        for target in config['targets']:
            matches = [i for i, name in model.names.items() if name == target['name']]
            if len(matches) != 1:
                raise ValueError(f"Invalid target class: {target['name']}")
            ids.append(matches[0])
        return ids
    if kind != 'products':
        raise ValueError('Unknown preview model_kind')
    values = {}
    for key in ('class_metadata', 'training_report'):
        reference = config[key]
        path = resolve_path(reference['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != reference['sha256']:
            raise ValueError(f'Product {key} changed from the runtime configuration')
        values[key] = json.loads(path.read_text('utf-8'))
    mapping = values['class_metadata']
    expected = {item['id']: item['name'] for item in mapping['classes']}
    head = model.model.model[-1]
    if (mapping['status'] != 'frozen' or mapping['mapping_version'] != 'products-v1' or
            list(expected) != [0, 1, 2] or model.names != expected or
            head.nc != 3 or not hasattr(head, 'proto')):
        raise ValueError('Expected the frozen three-product names and segmentation head')
    report = values['training_report']
    if (report['status'] != 'completed' or report['experiment'] != config['experiment'] or
            report['training']['checkpoints']['best'] != config['weights']):
        raise ValueError('Product weights must match the completed best checkpoint report')
    return None  # The trained network itself has only these three classes.


def load_model(config, device=None, all_classes=False):
    import numpy as np
    import ultralytics
    from ultralytics import YOLO

    weights = resolve_path(config['weights']['path'])
    if not weights.is_file():
        raise FileNotFoundError(f'Configured preview weights do not exist: {weights}')
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
    ids = verify_model_contract(config, model)
    options = {**config['predict'], 'save': False, 'save_txt': False, 'show': False, 'verbose': False}
    if device is not None:
        options['device'] = device
    options['classes'] = None if all_classes else ids
    # Initialize the GPU before acquiring the camera, as in the existing E0 entry.
    model.predict(np.zeros((480, 640, 3), dtype=np.uint8), **options)
    return model, options


def make_preview(frame, annotated, view, result_label='YOLO11n-seg | masks + boxes'):
    """Keep the camera image intact; both panes correspond to this same frame."""
    import cv2
    import numpy as np

    if view not in {'original', 'result', 'both'}:
        raise ValueError('Unknown preview view')
    if view != 'original' and (annotated is None or annotated.shape != frame.shape):
        raise ValueError('The result must have the same image shape as the camera frame')
    if view == 'both':
        picture = cv2.hconcat([frame, annotated])
        labels = [('Original camera', 12), (result_label, frame.shape[1] + 12)]
    else:
        picture = frame if view == 'original' else annotated
        labels = [('Original camera' if view == 'original' else result_label, 12)]
    title = np.zeros((40, picture.shape[1], 3), dtype=np.uint8)
    for label, x in labels:
        cv2.putText(title, label, (x, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.vconcat([title, picture])


def camera_worker(args, config):
    import cv2
    import torch

    print('Initializing YOLO. Preview does not record or save images.', flush=True)
    source_arg = getattr(args, 'source', None)
    source = resolve_path(source_arg) if source_arg else None
    image_source = source is not None and source.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    capture = None
    window_created = False
    processed = 0
    inferred = 0
    counts = Counter()
    presence = Counter()
    inference_ms = []
    exit_reason = 'duration_limit'
    report_path = resolve_path(args.report) if getattr(args, 'report', None) else None
    report = {'status': 'running', 'task': 'M7-01', 'started_at': datetime.now().astimezone().isoformat(),
              'experiment': config.get('experiment', 'E0'), 'weights': config['weights'],
              'source_kind': 'image' if image_source else 'video' if source else 'camera',
              'source': str(source) if source else config['camera']['index'] if args.camera_index is None else args.camera_index,
              'camera_images_or_videos_saved': False, 'final_test_inferred': False if source is None else None,
              'evaluation_note': 'Unlabeled live preview; class counts are predictions, not accuracy or recall.',
              'manual_window_confirmation': 'pending'}
    started = time.monotonic()
    heartbeat = started
    view = args.view
    try:
        if report_path is not None and report_path.exists():
            raise FileExistsError('Session report already exists; use a fresh --report path')
        runtime_config = config
        if getattr(args, 'conf', None) is not None:
            runtime_config = {**config, 'predict': {**config['predict'], 'conf': args.conf}}
        model, options = load_model(runtime_config, args.device, args.all_classes)
        report['predict'] = options
        report['class_names'] = model.names
        if source is not None and not source.is_file():
            raise FileNotFoundError(f'Preview source does not exist: {source}')
        if image_source:
            from app.product_data import image_read
            still = image_read(source)
        else:
            if source is None:
                api = {'DSHOW': cv2.CAP_DSHOW, 'MSMF': cv2.CAP_MSMF, 'ANY': cv2.CAP_ANY}[config['camera']['backend']]
                capture = cv2.VideoCapture(report['source'], api)
            else:
                capture = cv2.VideoCapture(str(source))
            if not capture.isOpened():
                raise RuntimeError('Could not open camera/video; check source, index and other camera applications')
            report['capture_backend'] = capture.getBackendName() if hasattr(capture, 'getBackendName') else 'simulated'
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        window_created = True
        started = time.monotonic()
        while args.seconds is None or time.monotonic() - started < args.seconds:
            ok, frame = (True, still.copy()) if image_source else capture.read()
            if not ok or frame is None:
                if source is not None:
                    exit_reason = 'video_end'
                    break
                raise RuntimeError('Could not read a camera frame')
            annotated = None
            if view != 'original':
                prediction_started = time.perf_counter()
                result = model.predict(frame.copy(), **options)[0]
                if result.boxes is None or not torch.isfinite(result.boxes.data).all().item():
                    raise ValueError('Invalid preview boxes')
                ids = [int(x) for x in result.boxes.cls.tolist()]
                if ids and (result.masks is None or len(result.masks) != len(ids) or
                            not torch.isfinite(result.masks.data).all().item()):
                    raise ValueError('Each preview instance must have a finite matching mask')
                inference_ms.append((time.perf_counter() - prediction_started) * 1000)
                frame_counts = Counter(result.names[i] for i in ids)
                counts.update(frame_counts)
                presence.update(frame_counts.keys())
                inferred += 1
                annotated = result.plot(color_mode='instance', line_width=2)
            label = f"{config.get('experiment', 'E0')} | conf {options['conf']:.2f} | masks + boxes"
            preview = make_preview(frame, annotated, view, label)
            if processed == 0:
                width = min(1280, preview.shape[1])
                cv2.resizeWindow(WINDOW, width, round(preview.shape[0] * width / preview.shape[1]))
            cv2.imshow(WINDOW, preview)
            processed += 1
            if processed == 1:
                report['frame_shape'] = list(frame.shape)
                print(f"Preview ready: {config.get('experiment', 'E0')}, classes={model.names}, conf={options['conf']}. "
                      '1/2/3: switch view; Q/Esc/window close: exit.', flush=True)
                event('ready')
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                exit_reason = 'key_exit'
                break
            if key in (ord('1'), ord('2'), ord('3')):
                view = {ord('1'): 'original', ord('2'): 'result', ord('3'): 'both'}[key]
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                exit_reason = 'window_close'
                break
            if time.monotonic() - heartbeat >= 1:
                event('heartbeat')
                heartbeat = time.monotonic()
        if processed == 0:
            raise RuntimeError('No valid preview frame was displayed')
        report['status'] = 'completed'
    except KeyboardInterrupt:
        exit_reason = 'keyboard_interrupt'
        report['status'] = 'interrupted'
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        event('closing')
        # Close the visible window first; the parent also bounds a blocked release.
        if window_created:
            cv2.destroyAllWindows()
        report['window_destroyed'] = window_created
        if capture is not None:
            capture.release()
        report.update(finished_at=datetime.now().astimezone().isoformat(), exit_reason=exit_reason,
                      frames_displayed=processed, frames_inferred=inferred, class_instance_predictions=dict(counts),
                      frames_with_class=dict(presence), capture_released=capture is not None,
                      preview_seconds=round(time.monotonic() - started, 3),
                      predict_ms_mean=sum(inference_ms) / len(inference_ms) if inference_ms else None,
                      timing_note='Preview includes capture/display; predict timing includes output checks. No controlled benchmark.')
        if report_path is not None and not report_path.exists():
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with report_path.open('x', encoding='utf-8') as stream:
                stream.write(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
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
    parser.add_argument('--source', help='Optional image/video path; default uses the camera')
    parser.add_argument('--report', help='Fresh JSON session report path; contains metadata only, no camera images')
    parser.add_argument('--camera-index', type=int)
    parser.add_argument('--device')
    parser.add_argument('--conf', type=float, help='Preview confidence override; frozen training/evaluation configs are unchanged')
    parser.add_argument('--all-classes', action='store_true')
    parser.add_argument('--seconds', type=float, help='Optional duration; default runs until Q/Esc/window close')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.seconds is not None and args.seconds <= 0:
        parser.error('seconds must be positive')
    if args.camera_index is not None and args.camera_index < 0:
        parser.error('camera-index must be nonnegative')
    if args.conf is not None and not 0 <= args.conf <= 1:
        parser.error('conf must be in [0,1]')
    config = yaml.safe_load(resolve_path(args.config).read_text('utf-8'))
    try:
        return camera_worker(args, config) if args.worker else supervise(args, config)
    except Exception as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
