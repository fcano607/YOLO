"""M1-03: inspect real tensors in eval, export and disposable train modes.

Reuses M1 preprocessing and metadata helpers. Saves one compact JSON record;
no camera, optimizer, backward pass, model export or dependency installation.
"""

import argparse
import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess

from inspect_model import ROOT, describe, digest, preprocess, relative, source_ref
from deploy.paths import resolve_path
import yaml


def state_digest(net):
    """Include parameters and persistent buffers, including BatchNorm statistics."""
    import hashlib
    result = hashlib.sha256()
    for name, value in net.state_dict().items():
        value = value.detach().cpu().contiguous()
        result.update(name.encode('utf-8'))
        result.update(str((tuple(value.shape), value.dtype)).encode('ascii'))
        result.update(value.numpy().tobytes())
    return result.hexdigest()


def run(args):
    import cv2
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox
    from ultralytics.nn.modules import Detect, Proto, Segment

    output = resolve_path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose an empty --output to preserve prior evidence')
    lock = json.loads((ROOT / 'configs/source-lock.json').read_text('utf-8'))
    config = yaml.safe_load((ROOT / 'configs/e0.yaml').read_text('utf-8'))
    inputs = yaml.safe_load((ROOT / 'configs/deploy-precheck.yaml').read_text('utf-8'))['cases']
    source = resolve_path(lock['path'])
    assert ultralytics.__version__ == lock['version'], 'Version differs from source lock'
    assert Path(ultralytics.__file__).resolve() == source / 'ultralytics/__init__.py'
    assert subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == lock['commit']
    assert not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip()
    weights = resolve_path(config['weights']['path'])
    assert digest(weights) == config['weights']['sha256'], 'Pretrained weight hash differs'
    device = torch.device(args.device)
    net = YOLO(str(weights)).model.to(device).float().eval()
    head = net.model[-1]
    assert isinstance(head, Segment) and not head.end2end and not head.xyxy
    assert (head.nc, head.nm, head.reg_max) == (80, 32, 16)
    head.export = False
    report = {
        'task': 'M1-03', 'checked_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'status': 'running', 'device': str(device),
        'versions': {'torch': torch.__version__, 'ultralytics': ultralytics.__version__},
        'source_commit': lock['commit'],
        'weights': {'path': relative(weights), 'sha256': digest(weights)},
        'source_refs': {name: source_ref(symbol) for name, symbol in {
            'preprocess': preprocess, 'LetterBox.__call__': LetterBox.__call__,
            'Detect.forward': Detect.forward, 'Detect._get_decode_boxes': Detect._get_decode_boxes,
            'Detect._inference': Detect._inference, 'Segment.forward': Segment.forward,
            'Segment._inference': Segment._inference, 'Segment.forward_head': Segment.forward_head,
            'Proto.forward': Proto.forward,
        }.items()},
        'cases': [],
    }
    for case in inputs:
        path = resolve_path(case['image'])
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f'Cannot decode input: {path}')
        x = preprocess(image, device)
        assert list(x.shape) == [1, 3, 640, 640] and x.dtype == torch.float32
        taps, handles = {}, []

        def attach(name, module):
            def hook(_, incoming, outgoing):
                taps[name] = {'input': describe(incoming[0]), 'output': describe(outgoing)}
            handles.append(module.register_forward_hook(hook))

        for index in (0, 4, 6, 10, 16, 19, 22):
            attach(f'layer_{index}_{type(net.model[index]).__name__}', net.model[index])
        for branch, modules in [('box', head.cv2), ('class', head.cv3), ('coefficient', head.cv4)]:
            for scale, module in zip(('P3', 'P4', 'P5'), modules):
                attach(f'{branch}_{scale}', module)
        attach('proto_upsample', head.proto.upsample)
        attach('proto_output', head.proto)
        try:
            with torch.inference_mode():
                ordinary = net(x)
        finally:
            for handle in handles:
                handle.remove()
        (decoded, proto), raw = ordinary
        assert list(decoded.shape) == [1, 116, 8400]
        assert list(proto.shape) == [1, 32, 160, 160]
        assert list(raw['boxes'].shape) == [1, 64, 8400]
        assert list(raw['scores'].shape) == [1, 80, 8400]
        assert list(raw['mask_coefficient'].shape) == [1, 32, 8400]
        assert [list(t.shape) for t in raw['feats']] == [[1, 64, 80, 80], [1, 128, 40, 40], [1, 256, 20, 20]]
        with torch.inference_mode():
            distances = head.dfl(raw['boxes'])
            reconstructed_boxes = head.decode_bboxes(distances, head.anchors.unsqueeze(0)) * head.strides
            torch.testing.assert_close(decoded[:, :4], reconstructed_boxes, rtol=0, atol=0)
            torch.testing.assert_close(decoded[:, 4:84], raw['scores'].sigmoid(), rtol=0, atol=0)
            torch.testing.assert_close(decoded[:, 84:116], raw['mask_coefficient'], rtol=0, atol=0)
            torch.testing.assert_close(proto, raw['proto'], rtol=0, atol=0)
        head.export = True
        try:
            with torch.inference_mode():
                exported = net(x)
            for actual, expected in zip(exported, (decoded, proto)):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        finally:
            head.export = False
        sizes = [int(feature.shape[-1]) for feature in raw['feats']]
        candidate_count = sum(size * size for size in sizes)
        assert candidate_count == decoded.shape[-1]
        scores, labels = decoded[0, 4:84].max(0)
        index = int(scores.argmax())
        sample = {'scope': 'One highest-score candidate before any NMS or class filtering',
                  'index': index, 'box_xywh_input_pixels': decoded[0, :4, index].tolist(),
                  'class_id': int(labels[index]), 'class_name': net.names[int(labels[index])],
                  'class_score': float(scores[index]),
                  'mask_coefficients': decoded[0, 84:116, index].tolist()}
        height, width = image.shape[:2]
        gain = min(640 / height, 640 / width)
        resized = [round(height * gain), round(width * gain)]
        pad_h, pad_w = 640 - resized[0], 640 - resized[1]
        record = {
            'name': case['name'], 'input_path': relative(path), 'input_sha256': digest(path),
            'original_hwc': list(image.shape), 'resized_hw': resized,
            'padding_ltrb': [round(pad_w / 2 - 0.1), round(pad_h / 2 - 0.1),
                             round(pad_w / 2 + 0.1), round(pad_h / 2 + 0.1)],
            'network_input': describe(x), 'input_range': [float(x.min()), float(x.max())],
            'key_tensors': taps, 'ordinary_output': describe(ordinary),
            'export_mode_output': describe(exported), 'dfl_distances': describe(distances),
            'candidate_count': candidate_count, 'candidate_example': sample,
            'checks': {'decoded_boxes_reconstructed_exactly': True,
                       'class_scores_are_sigmoid_of_logits': True,
                       'mask_coefficients_are_raw_values': True,
                       'ordinary_export_equal_exactly': True},
        }
        report['cases'].append(record)
    before = state_digest(net)
    temporary = copy.deepcopy(net).train()
    bn = next(m for m in temporary.modules() if isinstance(m, torch.nn.BatchNorm2d))
    batches_before = int(bn.num_batches_tracked)
    with torch.no_grad():
        training = temporary(x)
    assert isinstance(training, dict) and set(training) == set(raw)
    for name in ('boxes', 'scores', 'mask_coefficient', 'proto'):
        assert training[name].shape == raw[name].shape
    report['training_mode'] = {
        'input_case': inputs[-1]['name'], 'output': describe(training),
        'scope': 'Forward only on a disposable train-mode copy; no labels, loss, backward or optimizer',
        'batchnorm_batches_before': batches_before,
        'batchnorm_batches_after': int(bn.num_batches_tracked),
        'values_compared_to_eval': False,
    }
    del training, temporary
    assert state_digest(net) == before, 'Original model parameters or buffers changed'
    assert not net.training and not head.export
    assert digest(weights) == config['weights']['sha256']
    report['scope'] = {'original_model_state_unchanged': True, 'weight_file_unchanged': True,
                       'human_learning_acceptance': 'pending', 'camera_used': False,
                       'project_training': False, 'optimizer_steps': 0,
                       'export_file_created': False, 'new_packages_installed': False}
    report['script_sha256'] = digest(Path(__file__))
    report['status'] = 'tensor_checks_passed'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'tensor_check.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'cases': len(report['cases']),
                      'record': str(output / 'tensor_check.json')}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', default='reports/model/M1-03')
    run(parser.parse_args())
