"""M1-04: trace existing-image filtering, NMS and retina mask reconstruction.

Uses the locked pretrained model and existing E0 images. Writes one compact JSON;
does not capture a camera, train, export a model, or overwrite earlier evidence.
"""

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.inspect_model import digest, preprocess, reconstruct_mask, relative, source_ref
import yaml


def trace_nms(candidates, scores, labels, indices, iou_threshold):
    """Explain the small baseline pool; compare this greedy trace to official NMS."""
    import torch
    from ultralytics.utils.metrics import box_iou
    from ultralytics.utils.ops import xywh2xyxy

    boxes = xywh2xyxy(candidates[0, :4].T)
    active = indices[scores[indices].argsort(descending=True)]
    steps, kept = [], []
    while len(active):
        chosen, remaining = active[0], active[1:]
        kept.append(int(chosen))
        overlaps = box_iou(boxes[chosen][None], boxes[remaining])[0]
        suppress = (labels[remaining] == labels[chosen]) & (overlaps > iou_threshold)
        steps.append({
            'kept_candidate': int(chosen), 'class_id': int(labels[chosen]),
            'score': float(scores[chosen]),
            'suppressed': [
                {'candidate': int(index), 'score': float(scores[index]), 'box_iou': float(overlap)}
                for index, overlap in zip(remaining[suppress], overlaps[suppress])
            ],
        })
        active = remaining[~suppress]
    return steps, torch.tensor(kept, device=candidates.device, dtype=torch.long)


def run(args):
    import cv2
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.models.yolo.segment.predict import SegmentationPredictor
    from ultralytics.utils import nms, ops

    output = (ROOT / args.output).resolve()
    if output.exists():
        raise FileExistsError('Evidence exists; select a new --output to preserve it')
    config = yaml.safe_load((ROOT / 'configs/e0.yaml').read_text(encoding='utf-8'))
    lock = json.loads((ROOT / 'configs/source-lock.json').read_text(encoding='utf-8'))
    previous_path = ROOT / 'reports/model/M1/summary.json'
    previous = json.loads(previous_path.read_text(encoding='utf-8'))
    source = ROOT / lock['path']
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    assert commit == lock['commit'] and ultralytics.__version__ == lock['version']
    assert Path(ultralytics.__file__).resolve() == source / 'ultralytics/__init__.py'
    assert not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip()
    weights = ROOT / config['weights']['path']
    protected = [weights, previous_path, ROOT / 'reports/model/M1-02/structure_check.json',
                 ROOT / 'reports/model/M1-03/tensor_check.json']
    before = {relative(path): digest(path) for path in protected}
    assert before[relative(weights)] == config['weights']['sha256']
    net = YOLO(str(weights)).model.to(args.device).float().eval()
    head = net.model[-1]
    assert head.nc == 80 and head.nm == 32 and not head.end2end and not head.xyxy
    targets = [next(i for i, name in net.names.items() if name == target['name'])
               for target in config['targets']]
    conf, iou = config['predict']['conf'], config['predict']['iou']
    variants = [('project_baseline', conf, iou, targets), ('all_classes', conf, iou, None),
                ('confidence_low', 0.10, iou, targets), ('confidence_high', 0.50, iou, targets),
                ('nms_strict', conf, 0.30, targets), ('nms_lenient', conf, 0.90, targets)]
    report = {
        'task': 'M1-04', 'checked_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'status': 'running', 'device': args.device, 'source_commit': commit,
        'versions': {'python': sys.version.split()[0], 'torch': torch.__version__,
                     'ultralytics': ultralytics.__version__},
        'weights': {'path': relative(weights), 'sha256': digest(weights)},
        'project_coco_ids': dict(zip([item['name'] for item in config['targets']], targets)),
        'settings': {'single_label': True, 'class_aware_nms': True, 'retina_masks': True,
                     'max_det': config['predict']['max_det'], 'fixed_input': [1, 3, 640, 640]},
        'source_refs': {name: source_ref(symbol) for name, symbol in [
            ('NMS', nms.non_max_suppression), ('scale_boxes', ops.scale_boxes),
            ('official_masks', ops.process_mask_native), ('learning_masks', reconstruct_mask)]},
        'cases': [],
        'scope': 'Two existing E0 images; parameter variants reuse each image forward. No GT accuracy, training, '
                 'camera, ONNX/TensorRT work, independent deployment acceptance, or performance benchmark.',
    }
    for previous_case in previous['cases']:
        image_path = ROOT / previous_case['input_path']
        assert digest(image_path) == previous_case['input_sha256']
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f'Cannot read {image_path}')
        x = preprocess(image, torch.device(args.device))
        with torch.inference_mode():
            candidates, protos = net(x)[0]
            scores, labels = candidates[0, 4:84].max(0)
            assert list(candidates.shape) == [1, 116, 8400]
            assert list(protos.shape) == [1, 32, 160, 160]
            raw_before = candidates.clone()
            h, w = image.shape[:2]
            gain = min(640 / h, 640 / w)
            new_h, new_w = round(h * gain), round(w * gain)
            pad_h, pad_w = (640 - new_h) / 2, (640 - new_w) / 2
            record = {'name': previous_case['name'], 'input': relative(image_path),
                      'input_sha256': digest(image_path), 'original_hwc': list(image.shape),
                      'preprocess': {'gain': gain, 'resized_hw': [new_h, new_w],
                                     'padding_ltrb': [round(pad_w - .1), round(pad_h - .1),
                                                      round(pad_w + .1), round(pad_h + .1)],
                                     'network_input': list(x.shape), 'dtype': str(x.dtype),
                                     'range': [float(x.min()), float(x.max())]},
                      'raw_shapes': {'candidates': list(candidates.shape), 'proto': list(protos.shape)},
                      'variants': []}
            for variant, confidence, overlap, classes in variants if record['name'] == 'desktop' else variants[:2]:
                passed = scores > confidence
                class_passed = passed if classes is None else passed & torch.isin(
                    labels, torch.tensor(classes, device=x.device))
                outputs, indices = nms.non_max_suppression(
                    candidates.clone(), conf_thres=confidence, iou_thres=overlap, classes=classes,
                    nc=80, max_det=config['predict']['max_det'], return_idxs=True, max_time_img=10)
                pred = outputs[0]
                # Official NMS uses a floating, 2-D placeholder for empty indices.
                selected = indices[0].to(dtype=torch.long).reshape(-1)
                torch.testing.assert_close(pred[:, 6:], candidates[0, 84:, selected].T, rtol=0, atol=0)
                input_boxes = ops.xywh2xyxy(candidates[0, :4, selected].T)
                torch.testing.assert_close(pred[:, :4], input_boxes, rtol=0, atol=0)
                original_boxes = ops.scale_boxes(x.shape[2:], pred[:, :4].clone(), image.shape)
                fake = SimpleNamespace(args=SimpleNamespace(retina_masks=True), model=SimpleNamespace(names=net.names))
                reference = SegmentationPredictor.construct_result(fake, pred.clone(), x, image,
                                                                    str(image_path), protos[0])
                if len(pred):
                    logits, rebuilt = reconstruct_mask(protos[0], pred[:, 6:], original_boxes, image.shape[:2])
                    nonempty = rebuilt.amax((-2, -1)) > 0
                    assert reference.masks is not None
                    assert torch.equal(rebuilt[nonempty], reference.masks.data)
                    torch.testing.assert_close(original_boxes[nonempty], reference.boxes.xyxy, rtol=0, atol=0)
                    final_indices = selected[nonempty]
                    mask_shape = list(rebuilt[nonempty].shape)
                else:
                    assert reference.masks is None and len(reference.boxes) == 0
                    empty = ops.process_mask_native(protos[0], pred[:, 6:], original_boxes, image.shape[:2])
                    assert list(empty.shape) == [0, h, w]
                    final_indices, mask_shape = selected, list(empty.shape)
                item = {'name': variant, 'conf': confidence, 'nms_iou': overlap,
                        'class_filter': classes, 'counts': {'raw': 8400, 'after_confidence': int(passed.sum()),
                        'before_nms': int(class_passed.sum()), 'after_nms': len(pred),
                        'after_nonempty_masks': len(reference.boxes)},
                        'candidate_indices': selected.cpu().tolist(), 'mask_shape': mask_shape,
                        'class_counts': dict(Counter(net.names[int(c)] for c in reference.boxes.cls)),
                        'checks': {'box_and_coefficient_indices_equal': True, 'manual_masks_pixel_equal': True},
                        'detections': [{'candidate_index': int(index), 'name': net.names[int(box[5])],
                                        'score': float(box[4]), 'xyxy_original': box[:4].cpu().tolist(),
                                        'mask_area': int(reference.masks.data[j].sum())}
                                       for j, (index, box) in enumerate(zip(final_indices, reference.boxes.data))]}
                if variant in ['project_baseline', 'all_classes']:
                    old = previous_case['modes']['project_classes' if variant == 'project_baseline' else variant]
                    assert item['counts']['after_confidence'] == old['after_confidence']
                    assert item['counts']['before_nms'] == old['after_class_filter_before_nms']
                    assert item['counts']['after_nms'] == old['after_nms']
                    assert item['class_counts'] == old['class_counts']
                    item['checks']['previous_baseline_counts_match'] = True
                if variant == 'project_baseline' and record['name'] == 'desktop':
                    trace, trace_indices = trace_nms(candidates, scores, labels, class_passed.nonzero().flatten(), overlap)
                    assert torch.equal(trace_indices, selected)
                    item['nms_trace'] = trace
                    item['checks']['greedy_trace_indices_equal_official'] = True
                    item['coordinate_example'] = {
                        'candidate_index': int(selected[0]), 'xywh_input': candidates[0, :4, selected[0]].cpu().tolist(),
                        'xyxy_input': input_boxes[0].cpu().tolist(), 'xyxy_original': original_boxes[0].cpu().tolist(),
                        'coefficient_shape': list(pred[:, 6:].shape), 'flattened_proto_shape': [32, 25600],
                        'low_resolution_logits_shape': list(logits.shape), 'final_masks_shape': mask_shape,
                    }
                record['variants'].append(item)
            assert torch.equal(raw_before, candidates), 'Postprocessing modified the shared raw reference'
            record['shared_raw_unchanged'] = True
            report['cases'].append(record)
    after = {relative(path): digest(path) for path in protected}
    assert after == before, 'Protected earlier evidence changed'
    report.update(status='postprocess_checks_passed', protected_files_unchanged=after,
                  script_sha256=digest(Path(__file__)))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'output': relative(output),
                     'cases': [{'name': case['name'], 'variants': [
                         {key: item[key] for key in ['name', 'conf', 'nms_iou', 'counts', 'class_counts']}
                         for item in case['variants']]} for case in report['cases']]}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', default='reports/model/M1-04/postprocess_check.json')
    run(parser.parse_args())
