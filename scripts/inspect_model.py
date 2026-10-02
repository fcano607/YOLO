"""M1 source-grounded model study: tensors, NMS, masks and teaching examples.

Uses existing E0 inputs. No camera capture, optimizer step or weight export.
The training-mode/backward example uses a disposable copy and synthetic labels.
"""

import argparse
from collections import Counter
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['YOLO_CONFIG_DIR'] = str(ROOT / 'logs/ultralytics_settings')
os.environ['YOLO_AUTOINSTALL'] = 'false'
os.environ['YOLO_OFFLINE'] = 'true'

from deploy.paths import resolve_path
import yaml


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def describe(value):
    import torch
    if torch.is_tensor(value):
        return {'shape': list(value.shape), 'dtype': str(value.dtype), 'device': str(value.device)}
    if isinstance(value, dict):
        return {key: describe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return {'container': type(value).__name__, 'items': [describe(item) for item in value]}
    return str(value)


def source_ref(symbol):
    path = Path(inspect.getsourcefile(symbol)).resolve()
    lines, start = inspect.getsourcelines(symbol)
    return {'path': relative(path), 'line': start, 'end_line': start + len(lines) - 1,
            'sha256': digest(path)}


def teaching_cups():
    """Two exact hand-drawn instance masks; these are not model predictions."""
    import cv2
    import numpy as np
    masks = np.zeros((2, 240, 400), dtype=np.uint8)
    for mask, left in zip(masks, (55, 235)):
        cv2.ellipse(mask, (left + 72, 118), (30, 37), 0, 0, 360, 1, -1)
        cv2.ellipse(mask, (left + 72, 118), (16, 23), 0, 0, 360, 0, -1)
        cv2.rectangle(mask, (left, 65), (left + 69, 170), 1, -1)
        cv2.ellipse(mask, (left + 34, 170), (34, 13), 0, 0, 180, 1, -1)
    image = np.full((240, 400, 3), 242, dtype=np.uint8)
    image[masks.any(0)] = (175, 175, 175)
    boxes = []
    for mask in masks:
        y, x = np.where(mask)
        boxes.append([int(x.min()), int(y.min()), int(x.max() + 1), int(y.max() + 1)])
    return image, masks, np.asarray(boxes, dtype=np.float32)


def preprocess(image, device):
    import cv2
    import torch
    from ultralytics.data.augment import LetterBox
    padded = LetterBox(new_shape=(640, 640), auto=False, stride=32)(image=image)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).copy()
    return torch.from_numpy(rgb).unsqueeze(0).to(device).float() / 255.0


def reconstruct_mask(proto, coefficients, boxes, shape):
    """Learning implementation: explicitly combine, undo padding, interpolate, crop."""
    import torch
    import torch.nn.functional as functional
    c, mh, mw = proto.shape
    logits = (coefficients @ proto.float().reshape(c, -1)).reshape(-1, mh, mw)
    height, width = shape
    gain = min(mh / height, mw / width)
    pad_x, pad_y = (mw - round(width * gain)) / 2, (mh - round(height * gain)) / 2
    left, top = round(pad_x - 0.1), round(pad_y - 0.1)
    content = logits[:, top:top + round(height * gain), left:left + round(width * gain)]
    scaled = functional.interpolate(content[None].float(), size=shape, mode='bilinear',
                                    align_corners=False)[0]
    x = torch.arange(width, device=boxes.device, dtype=boxes.dtype)[None, None, :]
    y = torch.arange(height, device=boxes.device, dtype=boxes.dtype)[None, :, None]
    crop = ((x >= boxes[:, 0, None, None]) & (x < boxes[:, 2, None, None]) &
            (y >= boxes[:, 1, None, None]) & (y < boxes[:, 3, None, None]))
    return logits, ((scaled > 0) & crop).byte()


def toy_metrics(image, masks, boxes):
    import cv2
    import numpy as np
    import torch
    from ultralytics.engine.validator import BaseValidator
    from ultralytics.utils.metrics import ap_per_class, box_iou, mask_iou
    predicted_boxes = torch.tensor(np.stack([boxes[0] + [5, 0, 5, 0], boxes[0]]), dtype=torch.float32)
    predicted_masks = np.stack([cv2.warpAffine(masks[0], np.float32([[1, 0, 5], [0, 1, 0]]),
                                             (image.shape[1], image.shape[0])), masks[0]])
    gt_cls, pred_cls = torch.zeros(2), torch.zeros(2)
    thresholds = torch.linspace(0.5, 0.95, 10)
    validator = SimpleNamespace(iouv=thresholds)
    box_overlaps = box_iou(torch.tensor(boxes), predicted_boxes)
    mask_overlaps = mask_iou(torch.tensor(masks).float().flatten(1),
                            torch.tensor(predicted_masks).float().flatten(1))
    scores = np.array([0.9, 0.8])
    output = {'scope': 'Synthetic teaching only: two cup GTs, two predictions on cup A; cup B missed',
              'iou_thresholds': thresholds.tolist(), 'box_iou': box_overlaps.tolist(),
              'mask_iou': mask_overlaps.tolist(), 'scores': scores.tolist()}
    for name, overlaps in [('box', box_overlaps), ('mask', mask_overlaps)]:
        matched = BaseValidator.match_predictions(validator, pred_cls, gt_cls, overlaps).numpy()
        ap = ap_per_class(matched, scores, pred_cls.numpy(), gt_cls.numpy())[5]
        tp = int(matched[:, 0].sum())
        output[name] = {'tp_at_iou50': tp, 'fp_at_iou50': 2 - tp, 'fn_at_iou50': 2 - tp,
                        'precision_at_fixed_confidence': tp / 2, 'recall_at_fixed_confidence': tp / 2,
                        'ap_by_iou': ap[0].tolist(), 'map50_teaching': float(ap[:, 0].mean()),
                        'map50_95_teaching': float(ap.mean())}
        assert tp == 1, 'Duplicate detections must not count as a second true positive'
    return output


def backward_demo(net, device, cup_image, cup_masks, cup_boxes, class_id):
    import cv2
    import numpy as np
    import torch
    from ultralytics.cfg import get_cfg
    from ultralytics.data.augment import LetterBox
    temporary = copy.deepcopy(net).train().requires_grad_(True)
    temporary.args = get_cfg(overrides={'overlap_mask': False})
    x = preprocess(cup_image, device)
    masks = []
    for mask in cup_masks:
        padded = LetterBox(new_shape=(640, 640), auto=False, stride=32, padding_value=0)(image=mask)
        masks.append(cv2.resize(padded, (160, 160), interpolation=cv2.INTER_NEAREST))
    gain = min(640 / cup_image.shape[0], 640 / cup_image.shape[1])
    pad = np.array([(640 - round(cup_image.shape[1] * gain)) / 2,
                    (640 - round(cup_image.shape[0] * gain)) / 2] * 2)
    boxes = cup_boxes * gain + pad
    xywh = np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2,
                           (boxes[:, 1] + boxes[:, 3]) / 2,
                           boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]]) / 640
    batch = {'batch_idx': torch.zeros(2, device=device),
             'cls': torch.full((2, 1), class_id, device=device),
             'bboxes': torch.tensor(xywh, dtype=torch.float32, device=device),
             'masks': torch.tensor(np.stack(masks), dtype=torch.float32, device=device)}
    raw = temporary(x)
    criterion = temporary.init_criterion()
    losses, parts = criterion(raw, batch)
    assert torch.isfinite(losses).all(), 'Non-finite teaching loss'
    losses.sum().backward()
    gradient_sums = {}
    for name, module in [('backbone', temporary.model[0]),
                         ('prototype', temporary.model[-1].proto),
                         ('mask_coefficients', temporary.model[-1].cv4)]:
        total = sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)
        assert total > 0, f'No gradient reached {name}'
        gradient_sums[name] = total
    record = {'scope': 'One forward/backward on a disposable copy with hand-drawn cup labels; no optimizer step',
              'raw_training_output': describe(raw), 'criterion': type(criterion).__name__,
              'loss_parts': {key: float(value) for key, value in parts.items()},
              'weighted_loss_sum': float(losses.sum().detach()), 'gradient_abs_sums': gradient_sums,
              'optimizer_steps': 0, 'weights_saved': False, 'project_training_result': False,
              'gains': {name: getattr(temporary.args, name) for name in ['box', 'cls', 'dfl']},
              'overlap_mask': temporary.args.overlap_mask}
    del temporary
    return record


def plots(output, cup_image, cup_masks, cup_boxes, selected):
    import cv2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    import numpy as np
    font_path = Path('C:/Windows/Fonts/msyh.ttc')
    if font_path.is_file():
        plt.rcParams['font.family'] = FontProperties(fname=str(font_path)).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    colors = np.array([[54, 115, 203], [232, 130, 45]], dtype=np.uint8)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6), layout='constrained')
    titles = ['分类：图中有杯子', '检测：两个框，两个实例',
              '语义分割：杯子像素使用同一类别颜色', '实例分割：每个杯子有自己的掩膜']
    for index, ax in enumerate(axes.flat):
        picture = cup_image.copy()
        if index in (2, 3):
            for i, mask in enumerate(cup_masks):
                picture[mask.astype(bool)] = colors[0 if index == 2 else i]
        ax.imshow(picture)
        if index == 1:
            for i, (left, top, right, bottom) in enumerate(cup_boxes):
                ax.add_patch(plt.Rectangle((left, top), right-left, bottom-top,
                                          fill=False, edgecolor=colors[i]/255, linewidth=2))
                ax.text(left, top - 7, f'cup {chr(65+i)}', color=colors[i]/255)
        ax.set_title(titles[index], fontsize=12)
        ax.axis('off')
    fig.suptitle('M1-01 两个杯子的任务对比｜手工教学图，未运行模型', fontsize=15)
    fig.savefig(output / 'task_comparison.png', dpi=160)
    plt.close(fig)

    image, proto, coefficients, logits, mask, box, name = selected
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), layout='constrained')
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    axes[0, 0].imshow(rgb)
    left, top, right, bottom = box
    axes[0, 0].add_patch(plt.Rectangle((left, top), right-left, bottom-top,
                                      fill=False, edgecolor='#e7822d', linewidth=2))
    axes[0, 0].set_title(f'实际输入与选中实例：{name}')
    magnitude = np.abs(coefficients) * np.abs(proto).mean(axis=(1, 2))
    indices = np.argsort(magnitude)[-4:][::-1]
    montage = np.concatenate([np.concatenate([proto[indices[0]], proto[indices[1]]], 1),
                              np.concatenate([proto[indices[2]], proto[indices[3]]], 1)], 0)
    axes[0, 1].imshow(montage, cmap='coolwarm')
    axes[0, 1].set_title(f'共享原型示例：{indices.tolist()}（共 32 个）')
    axes[0, 2].bar(np.arange(32), coefficients, color='#3673cb')
    axes[0, 2].set_title('该实例的 32 个系数（可正可负）')
    axes[0, 2].set_xlabel('原型编号')
    axes[1, 0].imshow(logits, cmap='coolwarm')
    axes[1, 0].set_title('系数 × 原型 → 160×160 logits')
    axes[1, 1].imshow(mask, cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title('去填充 / 插值 / 阈值 / 框裁剪后的掩膜')
    overlay = rgb.copy()
    overlay[mask.astype(bool)] = (0.5 * overlay[mask.astype(bool)] + 0.5 * colors[1]).astype(np.uint8)
    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title('重建掩膜叠加至原始画面')
    for i, ax in enumerate(axes.flat):
        if i != 2:
            ax.axis('off')
    fig.suptitle('M1-04 真实实例掩膜重建｜与同输入的官方后处理逐像素对照', fontsize=15)
    fig.savefig(output / 'mask_reconstruction.png', dpi=160)
    plt.close(fig)


def run(args):
    import cv2
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.models.yolo.segment.predict import SegmentationPredictor
    from ultralytics.nn.modules import C2PSA, C3k2, Detect, Proto, Segment, SPPF
    from ultralytics.nn.tasks import SegmentationModel, parse_model
    from ultralytics.utils import loss, metrics, nms, ops
    from ultralytics.utils.tal import TaskAlignedAssigner

    output = resolve_path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Output is nonempty; choose a new --output to preserve prior evidence')
    config = yaml.safe_load((ROOT / 'configs/e0.yaml').read_text('utf-8'))
    lock = json.loads((ROOT / 'configs/source-lock.json').read_text('utf-8'))
    source = resolve_path(lock['path'])
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    assert commit == lock['commit'] and ultralytics.__version__ == lock['version'], 'Source lock mismatch'
    assert Path(ultralytics.__file__).resolve() == source / 'ultralytics/__init__.py', 'Wrong imported source'
    assert not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip(), 'Upstream source modified'
    weights = resolve_path(config['weights']['path'])
    assert digest(weights) == config['weights']['sha256'], 'Weight hash mismatch'
    device = torch.device(args.device)
    wrapper = YOLO(str(weights))
    net = wrapper.model.to(device).float().eval()
    head = net.model[-1]
    assert isinstance(head, Segment) and head.nc == 80 and head.nm == 32 and not head.end2end
    target_ids = [next(i for i, name in net.names.items() if name == target['name'])
                  for target in config['targets']]
    report = {'module': 'M1', 'checked_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
              'status': 'running', 'python': sys.executable,
              'versions': {'torch': torch.__version__, 'ultralytics': ultralytics.__version__},
              'source_commit': commit, 'weights': {'path': relative(weights), 'sha256': digest(weights)},
              'model': {'nc': head.nc, 'nm': head.nm, 'npr_actual': head.npr,
                        'reg_max': head.reg_max, 'strides': head.stride.tolist(),
                        'parameters_before_fusion': sum(p.numel() for p in net.parameters()),
                        'top_level_modules': len(net.model), 'scale': net.yaml.get('scale'),
                        'default_target_ids': target_ids}, 'source_refs': {}, 'cases': []}
    symbols = {'C3k2': C3k2, 'SPPF': SPPF, 'C2PSA': C2PSA, 'Proto': Proto,
               'parse_model': parse_model, 'Detect.forward': Detect.forward,
               'Detect._inference': Detect._inference, 'Segment.forward': Segment.forward,
               'Segment.forward_head': Segment.forward_head,
               'SegmentationPredictor.construct_result': SegmentationPredictor.construct_result,
               'NMS': nms.non_max_suppression, 'process_mask_native': ops.process_mask_native,
               'scale_masks': ops.scale_masks, 'crop_mask': ops.crop_mask,
               'SegmentationModel.init_criterion': SegmentationModel.init_criterion,
               'v8DetectionLoss': loss.v8DetectionLoss, 'v8SegmentationLoss': loss.v8SegmentationLoss,
               'BboxLoss': loss.BboxLoss, 'DFLoss': loss.DFLoss,
               'single_mask_loss': loss.v8SegmentationLoss.single_mask_loss,
               'TaskAlignedAssigner': TaskAlignedAssigner, 'box_iou': metrics.box_iou,
               'mask_iou': metrics.mask_iou, 'compute_ap': metrics.compute_ap,
               'ap_per_class': metrics.ap_per_class}
    report['source_refs'] = {name: source_ref(symbol) for name, symbol in symbols.items()}
    deploy = yaml.safe_load((ROOT / 'configs/deploy-precheck.yaml').read_text('utf-8'))
    selected = None
    overlays = []
    for case in deploy['cases']:
        path = resolve_path(case['image'])
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f'Cannot decode {path}')
        x = preprocess(image, device)
        layers = []
        def hook(module, inputs, outputs):
            layers.append({'index': module.i, 'from': module.f, 'module': type(module).__name__,
                           'parameters': sum(p.numel() for p in module.parameters()), 'output': describe(outputs)})
        handles = [module.register_forward_hook(hook) for module in net.model] if not report['cases'] else []
        with torch.inference_mode():
            normal = net(x)
        for handle in handles:
            handle.remove()
        candidates, protos = normal[0]
        assert list(candidates.shape) == [1, 116, 8400] and list(protos.shape) == [1, 32, 160, 160]
        if not report['cases']:
            report['layers'] = layers
            report['ordinary_output'] = describe(normal)
            head.export = True
            try:
                with torch.inference_mode():
                    export = net(x)
                report['export_mode_output'] = describe(export)
                torch.testing.assert_close(export[0], candidates, rtol=0, atol=0)
                torch.testing.assert_close(export[1], protos, rtol=0, atol=0)
            finally:
                head.export = False
        record = {'name': case['name'], 'input_path': relative(path), 'input_sha256': digest(path),
                  'original_shape': list(image.shape), 'network_input': describe(x),
                  'raw_candidates': describe(candidates), 'prototype': describe(protos), 'modes': {}}
        scores, labels = candidates[0, 4:84].max(0)
        passed = scores > config['predict']['conf']
        for mode, class_ids in [('all_classes', None), ('project_classes', target_ids)]:
            kwargs = {'conf_thres': config['predict']['conf'], 'iou_thres': config['predict']['iou'],
                      'classes': class_ids, 'nc': 80, 'max_det': config['predict']['max_det'], 'return_idxs': True}
            with torch.inference_mode():
                outputs, kept_indices = nms.non_max_suppression(candidates.clone(), **kwargs)
            predictions = outputs[0]
            pre_nms = passed if class_ids is None else passed & torch.isin(labels, torch.tensor(class_ids, device=device))
            fake_predictor = SimpleNamespace(args=SimpleNamespace(retina_masks=True),
                                             model=SimpleNamespace(names=net.names))
            reference = SegmentationPredictor.construct_result(fake_predictor, predictions.clone(), x,
                                                               image, str(path), protos[0])
            boxes = ops.scale_boxes(x.shape[2:], predictions[:, :4].clone(), image.shape)
            if len(predictions):
                logits, rebuilt = reconstruct_mask(protos[0], predictions[:, 6:], boxes, image.shape[:2])
                keep = rebuilt.amax((-2, -1)) > 0
                rebuilt = rebuilt[keep]
                assert reference.masks is not None and torch.equal(rebuilt, reference.masks.data), 'Mask reconstruction mismatch'
                torch.testing.assert_close(boxes[keep], reference.boxes.xyxy, rtol=0, atol=0)
                if mode == 'project_classes' and case['name'] == 'desktop':
                    index = next((i for i, item in enumerate(predictions) if int(item[5]) == target_ids[0]), 0)
                    selected = (image, protos[0].cpu().numpy(), predictions[index, 6:].cpu().numpy(),
                                logits[index].cpu().numpy(), reconstruct_mask(protos[0], predictions[index:index+1, 6:],
                                boxes[index:index+1], image.shape[:2])[1][0].cpu().numpy(),
                                boxes[index].cpu().tolist(), net.names[int(predictions[index, 5])])
            else:
                assert reference.masks is None and len(reference.boxes) == 0
                keep = torch.zeros(0, dtype=torch.bool, device=device)
            detections = []
            for idx, (box, row) in enumerate(zip(reference.boxes.data.cpu().tolist(),
                                                 predictions[keep].cpu().tolist())):
                detections.append({'class_id': int(box[5]), 'name': net.names[int(box[5])],
                                   'confidence': box[4], 'xyxy': box[:4],
                                   'mask_area_pixels': int(reference.masks.data[idx].sum()),
                                   'mask_coefficients': row[6:]})
            record['modes'][mode] = {'candidates': 8400, 'after_confidence': int(passed.sum()),
                                    'after_class_filter_before_nms': int(pre_nms.sum()),
                                    'after_nms': len(predictions), 'after_nonempty_masks': len(reference.boxes),
                                    'kept_candidate_indices': kept_indices[0].cpu().tolist(),
                                    'class_counts': dict(Counter(item['name'] for item in detections)),
                                    'detections': detections, 'manual_mask_pixel_equal_reference': True}
            if mode == 'project_classes':
                overlays.append((case['name'], reference.plot(color_mode='instance', line_width=2)))
        report['cases'].append(record)
    assert selected is not None, 'No project instance available for the mask figure'
    cup_image, cup_masks, cup_boxes = teaching_cups()
    report['teaching_metrics'] = toy_metrics(cup_image, cup_masks, cup_boxes)
    report['training_mode_backward_demo'] = backward_demo(net, device, cup_image, cup_masks, cup_boxes, target_ids[0])
    report['scope'] = {'human_learning_acceptance': 'pending', 'camera_used': False,
                       'project_training': False, 'project_map_evaluation': False,
                       'export_file_created': False, 'new_packages_installed': False}
    assert digest(weights) == report['weights']['sha256'], 'Weight file changed'
    output.mkdir(parents=True, exist_ok=True)
    plots(output, cup_image, cup_masks, cup_boxes, selected)
    for name, picture in overlays:
        assert cv2.imwrite(str(output / f'{name}_project_result.jpg'), picture)
    report['status'] = 'engineering_checks_passed'
    report['artifact_sha256'] = {p.name: digest(p) for p in output.iterdir() if p.is_file()}
    report['script_sha256'] = digest(Path(__file__))
    (output / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'output': relative(output),
                      'cases': [{item['name']: {mode: {key: value[key] for key in ['after_confidence',
                                      'after_class_filter_before_nms', 'after_nms', 'class_counts']}
                                      for mode, value in item['modes'].items()}} for item in report['cases']],
                      'loss_parts': report['training_mode_backward_demo']['loss_parts']}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', default='reports/model/M1')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
