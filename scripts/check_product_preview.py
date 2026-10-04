"""Check E1-A on fixed val images and E0 compatibility, without opening a camera."""
import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import live_camera as live
from app.product_data import digest, image_read, image_write, read_json, write_json
from app.product_training import check_guards
import numpy as np
import cv2
import torch
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='reports/application/M7-01_product_preview_precheck.json')
    args = parser.parse_args()
    destination = live.resolve_path(args.output)
    if destination.exists():
        raise FileExistsError('Precheck exists; use a fresh --output path to retain previous evidence')
    config_path = ROOT / 'configs/live_products.yaml'
    config = yaml.safe_load(config_path.read_text('utf-8'))
    training = read_json(ROOT / config['training_report']['path'])
    check_guards(training['guarded_inputs'])
    model, options = live.load_model(config)
    assert model.names == {0: 'sam_whole_milk', 1: 'yili_shuhua', 2: 'luckin_cup'}
    assert options['classes'] is None and options['save'] is False
    loading = read_json(ROOT / 'data/desktop/metadata/products-v1_loading.json')
    samples = [s for s in loading['samples'] if s['split'] == 'val']
    records = []
    for sample in samples:
        source = ROOT / sample['staged_image_path']
        image = image_read(source)
        before = digest(source)
        result = model.predict(image.copy(), **options)[0]
        assert torch.isfinite(result.boxes.data).all().item()
        ids = [int(x) for x in result.boxes.cls.tolist()]
        assert all(i in model.names for i in ids)
        if ids:
            assert result.masks is not None and len(result.masks) == len(ids)
            assert torch.isfinite(result.masks.data).all().item()
            assert list(result.masks.data.shape[1:]) == list(image.shape[:2])
            assert result.boxes.data.is_cuda and result.masks.data.is_cuda
        canvas = live.make_preview(image, result.plot(color_mode='instance', line_width=2), 'both', f"E1-A | conf {options['conf']:.2f}")
        assert np.array_equal(canvas[40:, :image.shape[1]], image)
        assert digest(source) == before
        if sample['image_id'].endswith('_004'):
            tag = format(options['conf'], '.2f').replace('.', '')
            preview = ROOT / f'runs/precheck/M7-01_product_preview/val004_both_conf{tag}.jpg'
            if preview.exists():
                ok, encoded = cv2.imencode('.jpg', canvas)
                if not ok or preview.read_bytes() != encoded.tobytes():
                    raise FileExistsError('Existing preview differs; preserve it and select a new preview location')
            else:
                image_write(preview, canvas)
        records.append({'image_id': sample['image_id'], 'source': sample['staged_image_path'],
                        'source_sha256': before, 'class_counts': dict(Counter(model.names[i] for i in ids)),
                        'scores': result.boxes.conf.tolist(), 'mask_shape': list(result.masks.data.shape) if ids else None})
    old_config = yaml.safe_load((ROOT / 'configs/e0.yaml').read_text('utf-8'))
    e0, e0_options = live.load_model(old_config)
    assert e0_options['classes'] == [41, 39, 67] and len(e0.names) == 80
    bus = image_read(ROOT / 'demo/input/E0/bus.jpg')
    result = e0.predict(bus, **{**e0_options, 'classes': None})[0]
    assert dict(Counter(result.names[int(i)] for i in result.boxes.cls.tolist())) == {'person': 4, 'bus': 1}
    assert result.masks is not None and len(result.masks) == 5
    check_guards(training['guarded_inputs'])
    report = {'status': 'passed', 'task': 'M7-01', 'checked_at': datetime.now().astimezone().isoformat(),
              'configuration': 'configs/live_products.yaml', 'configuration_sha256': digest(config_path),
              'implementation_sha256': {p: digest(ROOT / p) for p in ('app/live_camera.py', 'scripts/check_product_preview.py')},
              'weights': config['weights'], 'class_names': model.names, 'predict': options,
              'fixed_val_results': records, 'same_frame_panes_verified': True, 'original_sources_unchanged': True,
              'E0_compatibility': {'class_filter': e0_options['classes'], 'bus_instances': 5},
              'preview': str(preview.relative_to(ROOT)).replace('\\', '/'),
              'guarded_inputs_checked': len(training['guarded_inputs']), 'test_images_inferred': 0,
              'real_camera_tested': False, 'note': 'GPU fixed-image integration check only; live/manual confirmation pending.'}
    write_json(destination, report)
    print(str(destination))


if __name__ == '__main__':
    main()
