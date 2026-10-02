"""M0-06: reproduce E0 and ONNX from new processes outside the project cwd.

The default uses the public bus image restored by prepare_e0.py. An optional
desktop image adds a second real input. Earlier M0-04/M0-05 evidence is preserved.
"""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from deploy.paths import resolve_path
import yaml


def now():
    return datetime.now(timezone(timedelta(hours=8)))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def training_worker(output):
    from verify_training_setup import check_training_setup
    report = check_training_setup()
    report.update({'task': 'M0-06', 'check': 'fresh-process training environment check',
                   'pid': os.getpid(), 'cwd': str(Path.cwd()), 'timestamp': now().isoformat()})
    write_json(output, report)


def run(args):
    archive_path = ROOT / 'logs/environment/M0-06_latest.json'
    archive = json.loads(archive_path.read_text('utf-8'))
    assert Path(archive['prefix']).resolve() == Path(sys.prefix).resolve(), 'Archive the active environment first'
    for name, expected in archive['project_files_sha256'].items():
        assert sha(ROOT / name) == expected, f'Record changed; run save_environment.py again: {name}'
    run_id = 'M0-06_' + now().strftime('%Y%m%d_%H%M%S_%f')
    logs = ROOT / 'logs/reproducibility' / run_id
    logs.mkdir(parents=True, exist_ok=False)
    artifacts = ROOT / 'artifacts/reproducibility' / run_id
    artifacts.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONUTF8='1', YOLO_AUTOINSTALL='false', YOLO_OFFLINE='true')
    for key in ['PYTHONPATH', 'PYTHONHOME']:
        env.pop(key, None)
    cwd = ROOT.parent
    steps = []
    summary = {'task': 'M0-06', 'run_id': run_id, 'timestamp': now().isoformat(),
               'parent_pid': os.getpid(), 'python': sys.executable, 'worker_cwd': str(cwd),
               'cwd_is_outside_project': cwd != ROOT and ROOT not in cwd.parents,
               'archive': archive['archive'], 'steps': steps, 'passed': False,
               'scope': 'Fresh processes on this existing workstation; no clean environment reinstall or other-machine test.'}

    def execute(name, command, timeout):
        log = logs / f'{name}.log'
        started = time.perf_counter()
        print(f'Starting {name}; log: {log}', flush=True)
        with log.open('w', encoding='utf-8') as stream:
            process = subprocess.Popen([str(x) for x in command], cwd=cwd, env=env,
                                       stdout=stream, stderr=subprocess.STDOUT)
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                code = -1
        step = {'name': name, 'pid': process.pid, 'cwd': str(cwd),
                'command': [str(x) for x in command], 'exit_code': code,
                'elapsed_seconds': round(time.perf_counter() - started, 3),
                'log': str(log.relative_to(ROOT))}
        steps.append(step)
        print(f'{name}: exit={code}; seconds={step["elapsed_seconds"]}', flush=True)
        if code:
            raise RuntimeError(f'{name} failed: {log.read_text("utf-8")[-3500:]}')
        return step

    python = [sys.executable, '-X', 'utf8']
    try:
        execute('source', python + [ROOT / 'scripts/setup_source.py'], 90)
        training_path = logs / 'training.json'
        execute('training', python + [Path(__file__).resolve(), '--training-worker', training_path], 120)
        training = json.loads(training_path.read_text('utf-8'))
        expected_packages = json.loads((ROOT / archive['archive'] / 'packages.json').read_text('utf-8'))
        assert training['packages'] == expected_packages, 'Installed packages changed since archive'
        summary['training'] = {'passed': training['passed'], 'packages_unchanged': True,
                               'report': str(training_path.relative_to(ROOT)), 'gpu': training['gpu']}
        public_image = ROOT / 'demo/input/E0/bus.jpg'
        upstream_image = ROOT / 'third_party/ultralytics/ultralytics/assets/bus.jpg'
        assert public_image.is_file(), 'Run prepare_e0.py to restore the public sample'
        assert sha(public_image) == sha(upstream_image), 'Public sample differs from locked source'
        cases = [{'name': 'bus', 'image': str(public_image), 'all_classes': True}]
        if args.desktop:
            desktop = resolve_path(args.desktop)
            assert desktop.is_file(), desktop
            cases.append({'name': 'desktop', 'image': str(desktop), 'all_classes': False})
        summary['e0'] = []
        for case in cases:
            destination = ROOT / 'runs/E0' / f'{run_id}_{case["name"]}'
            command = python + [ROOT / 'scripts/predict_e0.py', '--source', case['image'], '--output', destination]
            if case['all_classes']:
                command.append('--all-classes')
            execute(f'e0_{case["name"]}', command, 120)
            report = json.loads((destination / 'summary.json').read_text('utf-8'))
            frames = [json.loads(line) for line in (destination / 'frames.jsonl').read_text('utf-8').splitlines()]
            assert report['status'] == 'passed' and report['frames_processed'] == len(frames) == 1
            assert report['model']['execution_device'] == 'cuda:0' and report['model']['fp16'] is False
            assert report['model']['class_count'] == 80
            assert frames[0]['detections'] and frames[0]['mask_device'] == 'cuda:0'
            assert all(item['mask_area_pixels'] > 0 for item in frames[0]['detections'])
            if case['name'] == 'bus':
                assert report['detections_by_name'].get('bus', 0) and report['detections_by_name'].get('person', 0)
            summary['e0'].append({'name': case['name'], 'passed': True,
                                  'image_sha256': sha(case['image']),
                                  'output': str(destination.relative_to(ROOT)),
                                  'detections_by_name': report['detections_by_name'],
                                  'mask_count': len(frames[0]['detections'])})
        config = yaml.safe_load((ROOT / 'configs/deploy-precheck.yaml').read_text('utf-8'))
        config.update({'output_dir': str(artifacts.relative_to(ROOT)),
                       'report_dir': str((logs / 'deployment').relative_to(ROOT)),
                       'cases': [{'name': c['name'], 'image': c['image']} for c in cases]})
        config_path = logs / 'deployment_config.yaml'
        config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding='utf-8')
        summary['onnx'] = {}
        for stage in ['export', 'ort_cpu', 'ort_cuda']:
            execute(stage, python + [ROOT / 'scripts/check_deployment.py', '--config', config_path,
                                     '--stage', stage, '--worker'], config['timeouts'][stage])
            path = logs / 'deployment' / f'M0-05_{stage}.json'
            report = json.loads(path.read_text('utf-8'))
            assert report['passed'], report
            summary['onnx'][stage] = {'passed': True, 'report': str(path.relative_to(ROOT)),
                                      'onnx_sha256': report['onnx_sha256'],
                                      'reference_manifest_sha256': report['reference_manifest_sha256'],
                                      'profile_node_events_by_provider': report.get('profile_node_events_by_provider'),
                                      'cases': report['cases']}
        identities = {(x['onnx_sha256'], x['reference_manifest_sha256']) for x in summary['onnx'].values()}
        assert len(identities) == 1
        summary['same_onnx_and_inputs'] = True
        summary['artifact_dir'] = str(artifacts.relative_to(ROOT))
        summary['fresh_worker_count'] = len(steps)
        summary['distinct_worker_pids'] = len({s['pid'] for s in steps}) == len(steps)
        summary['passed'] = True
    except Exception as error:
        summary.update({'error': str(error), 'traceback': traceback.format_exc()})
    summary['finished_at'] = now().isoformat()
    write_json(logs / 'summary.json', summary)
    write_json(ROOT / 'logs/reproducibility/M0-06_latest.json', summary)
    archive.update({'fresh_process_reproduction': 'passed' if summary['passed'] else 'failed',
                    'reproduction_summary': str((logs / 'summary.json').relative_to(ROOT))})
    write_json(archive_path, archive)
    write_json(ROOT / archive['archive'] / 'archive.json', archive)
    print(json.dumps({k: v for k, v in summary.items()
                      if k in ['passed', 'error', 'fresh_worker_count', 'worker_cwd', 'artifact_dir', 'e0']},
                     ensure_ascii=False, indent=2), flush=True)
    return 0 if summary['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--desktop', help='Optional desktop image; default checks the public bus image')
    parser.add_argument('--training-worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.training_worker:
        training_worker(args.training_worker)
        return 0
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
