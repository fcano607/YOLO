"""M0-06: record the current environment and generate Windows recovery files.

Exports are evidence, not a binary copy of the installed environment. This script
does not install packages, modify Conda environments, or access any credentials.
"""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def timestamp():
    return datetime.now(timezone(timedelta(hours=8)))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def save(conda):
    import torch
    import yaml
    from packaging.requirements import Requirement

    assert sys.version_info[:2] == (3, 9), 'Run with the verified Python 3.9 environment'
    assert torch.__version__ == '2.8.0+cu129' and torch.cuda.is_available()
    run_id = 'M0-06_' + timestamp().strftime('%Y%m%d_%H%M%S_%f')
    out = ROOT / 'logs/environment' / run_id
    out.mkdir(parents=True, exist_ok=False)
    commands = []
    env = dict(os.environ, PYTHONUTF8='1')

    def capture(args, filename):
        result = subprocess.run([str(arg) for arg in args], cwd=ROOT, env=env,
                                capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=90)
        (out / filename).write_text(result.stdout, encoding='utf-8')
        if result.stderr:
            (out / (filename + '.stderr.txt')).write_text(result.stderr, encoding='utf-8')
        commands.append({'args': [str(x) for x in args], 'exit_code': result.returncode, 'output': filename})
        if result.returncode:
            raise RuntimeError(f'{filename} failed: {result.stderr[-2000:]}')
        return result.stdout

    rows = json.loads(capture([conda, 'list', '--prefix', sys.prefix, '--json'], 'conda_list.json'))
    full = capture([conda, 'env', 'export', '--prefix', sys.prefix], 'conda_full.yml')
    capture([conda, 'list', '--prefix', sys.prefix, '--explicit'], 'conda_explicit_win64.txt')
    capture([sys.executable, '-m', 'pip', 'freeze', '--all'], 'pip_freeze_all.txt')
    capture([sys.executable, '-m', 'pip', 'check'], 'pip_check.txt')
    conda_version = capture([conda, '--version'], 'conda_version.txt').strip()
    gpu_info = capture(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,compute_cap',
                        '--format=csv,noheader'], 'nvidia_smi.txt').strip()
    packages = dict(sorted({d.metadata['Name']: d.version for d in metadata.distributions()}.items(),
                           key=lambda item: item[0].lower()))
    write_json(out / 'packages.json', packages)
    parsed = yaml.safe_load(full)
    assert parsed['name'] == Path(sys.prefix).name
    # A pip package can shadow a Conda entry in `conda list` (tzdata here).
    # The original Conda registrations remain in conda-meta and explicit export.
    conda_packages = []
    for path in sorted((Path(sys.prefix) / 'conda-meta').glob('*.json')):
        item = json.loads(path.read_text('utf-8'))
        conda_packages.append({'name': item['name'], 'version': item['version'],
                               'build_string': item['build'], 'channel': item.get('channel'),
                               'url': item.get('url'), 'md5': item.get('md5'), 'sha256': item.get('sha256')})
    write_json(out / 'conda_registered_packages.json', conda_packages)
    assert next(x['version'] for x in conda_packages if x['name'] == 'python') == '3.9.23'
    # Bootstrap only Conda-managed packages. Pip packages use the separate locked
    # requirements after GPU torch and project source are restored.
    recipe = {'name': 'yolo', 'channels': ['defaults'],
              'dependencies': [f'{row["name"]}={row["version"]}={row["build_string"]}'
                               for row in conda_packages]}
    environment_file = ROOT / 'environment.yml'
    constraint = ROOT / 'configs/constraints-runtime.txt'
    locked = {}
    for name in ['requirements-train.txt', 'requirements-deploy.txt', 'configs/constraints-train.txt']:
        for line in (ROOT / name).read_text('utf-8').splitlines():
            line = line.strip()
            if not line or line.startswith(('#', '-')):
                continue
            req = Requirement(line)
            version = metadata.version(req.name)
            assert version in req.specifier, (req.name, version)
            locked[req.name] = version
    lock = json.loads((ROOT / 'configs/source-lock.json').read_text('utf-8'))
    source = ROOT / lock['path']
    commit = capture(['git', '-C', source, 'rev-parse', 'HEAD'], 'source_commit.txt').strip()
    status = capture(['git', '-C', source, 'status', '--porcelain'], 'source_status.txt').strip()
    assert commit == lock['commit'] and not status, 'Unexpected upstream source modifications'
    e0 = yaml.safe_load((ROOT / 'configs/e0.yaml').read_text('utf-8'))
    weight = ROOT / e0['weights']['path']
    assert sha(weight) == e0['weights']['sha256']
    deployment = yaml.safe_load((ROOT / 'configs/deploy-precheck.yaml').read_text('utf-8'))
    model_dir = ROOT / deployment['output_dir']
    export_path = model_dir / 'export.json'
    onnx_path = model_dir / 'yolo11n-seg.onnx'
    export = None
    if export_path.is_file() and onnx_path.is_file():
        export = json.loads(export_path.read_text('utf-8'))
        assert sha(onnx_path) == export['onnx']['sha256']
    # Validate the active setup before updating its checked-in recovery files.
    environment_file.write_text('# Windows x64 Conda bootstrap; see docs/环境恢复与复现说明.md.\n'
                                '# Install GPU torch, locked source and pip requirements afterwards.\n'
                                + yaml.safe_dump(recipe, sort_keys=False, allow_unicode=True), encoding='utf-8')
    constraint.write_text('# Current Windows / Python 3.9 environment, recorded by M0-06.\n'
                          '# Constraints pin versions only when a package is required; they do not install all inherited packages.\n'
                          '# GPU torch uses the cu129 index; Ultralytics uses the locked editable source.\n'
                          + ''.join(f'{name}=={version}\n' for name, version in packages.items()), encoding='utf-8')
    write_json(out / 'source-lock.json', lock)
    write_json(out / 'model_origins.json', {
        'e0': e0['weights'],
        'onnx': export['onnx'] if export else {'available': False, 'generated_by': 'scripts/verify_reproducibility.py'},
        'cases': export['cases'] if export else [],
        'deployment_record': 'logs/deployment/M0-05_summary.json'
        if (ROOT / 'logs/deployment/M0-05_summary.json').is_file() else None})
    files = ['environment.yml', 'configs/constraints-runtime.txt', 'requirements-train.txt',
             'requirements-deploy.txt', 'configs/constraints-train.txt', 'configs/source-lock.json',
             'configs/e0.yaml', 'configs/deploy-precheck.yaml', 'configs/paths.yaml',
             'scripts/save_environment.py', 'scripts/verify_reproducibility.py', 'scripts/setup_source.py',
             'scripts/prepare_e0.py', 'scripts/predict_e0.py', 'scripts/check_deployment.py', 'deploy/paths.py']
    report = {'task': 'M0-06', 'run_id': run_id, 'timestamp': timestamp().isoformat(),
              'environment_saved': True, 'fresh_process_reproduction': 'pending',
              'python': sys.version, 'executable': sys.executable, 'prefix': sys.prefix,
              'platform': platform.platform(), 'conda_version': conda_version,
              'package_count': len(packages), 'conda_managed_count': len(conda_packages),
              'conda_list_non_pypi_count': len([x for x in rows if x['channel'] != 'pypi']),
              'gpu': {'nvidia_smi': gpu_info, 'name': torch.cuda.get_device_name(0),
                      'capability': list(torch.cuda.get_device_capability(0)), 'torch': torch.__version__,
                      'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version()},
              'source': lock, 'source_clean': True, 'pinned_direct_packages': locked,
              'project_files_sha256': {name: sha(ROOT / name) for name in files},
              'commands': commands, 'archive': str(out.relative_to(ROOT)),
              'scope': 'Current environment records and recovery recipe; no clean reinstall or other-machine test.'}
    write_json(out / 'archive.json', report)
    write_json(ROOT / 'logs/environment/M0-06_latest.json', report)
    print(json.dumps({'environment_saved': True, 'archive': str(out),
                      'package_count': len(packages), 'conda_managed_count': len(conda_packages),
                      'recipe': str(environment_file), 'constraints': str(constraint)}, ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--conda', type=Path,
                        default=Path(sys.prefix).parents[1] / 'Scripts/conda.exe')
    args = parser.parse_args()
    if not args.conda.is_file():
        parser.error('Conda executable not found; pass --conda with its absolute path')
    save(args.conda)


if __name__ == '__main__':
    main()
