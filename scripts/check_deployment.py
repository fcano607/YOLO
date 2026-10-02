"""M0-05: export and execute fixed FP32 raw outputs in isolated processes.

This precheck deliberately ends before NMS, mask decoding and accuracy evaluation.
Every backend consumes the exact same saved NCHW tensor; CUDA execution is checked.
"""

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['YOLO_CONFIG_DIR'] = str(ROOT / 'logs/ultralytics_settings')
os.environ['YOLO_AUTOINSTALL'] = 'false'
os.environ['YOLO_OFFLINE'] = 'true'

from deploy.paths import resolve_path
import yaml


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def locations(config):
    out = resolve_path(config['output_dir'])
    reports = resolve_path(config['report_dir'])
    out.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    return out, reports


def compare(reference, actual, config):
    import numpy as np
    result = []
    assert len(reference) == len(actual) == 2
    for index, (ref, pred) in enumerate(zip(reference, actual)):
        assert ref.shape == pred.shape, (ref.shape, pred.shape)
        assert ref.dtype == pred.dtype == np.float32, (ref.dtype, pred.dtype)
        finite = bool(np.isfinite(pred).all())
        error = np.abs(ref.astype(np.float64) - pred.astype(np.float64))
        bound = config['comparison']['atol'] + config['comparison']['rtol'] * np.abs(ref)
        result.append({'name': f'output{index}', 'shape': list(pred.shape), 'dtype': str(pred.dtype),
                       'finite': finite, 'max_abs': float(error.max()), 'mean_abs': float(error.mean()),
                       'p99_abs': float(np.quantile(error, .99)),
                       'outside_tolerance': int(np.count_nonzero(error > bound)),
                       'allclose': bool(finite and np.allclose(ref, pred, **config['comparison']))})
    return result


def load_cases(config, out):
    import numpy as np
    manifest = json.loads((out / 'export.json').read_text('utf-8'))
    assert sha(out / 'yolo11n-seg.onnx') == manifest['onnx']['sha256'], 'ONNX changed since export'
    cases = []
    for item in manifest['cases']:
        path = out / item['tensor_file']
        assert sha(path) == item['tensor_sha256'], 'Saved input/reference changed'
        with np.load(path, allow_pickle=False) as arrays:
            cases.append((item['name'], arrays['images'].copy(),
                          [arrays['output0'].copy(), arrays['output1'].copy()]))
    return cases


def export_reference(config):
    import cv2
    import numpy as np
    import onnx
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox
    from ultralytics.nn.modules import Detect

    out, _ = locations(config)
    weights_config = yaml.safe_load(resolve_path(config['weights_config']).read_text('utf-8'))
    weights = resolve_path(weights_config['weights']['path'])
    assert sha(weights) == weights_config['weights']['sha256'], 'Weights changed'
    lock = json.loads((ROOT / 'configs/source-lock.json').read_text('utf-8'))
    source = resolve_path(lock['path'])
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    assert commit == lock['commit'] and ultralytics.__version__ == lock['version']
    assert Path(ultralytics.__file__).resolve() == source / 'ultralytics/__init__.py'
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    # Export a copy so E0's original weight directory is never modified.
    copy = out / weights.name
    if not copy.exists() or sha(copy) != sha(weights):
        shutil.copyfile(weights, copy)
    wrapper = YOLO(str(copy), task='segment')
    assert len(wrapper.names) == 80 and wrapper.model.model[-1].nc == 80
    exported = Path(wrapper.export(format='onnx', **config['export']))
    assert exported.resolve() == (out / 'yolo11n-seg.onnx').resolve()
    graph = onnx.load(str(exported))
    onnx.checker.check_model(graph, full_check=True)

    def tensor_info(value):
        return {'name': value.name,
                'shape': [dim.dim_value for dim in value.type.tensor_type.shape.dim],
                'onnx_dtype': value.type.tensor_type.elem_type}

    inputs = [tensor_info(x) for x in graph.graph.input]
    outputs = [tensor_info(x) for x in graph.graph.output]
    assert inputs == [{'name': 'images', 'shape': [1, 3, 640, 640], 'onnx_dtype': 1}], inputs
    assert [x['shape'] for x in outputs] == [[1, 116, 8400], [1, 32, 160, 160]], outputs
    assert all(x['onnx_dtype'] == 1 for x in outputs)
    assert not any(node.op_type == 'NonMaxSuppression' for node in graph.graph.node)
    assert graph.ir_version <= 10, graph.ir_version
    # Same fused model and export head mode as the source exporter.
    reference = YOLO(str(copy), task='segment').model.to('cuda:0').float().eval()
    reference.fuse(imgsz=(640, 640), verbose=False)
    for module in reference.modules():
        if isinstance(module, Detect):
            module.dynamic = False
            module.export = True
            module.format = 'onnx'
    cases = []
    for item in config['cases']:
        image_path = resolve_path(item['image'])
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
        padded = LetterBox(new_shape=(640, 640), auto=False, stride=32)(image=image)
        images = np.ascontiguousarray(padded[..., ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        with torch.inference_mode():
            raw = reference(torch.from_numpy(images).to('cuda:0'))
        torch.cuda.synchronize()
        assert len(raw) == 2 and all(torch.isfinite(x).all() for x in raw)
        arrays = [x.cpu().numpy() for x in raw]
        saved = out / f'{item["name"]}_reference.npz'
        np.savez_compressed(saved, images=images, output0=arrays[0], output1=arrays[1])
        cases.append({'name': item['name'], 'image': str(image_path), 'image_sha256': sha(image_path),
                      'original_shape': list(image.shape), 'tensor_file': saved.name,
                      'tensor_sha256': sha(saved), 'input_shape': list(images.shape),
                      'input_sha256': hashlib.sha256(images.tobytes()).hexdigest()})
    report = {'passed': True, 'timestamp': datetime.now().astimezone().isoformat(),
              'weights': {'path': str(weights), 'sha256': sha(weights), 'classes': 80},
              'source_commit': commit, 'ultralytics': ultralytics.__version__, 'torch': torch.__version__,
              'reference_device': 'cuda:0', 'tf32': False, 'options': config['export'],
              'onnx': {'path': str(exported), 'sha256': sha(exported), 'size_bytes': exported.stat().st_size,
                       'version': onnx.__version__, 'ir_version': graph.ir_version,
                       'opsets': {x.domain or 'ai.onnx': x.version for x in graph.opset_import},
                       'inputs': inputs, 'outputs': outputs, 'nms_embedded': False,
                       'metadata': {x.key: x.value for x in graph.metadata_props}}, 'cases': cases}
    write_json(out / 'export.json', report)
    return report


def ort_execute(config, cuda):
    import numpy as np
    if cuda:
        # ORT 1.19 CUDA 12/cuDNN 9 loads the DLLs already distributed with torch.
        # This worker never imports Ultralytics or executes a PyTorch model.
        import torch
        assert torch.cuda.is_available()
    import onnxruntime as ort
    out, reports = locations(config)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.enable_profiling = True
    options.profile_file_prefix = str(out / ('ort_cuda_profile' if cuda else 'ort_cpu_profile'))
    providers = [('CUDAExecutionProvider', {'device_id': 0, 'use_tf32': 0}), 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
    session = ort.InferenceSession(str(out / 'yolo11n-seg.onnx'), sess_options=options, providers=providers)
    if cuda and 'CUDAExecutionProvider' not in session.get_providers():
        raise RuntimeError('CUDA provider failed; CPU fallback does not count as a CUDA pass')
    cases = []
    for name, images, reference in load_cases(config, out):
        raw = session.run(['output0', 'output1'], {'images': images})
        comparisons = compare(reference, raw, config)
        np.savez_compressed(out / f'{name}_ort_{"cuda" if cuda else "cpu"}.npz', output0=raw[0], output1=raw[1])
        cases.append({'name': name, 'comparisons': comparisons,
                      'passed': all(x['allclose'] for x in comparisons)})
    profile = Path(session.end_profiling())
    events = json.loads(profile.read_text('utf-8'))
    provider_counts = Counter(x.get('args', {}).get('provider') for x in events
                              if x.get('cat') == 'Node' and x.get('args', {}).get('provider'))
    if cuda and not provider_counts['CUDAExecutionProvider']:
        raise RuntimeError('Profiling found no nodes executing on CUDA')
    return {'passed': all(x['passed'] for x in cases), 'backend': 'ort_cuda' if cuda else 'ort_cpu',
            'version': ort.__version__, 'available_providers': ort.get_available_providers(),
            'session_providers': session.get_providers(), 'provider_options': session.get_provider_options(),
            'profile': str(profile), 'profile_node_events_by_provider': dict(provider_counts),
            'ultralytics_imported': 'ultralytics' in sys.modules,
            'pytorch_model_executed': False, 'cases': cases, 'tolerance': config['comparison']}


def trt_execute(config):
    import numpy as np
    import torch
    import tensorrt as trt
    out, _ = locations(config)
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    assert builder is not None, 'TensorRT builder unavailable'
    network = builder.create_network(0)  # TensorRT 11 always uses explicit batch and strong typing.
    parser = trt.OnnxParser(network, logger)
    # NVIDIA's Windows native file API fails on this project's Chinese path.
    # The model embeds its weights, so parsing Python-read bytes avoids that API.
    if not parser.parse((out / 'yolo11n-seg.onnx').read_bytes()):
        raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    settings = builder.create_builder_config()
    settings.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(config['trt']['workspace_gib'] * 1024**3))
    settings.clear_flag(trt.BuilderFlag.TF32)
    start = time.perf_counter()
    serialized = builder.build_serialized_network(network, settings)
    assert serialized is not None, 'TensorRT engine build returned None'
    build_seconds = time.perf_counter() - start
    engine_file = out / 'yolo11n-seg.engine'
    engine_file.write_bytes(bytes(serialized))
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_file.read_bytes())
    assert engine is not None
    context = engine.create_execution_context()
    assert context is not None
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    assert set(names) == {'images', 'output0', 'output1'}, names
    tensors = {name: {'shape': list(engine.get_tensor_shape(name)),
                      'dtype': str(engine.get_tensor_dtype(name)),
                      'mode': str(engine.get_tensor_mode(name))} for name in names}
    assert all(engine.get_tensor_dtype(name) == trt.float32 for name in names), tensors
    stream = torch.cuda.Stream()
    cases = []
    for name, images, reference in load_cases(config, out):
        # Keep every device allocation alive until stream completion.
        with torch.cuda.stream(stream):
            buffers = {'images': torch.from_numpy(images).to('cuda:0')}
            for output in ['output0', 'output1']:
                buffers[output] = torch.empty(tuple(engine.get_tensor_shape(output)), dtype=torch.float32, device='cuda:0')
            for tensor_name, buffer in buffers.items():
                assert context.set_tensor_address(tensor_name, buffer.data_ptr())
            assert context.execute_async_v3(stream.cuda_stream), 'TensorRT execution failed'
        stream.synchronize()
        raw = [buffers[x].cpu().numpy() for x in ['output0', 'output1']]
        comparisons = compare(reference, raw, config)
        np.savez_compressed(out / f'{name}_trt.npz', output0=raw[0], output1=raw[1])
        cases.append({'name': name, 'comparisons': comparisons,
                      'passed': all(x['allclose'] for x in comparisons)})
    return {'passed': all(x['passed'] for x in cases), 'version': trt.__version__, 'tf32': False,
            'strongly_typed': True, 'precision': 'FP32 ONNX types', 'workspace_gib': config['trt']['workspace_gib'],
            'build_seconds': build_seconds, 'engine': {'path': str(engine_file), 'sha256': sha(engine_file),
                                                      'size_bytes': engine_file.stat().st_size},
            'gpu': torch.cuda.get_device_name(0), 'compute_capability': list(torch.cuda.get_device_capability(0)),
            'torch_cuda': torch.version.cuda, 'tensors': tensors, 'cases': cases,
            'tolerance': config['comparison'], 'ultralytics_imported': 'ultralytics' in sys.modules,
            'pytorch_model_executed': False,
            'portability': 'Native plan tied to TensorRT version, Windows and the build GPU; no Ultralytics metadata header.'}


def worker(config, stage):
    out, reports = locations(config)
    try:
        if stage == 'export':
            result = export_reference(config)
        elif stage.startswith('ort_'):
            result = ort_execute(config, stage == 'ort_cuda')
        else:
            result = trt_execute(config)
        result.update({'task': 'M0-05', 'stage': stage, 'timestamp': datetime.now().astimezone().isoformat(),
                       'python': sys.executable,
                       'onnx_sha256': sha(out / 'yolo11n-seg.onnx'),
                       'reference_manifest_sha256': sha(out / 'export.json')})
    except Exception as error:
        result = {'task': 'M0-05', 'stage': stage, 'passed': False,
                  'error': str(error), 'traceback': traceback.format_exc(),
                  'timestamp': datetime.now().astimezone().isoformat(), 'python': sys.executable}
    write_json(reports / f'M0-05_{stage}.json', result)
    print(json.dumps({k: v for k, v in result.items() if k in ('passed', 'stage', 'error', 'cases')}, indent=2), flush=True)
    return 0 if result['passed'] else 1


def run_stages(args, config):
    out, reports = locations(config)
    stages = ['export', 'ort_cpu', 'ort_cuda', 'trt'] if args.stage == 'all' else [args.stage]
    failed = False
    for stage in stages:
        log = reports / f'M0-05_{stage}.log'
        command = [sys.executable, '-X', 'utf8', str(Path(__file__).resolve()), '--config', args.config,
                   '--stage', stage, '--worker']
        print(f'Starting {stage}; log: {log}', flush=True)
        started = time.perf_counter()
        write_json(reports / f'M0-05_{stage}.json',
                   {'task': 'M0-05', 'stage': stage, 'passed': False, 'status': 'running'})
        with log.open('w', encoding='utf-8') as stream:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                       env=dict(os.environ, PYTHONUTF8='1'))
            try:
                code = process.wait(timeout=config['timeouts'][stage])
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                code = -1
                write_json(reports / f'M0-05_{stage}.json',
                           {'task': 'M0-05', 'stage': stage, 'passed': False,
                            'error': f'Worker exceeded {config["timeouts"][stage]} seconds', 'exit_code': code})
        print(f'{stage}: exit={code}; seconds={time.perf_counter() - started:.2f}', flush=True)
        if code:
            failed = True
            print(log.read_text('utf-8')[-5000:], flush=True)
            if stage == 'export':
                break
    summary = {'task': 'M0-05', 'timestamp': datetime.now().astimezone().isoformat(),
               'config': str(resolve_path(args.config)), 'stages': {},
               'scope': 'Deployment feasibility and same-input raw tensor parity on two E0 images; no full postprocessing, training, mAP or controlled speed benchmark.'}
    for stage in ['export', 'ort_cpu', 'ort_cuda', 'trt']:
        p = reports / f'M0-05_{stage}.json'
        if p.exists():
            summary['stages'][stage] = json.loads(p.read_text('utf-8'))
    current_onnx = sha(out / 'yolo11n-seg.onnx') if (out / 'yolo11n-seg.onnx').exists() else None
    current_manifest = sha(out / 'export.json') if (out / 'export.json').exists() else None
    summary['same_artifacts_verified'] = all(
        summary['stages'].get(s, {}).get('onnx_sha256') == current_onnx
        and summary['stages'].get(s, {}).get('reference_manifest_sha256') == current_manifest
        and current_onnx is not None and current_manifest is not None
        for s in ['export', 'ort_cpu', 'ort_cuda', 'trt'])
    summary['all_backends_passed'] = summary['same_artifacts_verified'] and all(
        summary['stages'].get(s, {}).get('passed', False) for s in ['export', 'ort_cpu', 'ort_cuda', 'trt'])
    write_json(reports / 'M0-05_summary.json', summary)
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/deploy-precheck.yaml')
    parser.add_argument('--stage', choices=['all', 'export', 'ort_cpu', 'ort_cuda', 'trt'], default='all')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = yaml.safe_load(resolve_path(args.config).read_text('utf-8'))
    if args.worker:
        assert args.stage != 'all'
        return worker(config, args.stage)
    return run_stages(args, config)


if __name__ == '__main__':
    raise SystemExit(main())
