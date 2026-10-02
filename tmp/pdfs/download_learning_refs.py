from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import Request, urlopen
from io import BytesIO
import hashlib
import json
import re
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / 'ref'
EXTRACT = ROOT / 'tmp' / 'pdfs' / 'learning_refs'
EXTRACT.mkdir(parents=True, exist_ok=True)
PAPERS = [
    ('YOLO9000_2016_1612.08242.pdf', '1612.08242', 'YOLO9000: Better, Faster, Stronger'),
    ('YOLOv3_2018_1804.02767.pdf', '1804.02767', 'YOLOv3: An Incremental Improvement'),
    ('YOLOX_2021_2107.08430.pdf', '2107.08430', 'YOLOX: Exceeding YOLO Series in 2021'),
    ('GFL_2020_2006.04388.pdf', '2006.04388', 'Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection'),
    ('TOOD_2021_2108.07755.pdf', '2108.07755', 'TOOD: Task-aligned One-stage Object Detection'),
    ('YOLACT_2019_1904.02689.pdf', '1904.02689', 'YOLACT: Real-time Instance Segmentation'),
    ('YOLOv7_2022_2207.02696.pdf', '2207.02696', 'YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors'),
    ('YOLOv10_2024_2405.14458.pdf', '2405.14458', 'YOLOv10: Real-Time End-to-End Object Detection'),
    ('YOLOv12_2025_2502.12524.pdf', '2502.12524', 'YOLOv12: Attention-Centric Real-Time Object Detectors'),
    ('YOLO26_2026_2606.03748.pdf', '2606.03748', 'Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models'),
]

def download(item):
    filename, arxiv_id, title = item
    path = REF / filename
    url = 'https://arxiv.org/pdf/' + arxiv_id
    if path.exists():
        data = path.read_bytes()
    else:
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0 (research reference download)'})
        with urlopen(req, timeout=60) as response:
            data = response.read()
    if not data.startswith(b'%PDF-'):
        raise ValueError(f'{filename}: response is not a PDF')
    reader = PdfReader(BytesIO(data))
    pages = [p.extract_text() or '' for p in reader.pages]
    version = re.search(re.escape(arxiv_id) + r'v\d+', '\n'.join(pages[:2]))
    if arxiv_id not in '\n'.join(pages[:2]):
        raise ValueError(f'{filename}: arXiv identity was not found in first two pages')
    if not path.exists():
        path.write_bytes(data)
    (EXTRACT / (path.stem + '.txt')).write_text('\n\n'.join(f'=== PAGE {i+1} ===\n{text}' for i,text in enumerate(pages)), encoding='utf-8')
    return dict(filename=filename, arxiv_id=arxiv_id, title=title,
                version=version.group() if version else 'not extracted', pages=len(pages),
                bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                landing_url='https://arxiv.org/abs/'+arxiv_id, download_url=url,
                fetched_date='2026-09-30')

results, errors = [], []
with ThreadPoolExecutor(max_workers=3) as pool:
    futures = {pool.submit(download, p): p for p in PAPERS}
    for future in as_completed(futures):
        try:
            result = future.result()
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
        except Exception as exc:
            errors.append({'filename':futures[future][0], 'error':str(exc)})
            print(json.dumps(errors[-1], ensure_ascii=False), flush=True)
results.sort(key=lambda r:r['filename'])
(REF / 'papers_manifest.json').write_text(json.dumps({'fetched_date':'2026-09-30','papers':results,'errors':errors}, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
if errors:
    raise SystemExit(1)
