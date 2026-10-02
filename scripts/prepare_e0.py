"""Restore the pinned E0 weights and built-in image; do not install packages."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deploy.paths import resolve_path
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/e0.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(resolve_path(args.config).read_text(encoding="utf-8"))
    weights = config["weights"]
    destination = resolve_path(weights["path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        partial = destination.with_suffix(".pt.part")
        request = urllib.request.Request(weights["url"], headers={"User-Agent": "YOLO-project-M0-04"})
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=20) as response, partial.open("wb") as stream:
            while True:
                block = response.read(256 * 1024)
                if not block:
                    break
                stream.write(block)
                if time.perf_counter() - started > 90:
                    raise TimeoutError("Weight download exceeded 90 seconds; partial file retained")
        verify_weights(partial, weights)
        partial.replace(destination)
    verify_weights(destination, weights)
    source = ROOT / "third_party/ultralytics/ultralytics/assets/bus.jpg"
    image = ROOT / "demo/input/E0/bus.jpg"
    image.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, image)
    record = {"checked_at": datetime.now().astimezone().isoformat(), "weights": weights,
              "weights_verified": True, "sample": {"path": str(image.relative_to(ROOT)),
              "source": str(source.relative_to(ROOT)),
              "sha256": hashlib.sha256(image.read_bytes()).hexdigest()},
              "hash_note": "Weight hash was recorded from the official download; GitHub supplied no asset digest."}
    output = ROOT / "logs/environment/M0-04_e0_assets.json"
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))


def verify_weights(path, weights):
    if path.stat().st_size != weights["size_bytes"]:
        raise ValueError(f"Unexpected weight size: {path}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != weights["sha256"]:
        raise ValueError(f"Weight SHA256 mismatch: {path}")


if __name__ == "__main__":
    main()
