"""Restore the exact Ultralytics source specified in configs/source-lock.json."""

import json
from pathlib import Path
import subprocess


def main():
    root = Path(__file__).resolve().parents[1]
    lock = json.loads((root / "configs/source-lock.json").read_text(encoding="utf-8"))
    source = (root / lock["path"]).resolve()
    source.relative_to(root)
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", lock["tag"], lock["repository"], str(source)],
            check=True,
        )
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != lock["commit"]:
        raise RuntimeError(f"Source commit differs from source-lock.json: {commit}")
    for relative_patch in lock["patches"]:
        patch = (root / relative_patch).resolve()
        patch.relative_to(root)
        already_applied = subprocess.run(
            ["git", "-C", str(source), "apply", "--reverse", "--check", str(patch)],
            capture_output=True,
        ).returncode == 0
        if not already_applied:
            subprocess.run(["git", "-C", str(source), "apply", "--check", str(patch)], check=True)
            subprocess.run(["git", "-C", str(source), "apply", str(patch)], check=True)
    print(f"Ultralytics {lock['version']} | {commit} | {source}")


if __name__ == "__main__":
    main()
