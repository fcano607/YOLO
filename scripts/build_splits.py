"""Freeze debug batches or the reviewed formal sources with predeclared held-out groups."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.product_dataset import locked_source
from app.product_training import build_debug_split, build_formal_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal", action="store_true", help="Requires all new human reviews; freezes source roles only")
    args = parser.parse_args()
    locked_source()
    manifest = build_formal_splits() if args.formal else build_debug_split()
    print(manifest["dataset_version"], manifest["summary"])
    print(manifest["independence_scope"] if args.formal else manifest["warning"])


if __name__ == "__main__":
    main()
