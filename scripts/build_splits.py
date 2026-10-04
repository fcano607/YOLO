"""Freeze debug batches or the reviewed formal sources with predeclared held-out groups."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.product_dataset import locked_source
from app.product_training import build_debug_split, build_formal_splits, build_formal_loading


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--formal", action="store_true", help="Requires all new human reviews; freezes source roles only")
    modes.add_argument("--formal-loading", action="store_true", help="Build the frozen formal YAML/lists and reuse existing copies")
    args = parser.parse_args()
    locked_source()
    manifest = build_formal_loading() if args.formal_loading else (build_formal_splits() if args.formal else build_debug_split())
    print(manifest["dataset_version"], manifest["summary"])
    print(manifest["staging_counts"] if args.formal_loading else (manifest["independence_scope"] if args.formal else manifest["warning"]))


if __name__ == "__main__":
    main()
