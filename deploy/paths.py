"""Resolve project paths independently of the shell's current directory."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value):
    """Resolve an absolute path or a path relative to PROJECT_ROOT."""
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def load_paths(config=None):
    """Load the named paths from configs/paths.yaml or a supplied YAML file."""
    config_path = resolve_path(config or "configs/paths.yaml")
    with config_path.open(encoding="utf-8") as stream:
        values = yaml.safe_load(stream)["paths"]
    return {name: resolve_path(value) for name, value in values.items()}
