"""YAML config loading for the quanta reconstruction run scripts.

Settings for ``run_uwb3d_quanta.py`` / ``run_uwb1d_quanta.py`` live under
``config/`` and are split into three kinds:

* data-type configs (``config/data_types/*.yaml``): properties intrinsic to a
  source data type, e.g. ``src_H``, ``src_W``, ``src_fps`` and ``data_subdir``.
* run-parameter configs (``config/runs/*.yaml``): choices for a particular run,
  e.g. datanames, target fps, thinning probabilities, frame/recon slices and
  (3D only) window settings.
* ``config/recon_slices.yaml``: per-dataset reconstruction windows keyed by
  ``(dataname, target_fps)``.

Slices are written in YAML as a time-axis list ``[start, stop]`` or
``[start, stop, step]`` (``null`` for a full/unspecified slice).
"""
from pathlib import Path

import yaml

_PROJECT_DIR = Path(__file__).parent.parent
CONFIG_DIR = _PROJECT_DIR / "config"
DEFAULT_RECON_SLICES = CONFIG_DIR / "recon_slices.yaml"


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_slice(spec):
    """Convert a YAML slice spec to a bare time-axis ``slice`` (or ``None``).

    ``None`` -> ``None``; ``[start, stop]`` or ``[start, stop, step]`` -> slice.
    """
    if spec is None:
        return None
    if len(spec) == 2:
        start, stop = spec
        step = None
    elif len(spec) == 3:
        start, stop, step = spec
    else:
        raise ValueError(f"slice spec must have 2 or 3 elements, got {spec!r}")
    return slice(start, stop, step)


def as_txy_slice(time_slice):
    """Wrap a bare time-axis slice into a ``(T, H, W)`` tuple, or pass ``None``."""
    if time_slice is None:
        return None
    return (time_slice, slice(None), slice(None))


def load_recon_slices(path=DEFAULT_RECON_SLICES):
    """Load per-dataset time-axis recon slices keyed by ``(dataname, fps)``."""
    configs = {}
    for entry in load_yaml(path):
        configs[(entry["dataname"], entry["fps"])] = parse_slice(entry["recon_slice"])
    return configs
