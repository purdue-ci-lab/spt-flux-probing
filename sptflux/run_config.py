"""YAML config loading for the quanta reconstruction run scripts.

Settings for ``run_uwb3d_quanta.py`` / ``run_uwb1d_quanta.py`` live under
``config/`` and are split into three kinds:

* data-type configs (``config/data_types/*.yaml``): properties intrinsic to a
  source data type, e.g. ``src_H``, ``src_W``, ``src_fps`` and ``data_subdir``.
* run-parameter configs (``config/runs/*.yaml``): choices for a particular run,
  e.g. datanames, target fps, thinning probabilities, frame/recon slices and
  (3D only) window settings. Each run config names its data type with a
  ``data_type`` key (the stem of a file in ``config/data_types``).
* ``config/recon_slices.yaml``: per-dataset reconstruction windows keyed by
  ``(dataname, target_fps)``.

Slices are written in YAML as a list of ``"start:stop"`` / ``"start:stop:step"``
strings, one per ``(T, H, W)`` dimension, with empty fields meaning "open", e.g.
``["0:1000", "100:400", "::2"]``. A bare string (or a 1-element list) slices the
time axis only and leaves H and W full. ``null`` means no slice at all, which for
``recon_slice`` is what triggers the ``config/recon_slices.yaml`` lookup -- write
``"::"`` for an explicitly full slice instead.
"""
from pathlib import Path

import yaml

_PROJECT_DIR = Path(__file__).parent.parent
CONFIG_DIR = _PROJECT_DIR / "config"
DATA_TYPES_DIR = CONFIG_DIR / "data_types"
DEFAULT_RECON_SLICES = CONFIG_DIR / "recon_slices.yaml"


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_data_type(name):
    """Load the data-type config named by a run config's ``data_type`` key."""
    return load_yaml(DATA_TYPES_DIR / f"{name}.yaml")


def parse_dim_slice(spec):
    """Parse one dimension's ``"start:stop"``/``"start:stop:step"`` string."""
    if spec is None:
        return slice(None)
    parts = str(spec).split(":")
    if len(parts) not in (2, 3):
        raise ValueError(
            f"slice must be written with colons as 'start:stop' or "
            f"'start:stop:step', got {spec!r}"
        )
    start, stop, step = (int(p) if p.strip() else None for p in (parts + [""])[:3])
    return slice(start, stop, step)


def parse_slice(spec):
    """Convert a YAML slice spec to a ``(T, H, W)`` tuple of slices (or ``None``).

    ``None`` -> ``None``; a string or 1-element list slices time only; a
    3-element list slices all of ``(T, H, W)``.
    """
    if spec is None:
        return None
    if isinstance(spec, str):
        spec = [spec]
    if len(spec) not in (1, 3):
        raise ValueError(
            f"slice spec must be a string, or a list of 1 (time only) or 3 "
            f"(T, H, W) strings, got {spec!r}"
        )
    dims = list(spec) + [None] * (3 - len(spec))
    return tuple(parse_dim_slice(d) for d in dims)


def dim_slice_str(dim_slice):
    """Write one ``slice`` back out in the config's ``start:stop[:step]`` format."""
    body = ":".join("" if v is None else str(v) for v in (dim_slice.start, dim_slice.stop))
    if dim_slice.step is not None:
        body += f":{dim_slice.step}"
    return body


def slice_strs(txy_slice):
    """A ``(T, H, W)`` slice as the ``["t", "y", "x"]`` colon strings configs use.

    ``None`` (no slice) is written out as three fully open dimensions.
    """
    if txy_slice is None:
        txy_slice = (slice(None),) * 3
    return [dim_slice_str(s) for s in txy_slice]


def is_full_slice(dim_slice):
    """Whether one dimension's slice keeps everything."""
    return dim_slice.start is None and dim_slice.stop is None and dim_slice.step in (None, 1)


def hw_slice_name(txy_slice):
    """Name the H/W parts of a ``(T, H, W)`` slice; empty when they are full."""
    if txy_slice is None:
        return ""
    name = ""
    for axis, s in zip(("y", "x"), txy_slice[1:]):
        if is_full_slice(s):
            continue
        name += f"_{axis}{s.start}_{s.stop}"
        if s.step is not None and s.step != 1:
            name += f"_{s.step}"
    return name


def frame_slice_name(txy_slice):
    """Name a frame slice for an output directory; empty when nothing is sliced."""
    if txy_slice is None:
        return ""
    t = txy_slice[0]
    name = ""
    if not is_full_slice(t):
        name += f"{t.start}_{t.stop}"
        if t.step is not None and t.step != 1:
            name += f"_{t.step}"
    name += hw_slice_name(txy_slice)
    return f"frame{name}" if name else ""


def load_recon_slices(path=DEFAULT_RECON_SLICES):
    """Load per-dataset ``(T, H, W)`` recon slices keyed by ``(dataname, fps)``."""
    configs = {}
    for entry in load_yaml(path):
        configs[(entry["dataname"], entry["fps"])] = parse_slice(entry["recon_slice"])
    return configs
