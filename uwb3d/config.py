import logging
import os
from pathlib import Path
import sys
import warnings

import yaml


def load_path_config():
    project_dir = Path(__file__).parent.parent
    config_path = project_dir / "config" / "paths.yaml"
    default_config_path = project_dir / "config" / "default_paths.yaml"
    if not config_path.is_file():
        warnings.warn(f"Path config file not found: {config_path}, using default path config {default_config_path}\nMake a sure to create your own config/paths.yaml for local paths.")
        config_path = default_config_path
    with open(default_config_path, "r") as f:
        default_config = yaml.safe_load(f)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    for key, value in config.items():
        config[key] = Path(value)
        if not config[key].exists():
            try:
                config[key].mkdir(parents=True, exist_ok=True)
            except Exception as e:
                warnings.warn(f"Could not create directory {config[key]}: {e}\nUsing default path for {key}: {default_config[key]}")
                config[key] = Path(default_config[key])
    return config


paths_config = load_path_config()

DATA_DIR = paths_config["DATA_DIR"]
OUTPUT_DIR = paths_config["OUTPUT_DIR"]
LOG_DIR = paths_config["LOG_DIR"]
