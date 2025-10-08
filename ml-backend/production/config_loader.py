"""
production/config_loader.py
---------------------------
Load YAML configuration for Patra ML backend.
"""

import yaml
from functools import lru_cache

CONFIG_PATH = "production/settings.yaml"

@lru_cache(maxsize=1)
def load_config() -> dict:
    """Load and cache settings.yaml as a dictionary."""
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    return config
