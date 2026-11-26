"""
utils/config_loader.py
========================
Configuration loader for RATFA.
"""

import logging
import os
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "config.yaml",
)


def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load and return the RATFA configuration from a YAML file.

    Parameters
    ----------
    config_path : str
        Path to the config.yaml file.

    Returns
    -------
    dict
        Configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.debug("Configuration loaded from %s", config_path)
    return config

