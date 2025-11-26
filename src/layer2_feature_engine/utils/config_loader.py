"""
Configuration Loader
Load YAML config files
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file

    Args:
        config_path: Path to YAML config file

    Returns:
        Dict containing configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML config: {e}")

    return config


def get_nested_value(config: Dict, key_path: str, default: Any = None) -> Any:
    """
    Get nested value from config dict using dot notation

    Args:
        config: Config dictionary
        key_path: Dot-separated key path (e.g., 'model.transformer.d_model')
        default: Default value if key not found

    Returns:
        Config value or default

    Example:
        >>> config = {'model': {'transformer': {'d_model': 128}}}
        >>> get_nested_value(config, 'model.transformer.d_model')
        128
    """
    keys = key_path.split('.')
    value = config

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default

    return value


def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """
    Merge two config dicts, with override taking precedence

    Args:
        base_config: Base configuration
        override_config: Override configuration

    Returns:
        Merged configuration
    """
    merged = base_config.copy()

    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged
