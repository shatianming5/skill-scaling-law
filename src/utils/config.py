"""YAML 配置加载与合并。

支持 _base_ 继承机制：子配置通过 _base_ 字段继承父配置，
子配置中的值覆盖父配置。
"""

from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    """加载 YAML 配置文件，支持 _base_ 继承。"""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if "_base_" in config:
        base_path = config_path.parent / config.pop("_base_")
        base_config = load_config(str(base_path))
        config = deep_merge(base_config, config)

    return config


def deep_merge(base: dict, override: dict) -> dict:
    """深度合并两个字典，override 覆盖 base。"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result
