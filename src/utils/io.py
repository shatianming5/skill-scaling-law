"""文件 I/O 与结果序列化。"""

import json
from pathlib import Path


def save_json(data, path: str, indent: int = 2):
    """保存 JSON 文件。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=indent, ensure_ascii=False, default=str))


def load_json(path: str):
    """加载 JSON 文件。"""
    return json.loads(Path(path).read_text())
