#!/usr/bin/env python3
"""将 SkillsBench 任务的 Dockerfile 改为继承 opencode-base 基础镜像。

原始: FROM ubuntu:24.04
替换: FROM opencode-base:latest

这样每个任务启动时不需要重新安装 nvm/node/opencode，节省 5-10 分钟。
"""

import re
from pathlib import Path

SKILLSBENCH_DIR = Path("/tmp/skillsbench")
BASE_IMAGE = "opencode-base:latest"


def patch_dockerfile(dockerfile_path: Path) -> bool:
    """修改 Dockerfile 使用预构建基础镜像。"""
    content = dockerfile_path.read_text()
    original = content

    # 替换 FROM ubuntu:24.04 为我们的基础镜像
    # 但保留多阶段构建中的其他 FROM
    new_content = re.sub(
        r'^FROM\s+ubuntu:\d[\d.]*',
        f'FROM {BASE_IMAGE}',
        content,
        count=1,  # 只替换第一个 FROM
        flags=re.MULTILINE,
    )

    # 删除已在基础镜像中安装的包（避免重复）
    # 删除 curl、python3、python3-pip 等重复安装
    lines = new_content.split('\n')
    skip_next_continuation = False
    filtered = []
    for line in lines:
        # 跳过安装 curl/nvm/node/opencode 的行
        if any(kw in line for kw in [
            'curl -o- https://raw.githubusercontent.com/nvm-sh',
            'nvm install',
            'npm i -g opencode',
            'opencode --version',
        ]):
            skip_next_continuation = line.rstrip().endswith('\\')
            continue
        if skip_next_continuation:
            skip_next_continuation = line.rstrip().endswith('\\')
            continue
        filtered.append(line)

    new_content = '\n'.join(filtered)

    if new_content != original:
        dockerfile_path.write_text(new_content)
        return True
    return False


def main():
    tasks_dir = SKILLSBENCH_DIR / "tasks"
    patched = 0
    errors = 0

    for task_dir in sorted(tasks_dir.iterdir()):
        dockerfile = task_dir / "environment" / "Dockerfile"
        if not dockerfile.exists():
            continue

        try:
            if patch_dockerfile(dockerfile):
                patched += 1
                print(f"  ✓ {task_dir.name}")
            else:
                print(f"  - {task_dir.name} (no change needed)")
        except Exception as e:
            errors += 1
            print(f"  ✗ {task_dir.name}: {e}")

    print(f"\nPatched: {patched}, Errors: {errors}")


if __name__ == "__main__":
    main()
