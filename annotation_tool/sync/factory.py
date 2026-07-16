"""按配置创建同步后端。"""

from __future__ import annotations

from .cloudbase_client import CloudBaseBackend
from .config import SyncConfig
from .local_mirror import LocalMirrorBackend


def create_backend(config: SyncConfig):
    if config.backend == "local_mirror":
        return LocalMirrorBackend(config.mirror_root)
    if config.backend == "cloudbase":
        return CloudBaseBackend(config)
    raise ValueError(f"未知 backend: {config.backend!r}，应为 local_mirror 或 cloudbase")
