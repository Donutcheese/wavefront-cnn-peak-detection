"""CloudBase 同步适配层（OpenAPI 真云 + 本地镜像后端）。"""

from .config import SyncConfig, load_sync_config
from .factory import create_backend

__all__ = ["SyncConfig", "load_sync_config", "create_backend"]
