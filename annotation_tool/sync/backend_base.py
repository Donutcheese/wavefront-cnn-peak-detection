"""同步后端抽象接口。"""

from __future__ import annotations

from typing import Any, Protocol


class SyncBackend(Protocol):
    def ensure_collections(self) -> None: ...

    def upload_file(self, local_path: str, cloud_path: str) -> str:
        """上传文件，返回 file_id 或本地镜像路径标识。"""

    def download_file(self, cloud_path: str, local_path: str, file_id: str | None = None) -> None: ...

    def upsert_document(self, collection: str, doc_id: str, data: dict[str, Any]) -> None: ...

    def list_documents(self, collection: str, *, query: dict[str, Any] | None = None) -> list[dict[str, Any]]: ...

    def count_documents(self, collection: str, *, query: dict[str, Any] | None = None) -> int: ...
