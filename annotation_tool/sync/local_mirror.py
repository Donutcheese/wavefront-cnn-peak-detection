"""本地文件系统镜像：无 CloudBase 账号时验证 push/pull 往返。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


class LocalMirrorBackend:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.storage_root = self.root / "storage"
        self.db_root = self.root / "db"
        self.meta_path = self.root / "meta.json"
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.db_root.mkdir(parents=True, exist_ok=True)
        if not self.meta_path.exists():
            self.meta_path.write_text("{}", encoding="utf-8")

    def ensure_collections(self) -> None:
        for name in ("wf_samples", "wf_phase_labels"):
            (self.db_root / name).mkdir(parents=True, exist_ok=True)

    def _collection_dir(self, collection: str) -> Path:
        path = self.db_root / collection
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _safe_name(self, doc_id: str) -> str:
        return doc_id.replace("/", "_").replace("\\", "_").replace(":", "__")

    def upload_file(self, local_path: str, cloud_path: str) -> str:
        src = Path(local_path)
        if not src.is_file():
            raise FileNotFoundError(local_path)
        dst = self.storage_root / cloud_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        file_id = f"mirror://{cloud_path}"
        meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        meta[cloud_path] = {"file_id": file_id, "size": src.stat().st_size}
        self.meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return file_id

    def download_file(self, cloud_path: str, local_path: str, file_id: str | None = None) -> None:
        src = self.storage_root / cloud_path
        if not src.is_file():
            raise FileNotFoundError(f"镜像中不存在对象: {cloud_path}")
        dst = Path(local_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    def upsert_document(self, collection: str, doc_id: str, data: dict[str, Any]) -> None:
        payload = dict(data)
        payload["_id"] = doc_id
        path = self._collection_dir(collection) / f"{self._safe_name(doc_id)}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def upsert_documents_parallel(
        self,
        collection: str,
        docs: list[dict[str, Any]],
        *,
        workers: int = 16,
    ) -> int:
        for doc in docs:
            self.upsert_document(collection, str(doc["_id"]), doc)
        return len(docs)

    def list_documents(self, collection: str, *, query: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        docs: list[dict[str, Any]] = []
        for path in sorted(self._collection_dir(collection).glob("*.json")):
            doc = json.loads(path.read_text(encoding="utf-8"))
            if query and not all(doc.get(key) == value for key, value in query.items()):
                continue
            docs.append(doc)
        return docs

    def count_documents(self, collection: str, *, query: dict[str, Any] | None = None) -> int:
        return len(self.list_documents(collection, query=query))
