"""腾讯云 CloudBase OpenAPI 客户端（管理员密钥）。"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

from .config import SyncConfig


def _sha256_hex(message: str) -> str:
    return hashlib.sha256(message.encode("utf-8")).hexdigest()


def _hmac_sha256(key: bytes, message: str) -> bytes:
    return hmac.new(key, message.encode("utf-8"), hashlib.sha256).digest()


def build_cloudbase_auth_headers(secret_id: str, secret_key: str, session_token: str = "") -> dict[str, str]:
    """按 CloudBase OpenAPI 文档计算 X-CloudBase-Authorization。"""
    service = "tcb"
    version = "1.0"
    algorithm = "TC3-HMAC-SHA256"
    timestamp = int(time.time())
    date = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d")
    signed_headers = "content-type;host"
    canonical_request = (
        "POST\n"
        "//api.tcloudbase.com/\n"
        "\n"
        "content-type:application/json; charset=utf-8\n"
        "host:api.tcloudbase.com\n"
        "\n"
        f"{signed_headers}\n"
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )
    credential_scope = f"{date}/{service}/tc3_request"
    string_to_sign = (
        f"{algorithm}\n{timestamp}\n{credential_scope}\n{_sha256_hex(canonical_request)}"
    )
    secret_date = _hmac_sha256(("TC3" + secret_key).encode("utf-8"), date)
    secret_service = _hmac_sha256(secret_date, service)
    secret_signing = _hmac_sha256(secret_service, "tc3_request")
    signature = hmac.new(secret_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
    authorization = (
        f"{algorithm} Credential={secret_id}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )
    return {
        "X-CloudBase-Authorization": f"{version} {authorization}",
        "X-CloudBase-SessionToken": session_token or "",
        "X-CloudBase-TimeStamp": str(timestamp),
        "Content-Type": "application/json",
    }


class CloudBaseBackend:
    """OpenAPI：setDocument / find / 云存储上传下载。"""

    BASE = "https://tcb-api.tencentcloudapi.com/api/v2"

    def __init__(self, config: SyncConfig) -> None:
        config.require_cloudbase_creds()
        self.config = config
        self.session = requests.Session()

    def _headers(self) -> dict[str, str]:
        return build_cloudbase_auth_headers(
            self.config.secret_id,
            self.config.secret_key,
            self.config.session_token,
        )

    def _url(self, path: str) -> str:
        return f"{self.BASE}/envs/{self.config.env_id}{path}"

    def _request(self, method: str, path: str, *, json_body: dict | None = None, params: dict | None = None) -> dict:
        response = self.session.request(
            method,
            self._url(path),
            headers=self._headers(),
            json=json_body,
            params=params,
            timeout=self.config.timeout_s,
        )
        try:
            payload = response.json()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"CloudBase 响应非 JSON: HTTP {response.status_code} {response.text[:300]}") from exc
        if response.status_code >= 400 or payload.get("code"):
            raise RuntimeError(
                f"CloudBase API 失败: HTTP {response.status_code} "
                f"code={payload.get('code')} message={payload.get('message')} body={payload}"
            )
        return payload

    def ensure_collections(self) -> None:
        # 文档型库通常在首次写入时自动建集合；真云连通性由首次 upsert/upload 校验。
        return

    def upload_file(self, local_path: str, cloud_path: str) -> str:
        meta = self._request(
            "POST",
            "/storages:getUploadMetaData",
            json_body={"path": cloud_path},
        )
        data = meta.get("data") or meta.get("body", {}).get("data") or {}
        if "data" in data and isinstance(data["data"], dict):
            data = data["data"]
        url = data["url"]
        token = data["token"]
        authorization = data["authorization"]
        file_id = data["fileID"]
        cos_file_id = data["cosFileID"]
        with Path(local_path).open("rb") as handle:
            files = {
                "file": (Path(local_path).name, handle, "application/octet-stream"),
            }
            form = {
                "Signature": authorization,
                "x-cos-security-token": token,
                "x-cos-meta-fileid": cos_file_id,
                "key": cloud_path,
            }
            upload = self.session.post(url, data=form, files=files, timeout=max(self.config.timeout_s, 600.0))
        if upload.status_code >= 400:
            raise RuntimeError(f"云存储上传失败: HTTP {upload.status_code} {upload.text[:500]}")
        return str(file_id)

    def download_file(self, cloud_path: str, local_path: str, file_id: str | None = None) -> None:
        if not file_id:
            # CloudBase fileID 通常为 cloud://env.bucket/path
            file_id = f"cloud://{self.config.env_id}.{self.config.env_id}-bucket/{cloud_path}"
        payload = self._request(
            "POST",
            "/storages:batchGetTempUrls",
            json_body={"fileList": [{"fileID": file_id, "maxAge": 7200}]},
        )
        data = payload.get("data") or {}
        file_list = data.get("fileList") or []
        if not file_list:
            raise RuntimeError(f"未拿到下载链接: {payload}")
        item = file_list[0]
        if str(item.get("code", "SUCCESS")).upper() not in {"SUCCESS", "0", ""}:
            raise RuntimeError(f"下载链接失败: {item}")
        temp_url = item["tempFileURL"]
        response = self.session.get(temp_url, timeout=max(self.config.timeout_s, 600.0))
        response.raise_for_status()
        dst = Path(local_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(response.content)

    def upsert_document(self, collection: str, doc_id: str, data: dict[str, Any]) -> None:
        body = dict(data)
        body.pop("_id", None)
        encoded_id = quote(doc_id, safe="")
        self._request(
            "PUT",
            f"/databases/{collection}/documents/{encoded_id}",
            json_body={"data": json.dumps(body, ensure_ascii=False)},
        )

    def upsert_documents_parallel(
        self,
        collection: str,
        docs: list[dict[str, Any]],
        *,
        workers: int = 16,
    ) -> int:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _one(doc: dict[str, Any]) -> None:
            doc_id = str(doc["_id"])
            self.upsert_document(collection, doc_id, doc)

        ok = 0
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = [pool.submit(_one, doc) for doc in docs]
            for future in as_completed(futures):
                future.result()
                ok += 1
        return ok

    def list_documents(
        self,
        collection: str,
        *,
        query: dict[str, Any] | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        docs: list[dict[str, Any]] = []
        skip = 0
        query_obj = query or {}
        while True:
            payload = self._request(
                "POST",
                f"/databases/{collection}/documents:find",
                params={
                    "limit": str(limit),
                    "skip": str(skip),
                    "fields": "{}",
                    "sort": "{}",
                },
                json_body={"query": json.dumps(query_obj, ensure_ascii=False)},
            )
            data = payload.get("data") or {}
            chunk_raw = data.get("list") or []
            chunk: list[dict[str, Any]] = []
            for item in chunk_raw:
                if isinstance(item, str):
                    chunk.append(json.loads(item))
                else:
                    chunk.append(item)
            docs.extend(chunk)
            if len(chunk) < limit:
                break
            skip += limit
        return docs

    def count_documents(self, collection: str, *, query: dict[str, Any] | None = None) -> int:
        return len(self.list_documents(collection, query=query))
