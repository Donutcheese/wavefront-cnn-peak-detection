"""同步配置：优先环境变量，其次 cloudbase.local.json（兼容冻结 exe）。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from .runtime_paths import resolve_config_candidates, resolve_writable_label_dir


@dataclass(frozen=True)
class SyncConfig:
    backend: str  # "local_mirror" | "cloudbase"
    env_id: str = ""
    secret_id: str = ""
    secret_key: str = ""
    session_token: str = ""
    region: str = "ap-shanghai"
    dataset: str = "ningxia_core"
    storage_key: str = "wavefront/ningxia_core/signals/waveforms.h5"
    samples_collection: str = "wf_samples"
    labels_collection: str = "wf_phase_labels"
    mirror_root: str = ""
    timeout_s: float = 120.0
    annotator: str = "wavefront_operator"

    def require_cloudbase_creds(self) -> None:
        if self.backend != "cloudbase":
            return
        missing = [
            name
            for name, value in (
                ("env_id", self.env_id),
                ("secret_id", self.secret_id),
                ("secret_key", self.secret_key),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                "CloudBase 后端缺少配置: "
                + ", ".join(missing)
                + "。请在 exe 旁放置 cloudbase.local.json 或重新登录导出临时密钥。"
            )


def load_sync_config(path: str | Path | None = None) -> SyncConfig:
    payload: dict = {}
    config_path: Path | None = Path(path) if path else None
    if config_path is None:
        for candidate in resolve_config_candidates():
            if candidate.is_file():
                config_path = candidate
                break
    if config_path is not None and config_path.is_file():
        payload = json.loads(config_path.read_text(encoding="utf-8"))

    backend = str(
        os.environ.get("TCB_BACKEND")
        or payload.get("backend")
        or "cloudbase"
    ).strip()
    mirror_default = str(resolve_writable_label_dir() / "cloudbase_mirror")
    return SyncConfig(
        backend=backend,
        env_id=str(os.environ.get("TCB_ENV") or payload.get("env_id") or ""),
        secret_id=str(
            os.environ.get("TENCENTCLOUD_SECRETID")
            or os.environ.get("TCB_SECRET_ID")
            or payload.get("secret_id")
            or ""
        ),
        secret_key=str(
            os.environ.get("TENCENTCLOUD_SECRETKEY")
            or os.environ.get("TCB_SECRET_KEY")
            or payload.get("secret_key")
            or ""
        ),
        session_token=str(
            os.environ.get("TENCENTCLOUD_SESSIONTOKEN")
            or os.environ.get("TCB_SESSION_TOKEN")
            or payload.get("session_token")
            or ""
        ),
        region=str(payload.get("region") or "ap-shanghai"),
        dataset=str(payload.get("dataset") or "ningxia_core"),
        storage_key=str(
            payload.get("storage_key") or "wavefront/ningxia_core/signals/waveforms.h5"
        ),
        samples_collection=str(payload.get("samples_collection") or "wf_samples"),
        labels_collection=str(payload.get("labels_collection") or "wf_phase_labels"),
        mirror_root=str(payload.get("mirror_root") or mirror_default),
        timeout_s=float(payload.get("timeout_s") or 180.0),
        annotator=str(payload.get("annotator") or "wavefront_operator"),
    )
