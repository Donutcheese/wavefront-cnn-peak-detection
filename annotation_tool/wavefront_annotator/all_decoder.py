"""`.all` / `.vall` 录波文件解码器。

移植自 FaultLocation_demo 的 offline_faultlocation/decode/all_file_decoder.py，
保持二进制布局解析逻辑一致：
- 文本头部以 16 个空格分隔字段，随后首个换行符之后为二进制数据区；
- 每个采样点 6 字节，短录波（<32769 点）按 12bit 压缩布局解包，
  长录波按小端 int16 三通道解包。

与训练管线不同，此处不做补零填充，直接返回原始坐标系波形，
保证 gold 标签落在 `raw_wavefront_index` 语义上。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# 单文件体积上限：正常 16384 点约 98KB，放宽到 8MB 以兼容长录波
MAX_DATA_LENGTH = 8 * 1024 * 1024
# 12bit 压缩布局与 int16 布局的切换阈值（与现场固件约定一致）
PACKED_12BIT_MAX_POINTS = 32769


@dataclass(frozen=True)
class WaveRecord:
    """一次录波的头部信息与三相原始波形。"""

    file_path: str
    file_name: str
    station: int
    line: int
    timestamp_text: str
    gps_frequency: str
    gps_flag: int
    break_flag: int
    startup_type: int
    data_length: int
    sampling_rate_hz: float
    sampling_rate_valid: bool
    signals: np.ndarray  # shape (3, N)，float32，顺序 A/B/C


def _parse_string_field(buf: memoryview, start: int, end: int) -> str:
    start = max(0, start)
    end = min(len(buf), end)
    while start < end and buf[start] in b" \t\r\n":
        start += 1
    while end > start and buf[end - 1] in b" \t\r\n":
        end -= 1
    if end <= start:
        return ""
    return bytes(buf[start:end]).decode("ascii", errors="ignore").strip()


def _parse_int_field(buf: memoryview, start: int, end: int) -> int:
    s = _parse_string_field(buf, start, end)
    try:
        return int(s) if s else 0
    except ValueError:
        return 0


def _binary_payload_start(buf: memoryview, pos16_space: int, path: Path) -> int:
    """自第 16 个空格起向后查找首个 LF，数据区起点为其后一字节。"""
    n = len(buf)
    i0 = max(0, int(pos16_space))
    scan_end = min(n, i0 + 4096)
    for j in range(i0, scan_end):
        if buf[j] == ord("\n"):
            return j + 1
    raise ValueError(f"头部未找到换行符，无法定位二进制数据起点: {path}")


def decode_all_file(path: str | Path) -> WaveRecord:
    """解析单个 .all/.vall 文件，返回三相波形与头部信息。"""
    path = Path(path)
    raw = path.read_bytes()
    if not raw:
        raise IOError(f"文件为空: {path}")
    if len(raw) > MAX_DATA_LENGTH:
        raise IOError(f"文件过大(>{MAX_DATA_LENGTH} bytes): {path}")

    buf = memoryview(raw)

    pos: list[int] = []
    limit = min(4096, len(buf))
    for i in range(limit):
        if buf[i] == ord(" "):
            pos.append(i)
            if len(pos) == 16:
                break
    if len(pos) < 16:
        raise ValueError(f"头部格式异常: 前 {limit} 字节内未找到 16 个分隔空格: {path}")

    start = _binary_payload_start(buf, pos[15], path)

    station = _parse_int_field(buf, 0, pos[0])
    line = _parse_int_field(buf, pos[0], pos[1])
    year = _parse_int_field(buf, pos[1], pos[2])
    month = _parse_int_field(buf, pos[2], pos[3])
    day = _parse_int_field(buf, pos[3], pos[4])
    hour = _parse_int_field(buf, pos[4], pos[5])
    minute = _parse_int_field(buf, pos[5], pos[6])
    second = _parse_int_field(buf, pos[6], pos[7])
    micro_second = _parse_string_field(buf, pos[7], pos[8])
    gps_frequency = _parse_string_field(buf, pos[8], pos[9])
    gps_flag = _parse_int_field(buf, pos[9], pos[10])
    break_flag = _parse_int_field(buf, pos[10], pos[11])
    startup_type = _parse_int_field(buf, pos[11], pos[12])

    raw_data = buf[start:]
    nbytes = len(raw_data)
    rem = nbytes % 6
    if rem != 0:
        raise ValueError(
            f"数据区长度 {nbytes} 无法被 6 整除(余 {rem})，多为头部偏移错误: {path}"
        )
    data_length = nbytes // 6
    if data_length <= 0:
        raise ValueError(f"数据点数为 0: {path}")

    valid_bytes = raw_data[: data_length * 6]

    if data_length < PACKED_12BIT_MAX_POINTS:
        raw_np = np.frombuffer(valid_bytes, dtype=np.uint8)
        a0 = raw_np[0::6].astype(np.int16)
        a1 = raw_np[1::6].astype(np.int16)
        b0 = raw_np[2::6].astype(np.int16)
        b1 = raw_np[3::6].astype(np.int16)
        c0 = raw_np[4::6].astype(np.int16)
        c1 = raw_np[5::6].astype(np.int16)
        # 12bit 有符号：高低字节拼接后减去偏置 0x800
        data_a = ((a1 << 4) | a0) - 0x800
        data_b = ((b1 << 4) | b0) - 0x800
        data_c = ((c1 << 4) | c0) - 0x800
    else:
        reshaped = np.frombuffer(valid_bytes, dtype="<i2").reshape(-1, 3)
        data_a = reshaped[:, 0]
        data_b = reshaped[:, 1]
        data_c = reshaped[:, 2]

    signals = np.stack(
        [
            np.asarray(data_a, dtype=np.float32),
            np.asarray(data_b, dtype=np.float32),
            np.asarray(data_c, dtype=np.float32),
        ]
    )

    # 采样率来自 GPS 频率字段（单位 kHz），例如 "624.98581" -> 624985.81 Hz
    sampling_rate_hz = 0.0
    sampling_rate_valid = False
    try:
        value = float(gps_frequency)
        if value > 0:
            sampling_rate_hz = value * 1000.0
            sampling_rate_valid = True
    except ValueError:
        pass
    if not sampling_rate_valid:
        sampling_rate_hz = 1_250_000.0  # 无 GPS 频率时的名义采样率兜底

    timestamp_text = (
        f"{year:04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02d}"
        f".{micro_second}" if year else ""
    )

    return WaveRecord(
        file_path=str(path),
        file_name=path.name,
        station=station,
        line=line,
        timestamp_text=timestamp_text,
        gps_frequency=gps_frequency,
        gps_flag=gps_flag,
        break_flag=break_flag,
        startup_type=startup_type,
        data_length=data_length,
        sampling_rate_hz=sampling_rate_hz,
        sampling_rate_valid=sampling_rate_valid,
        signals=signals,
    )
