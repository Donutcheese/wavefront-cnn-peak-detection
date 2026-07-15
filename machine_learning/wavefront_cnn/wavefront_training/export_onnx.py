"""Export a trained checkpoint to ONNX and verify runtime parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from scipy.io import savemat

from .engine import load_checkpoint
from .model import WavefrontResUNet


def export_checkpoint(
    checkpoint: str | Path,
    output: str | Path,
    *,
    fixture: torch.Tensor | None = None,
) -> dict:
    checkpoint = Path(checkpoint)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = state["config"]
    length = int(config.get("window_samples", 8192))
    model = WavefrontResUNet(
        base_channels=int(config["base_channels"]),
        bottleneck_dilations=tuple(config["bottleneck_dilations"]),
    )
    load_checkpoint(checkpoint, model, optimizer=None, device=torch.device("cpu"))
    model.eval()
    if fixture is None:
        generator = torch.Generator().manual_seed(20260715)
        fixture = torch.randn(1, 1, length, generator=generator)
    fixture = fixture.detach().cpu().float()
    if fixture.shape != (1, 1, length):
        raise ValueError(f"fixture must have shape [1, 1, {length}], got {tuple(fixture.shape)}")

    torch.onnx.export(
        model,
        fixture,
        output,
        input_names=["waveform"],
        output_names=["wavefront_logits"],
        dynamic_axes={"waveform": {0: "batch"}, "wavefront_logits": {0: "batch"}},
        opset_version=18,
        dynamo=False,
    )
    onnx_model = onnx.load(output)
    onnx.checker.check_model(onnx_model)
    with torch.inference_mode():
        torch_output = model(fixture).numpy()
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    onnx_output = session.run(None, {"waveform": fixture.numpy()})[0]
    max_abs_difference = float(np.max(np.abs(torch_output - onnx_output)))
    fixture_path = output.with_name(output.stem + "_fixture.mat")
    savemat(
        fixture_path,
        {
            "waveform_bcl": fixture.numpy(),
            "pytorch_logits_bcl": torch_output,
            "sampling_rate_hz": float(config.get("sampling_rate_hz", 1_250_000.0)),
        },
    )
    report = {
        "checkpoint": str(checkpoint.resolve()),
        "onnx": str(output.resolve()),
        "fixture_mat": str(fixture_path.resolve()),
        "opset": 18,
        "max_abs_difference": max_abs_difference,
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if max_abs_difference >= 1e-4:
        raise RuntimeError(f"ONNX parity failed: max_abs_difference={max_abs_difference}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="data/derived/wavefront_cnn_run/wavefront_resunet.onnx")
    args = parser.parse_args()
    print(json.dumps(export_checkpoint(args.checkpoint, args.output), indent=2))


if __name__ == "__main__":
    main()

