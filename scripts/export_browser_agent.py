#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _path  # noqa: F401
from kalah_zero.network import load_checkpoint


def tensor_payload(tensor) -> dict:
    values = tensor.detach().cpu().flatten().tolist()
    return {
        "shape": list(tensor.shape),
        "values": [float(value) for value in values],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a KalahZero checkpoint for browser inference.")
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    args = parser.parse_args()

    model, checkpoint = load_checkpoint(args.checkpoint)
    model.eval()
    payload = {
        "format": "kalah-zero-browser-agent-v1",
        "source": Path(args.checkpoint).name,
        "config": {
            "pits": int(getattr(model, "pits", checkpoint.get("pits", 6))),
            "model_type": getattr(model, "model_type", checkpoint.get("model_type", "mlp")),
            "hidden_size": int(getattr(model, "hidden_size", checkpoint.get("hidden_size", 128))),
            "residual_blocks": int(
                getattr(model, "residual_blocks", checkpoint.get("residual_blocks", 0))
            ),
        },
        "state_dict": {
            name: tensor_payload(tensor)
            for name, tensor in model.state_dict().items()
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {output} ({output.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
