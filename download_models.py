"""Download model weights into ./models/<name> so they are stored locally in the project.

Usage:
    python download_models.py smolvlm2-500m lfm2.5-vl-450m
"""
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

MODELS = {
    "smolvlm2-500m": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    "lfm2.5-vl-450m": "LiquidAI/LFM2.5-VL-450M",
}

if __name__ == "__main__":
    for name in sys.argv[1:] or MODELS:
        target = Path("models") / name
        print(f"Downloading {MODELS[name]} -> {target}")
        snapshot_download(MODELS[name], local_dir=target, ignore_patterns=["onnx/*", "*.onnx*"])
