"""Download weights for every role in config.ROLES into models/<role>/<name>/.

Usage (from repo root):  python3 scripts/download_models.py [vlm translator translator_back llm]
EasyOCR downloads its own weights into models/ocr/easyocr on first use.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from huggingface_hub import snapshot_download

from matura.config import MODELS_DIR, ROLES

if __name__ == "__main__":
    for role in sys.argv[1:] or [r for r in ROLES if "hf" in ROLES[r]]:
        spec = ROLES[role]
        target = MODELS_DIR / role / spec["name"]
        print(f"{role}: {spec['hf']} -> {target}")
        snapshot_download(spec["hf"], local_dir=target, ignore_patterns=["onnx/*", "*.onnx*", "*.h5", "*.ot", "*.msgpack", "*.gguf"])
