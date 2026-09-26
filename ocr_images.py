"""Pipeline: image -> EasyOCR (Polish + English) -> text.

Writes description_easyocr_pl-en_ocr.json: {image_name: extracted_text}.

Usage:
    python ocr_images.py
    python ocr_images.py --images "Images from Matura"
"""
import argparse
import json
from pathlib import Path

import easyocr

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def natural_key(p: Path):
    import re
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.name)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="Images from Matura")
    ap.add_argument("--out", default="description_easyocr_pl-en_ocr.json")
    ap.add_argument("--model-dir", default="models/easyocr")
    args = ap.parse_args()

    paths = sorted((p for p in Path(args.images).iterdir() if p.suffix.lower() in IMAGE_EXTS), key=natural_key)
    reader = easyocr.Reader(["pl", "en"], model_storage_directory=args.model_dir, gpu=False)

    results = {}
    for path in paths:
        # paragraph=True merges nearby boxes into readable blocks, in reading order
        blocks = reader.readtext(str(path), detail=0, paragraph=True)
        results[path.name] = "\n".join(blocks)
        print(f"\n{path.name}\n{results[path.name]}")

    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved {len(results)} OCR results to {args.out}")


if __name__ == "__main__":
    main()
