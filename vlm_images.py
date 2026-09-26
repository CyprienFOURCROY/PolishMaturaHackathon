"""Pipeline: image -> small VLM -> description.

Writes description_<model>_vlm.json: {image_name: description}.

Usage:
    python vlm_images.py --model smolvlm2-500m
    python vlm_images.py --model lfm2.5-vl-450m
"""
import argparse
import json
import re
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
PROMPT = (
    "This image comes from a Polish history exam (Matura). Describe it in detail: "
    "what kind of source it is (map, photo, table, diagram, text, caricature...), "
    "everything visible, and any text you can read."
)


def natural_key(p: Path):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.name)]


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.inference_mode()
def describe(model, processor, image: Image.Image, device, max_new_tokens: int) -> str:
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": PROMPT}]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    ).to(device)
    inputs = {k: v.to(model.dtype) if v.is_floating_point() else v for k, v in inputs.items()}
    ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.15)
    return processor.batch_decode(ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="folder name under models/, e.g. smolvlm2-500m")
    ap.add_argument("--images", default="Images from Matura")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    paths = sorted((p for p in Path(args.images).iterdir() if p.suffix.lower() in IMAGE_EXTS), key=natural_key)
    device = pick_device()
    model_dir = Path("models") / args.model
    print(f"Loading {model_dir} on {device}...")
    processor = AutoProcessor.from_pretrained(model_dir)
    model = AutoModelForImageTextToText.from_pretrained(model_dir, torch_dtype=torch.bfloat16).to(device).eval()

    results = {}
    for path in paths:
        results[path.name] = describe(model, processor, Image.open(path).convert("RGB"), device, args.max_new_tokens)
        print(f"\n{path.name}\n{results[path.name]}")

    out = f"description_{args.model.replace('-', '_')}_vlm.json"
    Path(out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved {len(results)} descriptions to {out}")


if __name__ == "__main__":
    main()
