"""Pobranie modeli zestawów kandydackich MaturaAI do data/modele/ i data/hf/ (lista i licencje w SOURCE.md).

Co to jest: hf_hub_download / snapshot_download dokładnych plików i commitów z SOURCE.md, ze sprawdzeniem rozmiaru.
Po co: odtworzenie zestawu na czystej maszynie (wagi nie są w repo).
Co zrobić: uv run python scripts/pobierz_modele.py [--tylko baza,opisywacz,...]; wznawialne (pomija pobrane pliki).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parent.parent
# rola → (repo, commit, [pliki], katalog docelowy, {plik: bajty}); None w plikach = cały katalog (MarianMT)
MODELE = {
    "baza": ("unsloth/Qwen3.5-4B-GGUF", "e87f176479", ["Qwen3.5-4B-UD-IQ3_XXS.gguf"], "data/modele/qwen3.5-4b-iq3xxs",
             {"Qwen3.5-4B-UD-IQ3_XXS.gguf": 1949047968}),
    "baza_iq4xs": ("unsloth/Qwen3.5-4B-GGUF", "e87f176479", ["Qwen3.5-4B-IQ4_XS.gguf"], "data/modele/qwen3.5-4b-iq4xs",
                   {"Qwen3.5-4B-IQ4_XS.gguf": 2477053088}),
    "opisywacz": ("unsloth/Qwen3.5-2B-GGUF", "f6d5376be1", ["Qwen3.5-2B-Q4_K_M.gguf", "mmproj-F16.gguf"],
                  "data/modele/qwen3.5-2b-q4", {"Qwen3.5-2B-Q4_K_M.gguf": 1280835840, "mmproj-F16.gguf": 668227264}),
    "baza_en": ("unsloth/Qwen3-4B-Instruct-2507-GGUF", "a06e946bb6", ["Qwen3-4B-Instruct-2507-UD-IQ3_XXS.gguf"],
                "data/modele/qwen3-4b-2507-iq3xxs", {"Qwen3-4B-Instruct-2507-UD-IQ3_XXS.gguf": 1674376800}),
    "esej_lfm2": ("LiquidAI/LFM2-2.6B-GGUF", "a759abdc59", ["LFM2-2.6B-Q4_K_M.gguf"], "data/modele/lfm2-2.6b-q4km",
                  {"LFM2-2.6B-Q4_K_M.gguf": 1563668704}),
    "marian_pl_en": ("Helsinki-NLP/opus-mt-pl-en", "7f2bb874fd", None, "data/hf/opus-mt-pl-en", {}),
    "marian_en_pl": ("Helsinki-NLP/opus-mt-en-zlw", "92ffd5aa93", None, "data/hf/opus-mt-en-zlw", {}),
}
BEZ = ["*.h5", "*.ot", "*.msgpack", "tf_model*", "flax*", "rust*", "onnx/*"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tylko", help="role po przecinku (domyślnie wszystkie): " + ",".join(MODELE))
    a = ap.parse_args(argv)
    role = a.tylko.split(",") if a.tylko else list(MODELE)
    bledy = 0
    for r in role:
        repo, commit, pliki, kat, rozm = MODELE[r]
        cel = ROOT / kat
        if pliki is None:
            snapshot_download(repo, revision=commit, local_dir=cel, ignore_patterns=BEZ)
        for f in pliki or []:
            p = cel / f
            if not (p.exists() and p.stat().st_size == rozm.get(f, p.stat().st_size if p.exists() else -1)):
                hf_hub_download(repo, f, revision=commit, local_dir=cel)
            if rozm.get(f) and p.stat().st_size != rozm[f]:
                print(f"BŁĄD rozmiaru {p}: {p.stat().st_size} zamiast {rozm[f]}", file=sys.stderr)
                bledy += 1
        print(f"{r}: {repo}@{commit} → {kat}")
    return 1 if bledy else 0


if __name__ == "__main__":
    sys.exit(main())
