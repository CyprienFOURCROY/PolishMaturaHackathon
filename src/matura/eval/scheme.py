"""Parse the official CKE marking scheme (data/rules/MHIP-R0-100-2305-zasady.pdf) into {task_id: {max, rules}}."""
import re
from functools import lru_cache

from pypdf import PdfReader

from ..config import ROOT

PDF = ROOT / "data" / "rules" / "MHIP-R0-100-2305-zasady.pdf"
HEADER = re.compile(r"Zadanie (\d+(?:\.\d+)?)\. \(0[–-](\d+)\)")
NOISE = re.compile(r"^(Egzamin maturalny z historii.*|Strona \d+ z \d+|Zasady oceniania rozwiązań zadań)\s*$", re.M)


@lru_cache(maxsize=1)
def load() -> dict[str, dict]:
    text = "\n".join(p.extract_text() for p in PdfReader(PDF).pages)
    heads = list(HEADER.finditer(text))
    out = {}
    for i, h in enumerate(heads):
        tid = h.group(1)
        if tid in out:  # the essay header is repeated
            continue
        block = text[h.end(): heads[i + 1].start() if i + 1 < len(heads) else len(text)]
        k = block.find("Zasady oceniania")  # drop the curriculum-requirements table
        block = NOISE.sub("", block[k:] if k >= 0 else block)
        out[tid] = {"max": int(h.group(2)), "rules": re.sub(r"\n{2,}", "\n", block).strip()}
    return out
