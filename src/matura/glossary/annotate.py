"""protect -> translate -> restore -> glossary block.

Glossary terms are swapped for placeholders so the translator cannot mangle them; afterwards each
placeholder becomes the original Polish term followed by its English gloss.
"""
from typing import Callable

from ..config import GLOSSARY_MAX_TERMS, GLOSSARY_PATH
from . import match


def _ph(i: int) -> str:
    return f"ZQ{i}QZ"


def translate_with_glossary(text: str, translate: Callable[[str], str]) -> tuple[str, list[dict]]:
    """Returns (english_text, glossary_entries_used)."""
    entries = match.load(GLOSSARY_PATH)
    spans = match.find(text, entries)

    protected, pos, mapping = [], 0, {}
    for i, (s, e, entry) in enumerate(spans):
        protected.append(text[pos:s])
        protected.append(_ph(i))
        mapping[_ph(i)] = (text[s:e], entry)
        pos = e
    protected.append(text[pos:])

    out = translate("".join(protected))
    used, missing = [], []
    for ph, (original, entry) in mapping.items():
        replacement = f"{original} ({entry['en']})"
        if ph in out:
            out = out.replace(ph, replacement)
        else:
            missing.append(replacement)  # translator dropped it: keep it as a trailing note
        used.append(entry)
    if missing:
        out += "\n[Terms: " + "; ".join(missing) + "]"
    return out, used


def glossary_block(entries: list[dict]) -> str:
    seen, lines = set(), []
    for e in entries:
        if e["id"] in seen:
            continue
        seen.add(e["id"])
        lines.append(f"- {e['en']}: {e['definition']}")
        if len(lines) >= GLOSSARY_MAX_TERMS:
            break
    return "Glossary:\n" + "\n".join(lines) if lines else ""
