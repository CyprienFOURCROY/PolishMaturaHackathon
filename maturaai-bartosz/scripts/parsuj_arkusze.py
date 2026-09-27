"""Parsuje arkusze CKE z historii PR (PDF) i zasady oceniania do JSON.

Cel: surowiec dev-setu. Wejście: data/cke/historia-{rok}-{sesja}-{arkusz,odpowiedzi}.pdf.
Wyjście: data/cke/json/historia-{rok}-{sesja}.json — lista zadań:
  {id, rok, sesja, nr_zadania, nr_glowny, pkt_max, polecenie, zrodla_tekst, obrazy[],
   zasady_oceniania, rozwiazanie, esej}
oraz obrazy źródeł w data/cke/obrazy/{rok}-{sesja}/str{NN}_{k}.png.
Uruchomienie: `uv run python scripts/parsuj_arkusze.py [rok-sesja ...]`.
Heurystyki (znane ograniczenia): obraz przypisany do zadań głównych obecnych na tej samej
stronie; wypracowanie dopasowane po pkt_max=15 (CKE bywa niespójne w numeracji).
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pypdf

KAT = Path(__file__).resolve().parent.parent / "data" / "cke"
WY = KAT / "json"
OBR = KAT / "obrazy"

RE_ZAD = re.compile(r"^\s*Zadanie\s+(\d+)(?:\.(\d+))?\.?\s*(?:\((\d+)\s*[–-]\s*(\d+)\))?", re.M)
RE_ROZW = r"(?:Rozwiązanie|Rozwiązania|Przykładowe rozwiązanie|Przykładowe rozwiązania|Przykładowe odpowiedzi|Przykładowa odpowiedź|Poprawna odpowiedź|Poprawne odpowiedzi|Odpowiedź)"
SMIECI = [
    re.compile(r"Więcej arkuszy znajdziesz na stronie: arkusze\.pl"),
    re.compile(r"^\s*Strona \d+ z \d+\s*$", re.M),
    re.compile(r"^\s*MHIP-R0_\d+\s*$", re.M),
    re.compile(r"^\s*Egzamin maturalny z historii – termin \w+ \d{4} r\.\s*$", re.M),
    re.compile(r"^\s*Zasady oceniania rozwiązań zadań\s*$", re.M),
    re.compile(r"\.{5,}|…{3,}"),                       # linie do pisania
    re.compile(r"^\s*\d+(?:\.\d+)?\.\s*\n\s*0\s*[–-]\s*\d+\s*$", re.M),  # tabelka egzaminatora
    re.compile(r"^\s*BRUDNOPIS.*$", re.M),
]


def czysc(t: str) -> str:
    for r in SMIECI:
        t = r.sub("", t)
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def strony_pdf(p: Path, obrazy_do: Path | None = None) -> list[tuple[str, list[str]]]:
    """Zwraca (tekst, [ścieżki obrazów ≥150 px]) per strona; zapisuje obrazy, jeśli obrazy_do."""
    r = pypdf.PdfReader(str(p))
    out, hashe = [], {}
    for i, pg in enumerate(r.pages):
        t = pg.extract_text() or ""
        pliki = []
        if obrazy_do is not None:
            for k, im in enumerate(pg.images):
                try:
                    img = im.image
                    if img.size[0] < 150 or img.size[1] < 150:
                        continue
                    obrazy_do.mkdir(parents=True, exist_ok=True)
                    cel = obrazy_do / f"str{i+1:02d}_{k}.png"
                    hashe.setdefault(hashlib.md5(img.tobytes()).hexdigest(), set()).add(str(cel.relative_to(KAT.parent.parent)))
                    if not cel.exists():
                        img.convert("RGB").save(cel)
                    pliki.append(str(cel.relative_to(KAT.parent.parent)))
                except Exception:
                    continue
        out.append((t, pliki))
    # znaki wodne / logotypy: ten sam obraz na >=3 stronach → usuń
    ozdoby = {f for pl in hashe.values() if len(pl) >= 3 for f in pl}
    for f in ozdoby:
        (KAT.parent.parent / f).unlink(missing_ok=True)
    return [(t, [f for f in pl if f not in ozdoby]) for t, pl in out]


def tnij_na_zadania(strony: list[tuple[str, list[str]]], od_strony: int = 0) -> list[dict]:
    """Tnie połączony tekst po nagłówkach 'Zadanie N.M. (0–k)'; zapamiętuje strony."""
    tekst, mapa = "", []
    for i, (t, _) in enumerate(strony):
        if i < od_strony:
            continue
        mapa.append((len(tekst), i))
        tekst += "\n" + t
    hity = list(RE_ZAD.finditer(tekst))
    zad = []
    for k, m in enumerate(hity):
        start, koniec = m.start(), (hity[k + 1].start() if k + 1 < len(hity) else len(tekst))
        s0 = max(s for off, s in mapa if off <= start)
        strony_z = sorted({s0} | {s for off, s in mapa if start <= off < koniec})
        nr = m.group(1) + (f".{m.group(2)}" if m.group(2) else "")
        zad.append({"nr": nr, "glowny": m.group(1), "pkt_max": int(m.group(4)) if m.group(4) else None,
                    "tekst": czysc(tekst[start:koniec]), "strony": strony_z})
    return zad


def parsuj_odpowiedzi(strony) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for z in tnij_na_zadania(strony):
        if z["nr"] in out:  # ciąg dalszy (np. rubryka wypracowania na kilku stronach)
            out[z["nr"]]["pelny"] += "\n" + z["tekst"]
            continue
        out[z["nr"]] = {"pkt_max": z["pkt_max"], "pelny": z["tekst"]}
    for o in out.values():
        t = o["pelny"]
        m_z = re.search(r"Zasady oceniania\s*\n(.*?)(?=\n\s*" + RE_ROZW + r"\b|\Z)", t, re.S)
        m_r = re.search(r"\n\s*" + RE_ROZW + r"\b[^\n]*\n(.*)", t, re.S)
        o["zasady_oceniania"] = m_z.group(1).strip() if m_z else ""
        o["rozwiazanie"] = m_r.group(1).strip() if m_r else ""
    return out


def parsuj_sesje(rok: int, sesja: str) -> list[dict] | None:
    pa, po = KAT / f"historia-{rok}-{sesja}-arkusz.pdf", KAT / f"historia-{rok}-{sesja}-odpowiedzi.pdf"
    if not (pa.exists() and po.exists()):
        return None
    sa = strony_pdf(pa, OBR / f"{rok}-{sesja}")
    odp = parsuj_odpowiedzi(strony_pdf(po))
    zad = tnij_na_zadania(sa, od_strony=2)
    # obrazy strony → zadania główne obecne na stronie
    glowne_na_stronie: dict[int, set[str]] = {}
    for z in zad:
        if "." in z["nr"]:
            continue  # obrazy są źródłami → należą do bloku źródeł zadania głównego
        for s in z["strony"]:
            glowne_na_stronie.setdefault(s, set()).add(z["glowny"])
    obrazy_glownego: dict[str, list[str]] = {}
    for s, glowne in glowne_na_stronie.items():
        for g in glowne:
            obrazy_glownego.setdefault(g, []).extend(sa[s][1])
    esej_odp = next((o for o in odp.values() if o["pkt_max"] == 15), None)
    wynik, zrodla = [], {}
    for z in zad:
        if "." not in z["nr"]:
            zrodla[z["glowny"]] = z["tekst"]
            if z["pkt_max"] is None:
                continue  # sam nagłówek ze źródłami; podzadania dalej
        esej = z["pkt_max"] == 15
        o = esej_odp if esej else odp.get(z["nr"], {})
        wynik.append({
            "id": f"{rok}-{sesja}-{z['nr']}", "rok": rok, "sesja": sesja,
            "nr_zadania": z["nr"], "nr_glowny": z["glowny"],
            "pkt_max": z["pkt_max"] or (o or {}).get("pkt_max"),
            "polecenie": z["tekst"],
            "zrodla_tekst": zrodla.get(z["glowny"], "") if "." in z["nr"] else "",
            "obrazy": sorted(set(obrazy_glownego.get(z["glowny"], []))),
            "zasady_oceniania": (o or {}).get("pelny" if esej else "zasady_oceniania", ""),
            "rozwiazanie": "" if esej else (o or {}).get("rozwiazanie", ""),
            "esej": esej,
        })
    return wynik


def main(argv: list[str]) -> int:
    WY.mkdir(exist_ok=True)
    sesje = [tuple(a.split("-")) for a in argv] if argv else [(r, s) for r in range(2015, 2027) for s in ("maj", "czerwiec")]
    for rok, sesja in sesje:
        w = parsuj_sesje(int(rok), sesja)
        if w is None:
            continue
        (WY / f"historia-{rok}-{sesja}.json").write_text(json.dumps(w, ensure_ascii=False, indent=1), encoding="utf-8")
        bez = [z["nr_zadania"] for z in w if not z["rozwiazanie"] and not z["esej"]]
        print(f"{rok}-{sesja}: {len(w)} zadań, {sum(z['pkt_max'] or 0 for z in w)} pkt, "
              f"z obrazem {sum(1 for z in w if z['obrazy'])}, esej {sum(z['esej'] for z in w)}, bez rozwiązania {bez}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
