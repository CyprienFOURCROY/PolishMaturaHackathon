"""Wikipedia PL → fragmenty tekstu do wyszukiwania (BM25).

Cel: z dumpu plwiki (data/wiki/plwiki-latest-pages-articles.xml.bz2) zrobić plik
data/wiki/fragmenty.jsonl: {"id", "tytul", "tekst"} po ok. 120 słów.
Zakres (żeby indeks zmieścił się w RAM): z KAŻDEGO artykułu wstęp (do 2 fragmentów);
z artykułów „historycznych" (kategorie z listą słów kluczowych) do 40 fragmentów treści.
Uruchomienie: `uv run python -m matura.wiki [--limit N]`. Licencja danych: CC BY-SA (Wikipedia).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from multiprocessing import Pool
from pathlib import Path

KAT = Path(__file__).resolve().parent.parent / "data" / "wiki"
DUMP = KAT / "plwiki-latest-pages-articles.xml.bz2"
WYJ = KAT / "fragmenty.jsonl"
NS = "{http://www.mediawiki.org/xml/export-0.11/}"

SLOWA_HIST = (
    "histor", "bitw", "wojn", "władc", "król", "królow", "książ", "cesarz", "papież", "papiest",
    "dynasti", "powstan", "trakta", "sejm", "konfederac", "starożytn", "średniowiecz", "rewolucj",
    "zabor", "okupac", "konspirac", "ruch opor", "zakon", "reformacj", "renesans", "barok",
    "oświecen", "romantyzm", "pozytywizm", "rzeczpospolit", "prl", "polityc", "partie", "ustr",
    "dyplomac", "armi", "wojsk", "szlacht", "chłop", "unie", "unia ", "zjazd", "kongres",
    "imperi", "państw", "kolonial", "zimna wojna", "holokaust", "zagład", "komunizm", "faszyzm",
    "nazizm", "antyk", "rzym", "grecj", "egipt", "bizancj", "krzyżow", "piastow", "jagiellon",
    "wazow", "sas", "poniatowsk", "napoleon", "legion", "solidarno", "architektur", "sztuk",
    "filozof", "religi", "kości", "prawosław", "islam", "judaizm",
)

RE_KAT = re.compile(r"\[\[Kategoria:([^\]|]+)", re.I)
RE_REF = re.compile(r"<ref[^>/]*/>|<ref[^>]*>.*?</ref>", re.S | re.I)
RE_TAG = re.compile(r"<[^>]+>")
RE_KOM = re.compile(r"<!--.*?-->", re.S)
RE_TAB = re.compile(r"\{\|.*?\|\}", re.S)
RE_PLIK = re.compile(r"\[\[(?:Plik|File|Grafika|Image|Kategoria|Category):[^\[\]]*(?:\[\[[^\]]*\]\][^\[\]]*)*\]\]", re.I)
RE_LINK = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]")
RE_EXT = re.compile(r"\[https?://[^\s\]]+\s*([^\]]*)\]")
RE_NAGL = re.compile(r"^=+\s*(.*?)\s*=+\s*$", re.M)
RE_FORM = re.compile(r"'{2,}")
SEKCJE_STOP = ("Przypisy", "Bibliografia", "Linki zewnętrzne", "Zobacz też", "Uwagi", "Literatura")


def usun_szablony(t: str) -> str:
    """Usuwa {{...}} z obsługą zagnieżdżeń (liniowo)."""
    out, gl, i, n = [], 0, 0, len(t)
    while i < n:
        if t.startswith("{{", i):
            gl += 1; i += 2; continue
        if gl and t.startswith("}}", i):
            gl -= 1; i += 2; continue
        if not gl:
            out.append(t[i])
        i += 1
    return "".join(out)


def czysc(wt: str) -> list[tuple[str, str]]:
    """Wikitekst → lista (sekcja, akapit) czystego tekstu."""
    t = RE_KOM.sub("", wt)
    t = RE_REF.sub("", t)
    t = usun_szablony(t)
    t = RE_TAB.sub("", t)
    t = RE_PLIK.sub("", t)
    t = RE_LINK.sub(r"\1", t)
    t = RE_EXT.sub(r"\1", t)
    t = RE_TAG.sub("", t)
    t = RE_FORM.sub("", t)
    wynik, sekcja = [], "Wstęp"
    for blok in re.split(r"\n\s*\n", t):
        blok = blok.strip()
        if not blok:
            continue
        m = RE_NAGL.match(blok.split("\n")[0])
        if m:
            sekcja = m.group(1)
            blok = "\n".join(blok.split("\n")[1:]).strip()
            if not blok:
                continue
        if sekcja in SEKCJE_STOP:
            continue
        linie = [l.lstrip("*#:; ").strip() for l in blok.split("\n")]
        blok = " ".join(l for l in linie if l)
        if len(blok) > 40:
            wynik.append((sekcja, blok))
    return wynik


def tnij(akapity: list[str], slow: int = 120) -> list[str]:
    frag, biez = [], []
    for a in akapity:
        biez.extend(a.split())
        while len(biez) >= slow:
            frag.append(" ".join(biez[:slow])); biez = biez[slow - 20:]  # zakładka 20 słów
    if len(biez) > 25:
        frag.append(" ".join(biez))
    return frag


def przetworz(arg: tuple[str, str]) -> list[dict]:
    tytul, wt = arg
    kategorie = [k.lower() for k in RE_KAT.findall(wt)]
    hist = any(s in k for k in kategorie for s in SLOWA_HIST)
    ak = czysc(wt)
    if not ak:
        return []
    wstep = [a for s, a in ak if s == "Wstęp"]
    frag = tnij(wstep)[:2]
    if hist:
        frag += tnij([a for s, a in ak if s != "Wstęp"])[:40]
    return [{"tytul": tytul, "hist": hist, "tekst": f} for f in frag]


def strony(limit: int | None):
    proc = subprocess.Popen(["bzip2", "-dc", str(DUMP)], stdout=subprocess.PIPE, bufsize=1 << 20)
    n = 0
    for _, el in ET.iterparse(proc.stdout, events=("end",)):
        if el.tag != NS + "page":
            continue
        ns = el.findtext(NS + "ns")
        if ns == "0" and el.find(NS + "redirect") is None:
            tekst = el.findtext(f"{NS}revision/{NS}text") or ""
            tytul = el.findtext(NS + "title") or ""
            if not tytul.startswith(("Lista ", "Ujednoznacznienie")) and "(ujednoznacznienie)" not in tytul:
                yield tytul, tekst
                n += 1
                if limit and n >= limit:
                    break
        el.clear()
    proc.kill()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--wyj", default=str(WYJ))
    a = ap.parse_args()
    t0, n_art, n_frag, idf = time.time(), 0, 0, 0
    tmp = Path(a.wyj + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f, Pool(14) as pool:
        for frs in pool.imap(przetworz, strony(a.limit), chunksize=64):
            n_art += 1
            for fr in frs:
                fr["id"] = idf; idf += 1
                f.write(json.dumps(fr, ensure_ascii=False) + "\n")
            n_frag += len(frs)
            if n_art % 50000 == 0:
                print(f"{n_art} art., {n_frag} frag., {time.time()-t0:.0f}s", flush=True)
    tmp.rename(a.wyj)
    print(f"GOTOWE: {n_art} artykułów, {n_frag} fragmentów, {time.time()-t0:.0f}s → {a.wyj}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
