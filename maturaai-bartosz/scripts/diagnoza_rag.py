"""Diagnoza RAG i języka (R1 planu docs/plans/2026-09-26-plan-po-1a-1b.md, sekcja „RAG i język”), bez sędziego.

Co to jest: dla zadań krótkich dev (2024-maj, 2025-maj) z faktami idealnymi (data/wiedza/kontekst_wzorcowy.jsonl, Claude,
tylko pomiar) liczy, jaką część faktów pokrywają 2 hasła bazy znalezione w 4 wariantach zapytania, i rozdziela punkty
stracone między `goly_vlm_kb` (35,2%) a `goly_vlm_wiedza` (45,2%) na: brak faktu w bazie / fakt nieznaleziony /
fakt niewykorzystany. Punkty z istniejących ocen (automat klucza + cache sędziego), bez nowych wywołań.
Po co: zdecydować, co poprawiać: bazę (pokrycie), wyszukiwanie (w tym język zapytania) czy użycie wiedzy przez model.
Co zrobić: uv run python scripts/diagnoza_rag.py → review/rag-jezyk-2026-09-26/RAPORT.md i raport.json.
Warianty zapytania: pl_bez_opisu; pl_opis_en (obecny potok: polskie zadanie + angielski opis obrazu); pl_opis_pl (opis
przetłumaczony Marianem na polski); en_en (angielskie zadanie z tłumaczenia Claude + opis EN, wyszukiwanie po
angielskich tekstach haseł). Fakt „pokryty” przez tekst: co najmniej 60% jego tokenów (`retrieval.tokeny`) jest w tekście.
Zasada D25: to diagnoza na dev; nic nie jest tu mierzone na arkuszach nietkniętych.
"""
from __future__ import annotations

import json
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from matura import devset, jezyk, klucz, obrazy, sedzia, wiedza  # noqa: E402
from matura.harness import _baza_hasel  # noqa: E402
from matura.retrieval import tokeny  # noqa: E402

WYN = ROOT / "review" / "rag-jezyk-2026-09-26"
SESJE = ("2024-maj", "2025-maj")
PROG_FAKTU = 0.6
STOP_EN = set("the a an of in on at to and or is was were by for with as from that this it its be are which who "
              "their his her they he she not but also had has have been into than more most".split())
ODP_KB = ROOT / "review/etapB2-2026-09-26/odpowiedzi/qwen3.5-4b-iq3xxs__goly_vlm_kb.jsonl"
ODP_WIEDZA = ROOT / "review/etapB1-2026-09-26/odpowiedzi/qwen3.5-4b-iq3xxs__goly_vlm_wiedza.jsonl"


def tokeny_en(t: str) -> list[str]:
    return [w[:6] for w in re.findall(r"[a-z0-9]+", t.lower()) if w not in STOP_EN and len(w) > 1]


def pokryty(fakt: str, tok: set[str]) -> bool:
    f = set(tokeny(fakt))
    return bool(f) and len(f & tok) / len(f) >= PROG_FAKTU


def pkt(z: dict, odp: str, cache: dict) -> float | None:
    if klucz.obslugiwane(z):
        r = klucz.ocen(z, odp)
        if r["pewne"]:
            return r["pkt"]
    c = cache.get(sedzia.klucz(z["id"], odp, "claude:opus"))
    return None if c is None else c["pkt"]


def main() -> int:
    WYN.mkdir(parents=True, exist_ok=True)
    fakty = {w["id"]: w["fakty"] for w in jezyk.wczytaj_jsonl(ROOT / "data/wiedza/kontekst_wzorcowy.jsonl")}
    baza = _baza_hasel()
    tekst_pl = {h["id"]: wiedza._tekst_hasla(h) for h in baza.h}
    tok_pl = {i: set(tokeny(t)) for i, t in tekst_pl.items()}
    en = jezyk.hasla_en()
    import bm25s
    ids = list(tekst_pl)
    bm_en = bm25s.BM25()
    bm_en.index([tokeny_en(en.get(i, "")) for i in ids], show_progress=False)
    zad_en = {w["id"]: w["en"] for w in jezyk.wczytaj_jsonl(jezyk.DANE / "zadania_en.jsonl")}

    zz = [z for s in SESJE for z in devset._cke(s) if not z.get("esej") and z["id"] in fakty]
    bloki_en = {}
    for z in zz:
        try:
            bloki_en[z["id"]] = obrazy.blok_opisu(z, "vlm_q2b_en")
        except Exception:  # noqa: BLE001 (brak opisu = zadanie bez bloku)
            bloki_en[z["id"]] = ""
    z_opisem = [i for i, b in bloki_en.items() if b]
    from matura.tlumacz import Marian
    bloki_pl = dict(zip(z_opisem, Marian().tlumacz([bloki_en[i] for i in z_opisem], "en-pl"))) if z_opisem else {}

    def szukaj_en(q: str, n: int = 2) -> list[str]:
        tk = tokeny_en(q)
        if not tk:
            return []
        wyn, _ = bm_en.retrieve([tk], k=n, show_progress=False)
        return [ids[int(i)] for i in wyn[0]]

    warianty = {
        "pl_bez_opisu": lambda z: [h["id"] for h in baza.szukaj(wiedza.zapytanie(z, ""), 2)],
        "pl_opis_en": lambda z: [h["id"] for h in baza.szukaj(wiedza.zapytanie(z, bloki_en[z["id"]]), 2)],
        "pl_opis_pl": lambda z: [h["id"] for h in baza.szukaj(wiedza.zapytanie(z, bloki_pl.get(z["id"], "")), 2)],
        "en_en": lambda z: szukaj_en(zad_en.get(z["id"], "") + " " + bloki_en[z["id"]][:500]),
    }
    cache = {json.loads(l)["klucz"]: json.loads(l) for l in open(ROOT / "review/oceny_cache.jsonl") if l.strip()}
    kb = jezyk.wczytaj_odp(ODP_KB)
    wz = jezyk.wczytaj_odp(ODP_WIEDZA)

    from matura.retrieval import Wikipedia
    wiki = Wikipedia()

    def tok_wiki(z: dict, k: int) -> set[str]:
        frag = wiki.szukaj(wiedza.zapytanie(z, ""), k)
        return set(tokeny(" ".join(f["tytul"] + " " + f["tekst"] for f in frag)))

    wiersze = []
    for z in zz:
        fk = fakty[z["id"]]
        w_bazie = [any(pokryty(f, t) for t in tok_pl.values()) for f in fk]
        rek = {"id": z["id"], "pkt_max": z["pkt_max"], "z_opisem": bool(bloki_en[z["id"]]), "fakty": len(fk),
               "w_bazie": sum(w_bazie) / len(fk)}
        for nazwa, fn in warianty.items():
            got = fn(z)
            tok = set().union(*(tok_pl[i] for i in got)) if got else set()
            rek[nazwa] = sum(pokryty(f, tok) for f in fk) / len(fk)
            rek[f"{nazwa}_ids"] = got
        rek["dlugosc"] = len(devset.tresc_dla_modelu(z))
        t_haslo = set().union(*(tok_pl[i] for i in rek["pl_bez_opisu_ids"])) if rek["pl_bez_opisu_ids"] else set()
        for k in (2, 4):
            tw = tok_wiki(z, k)
            rek[f"wiki{k}"] = sum(pokryty(f, tw) for f in fk) / len(fk)
            rek[f"hasla+wiki{k}"] = sum(pokryty(f, tw | t_haslo) for f in fk) / len(fk)
        rek["zgodne_z_potokiem"] = kb.get(z["id"], {}).get("hasla") == rek["pl_opis_en_ids"]
        rek["pkt_kb"] = pkt(z, kb.get(z["id"], {}).get("odpowiedz", ""), cache)
        rek["pkt_wiedza"] = pkt(z, wz.get(z["id"], {}).get("odpowiedz", ""), cache)
        wiersze.append(rek)

    # rozkład straty: zadania, w których fakty idealne dały więcej punktów niż baza haseł
    kat = {"brak w bazie": 0.0, "nieznalezione": 0.0, "niewykorzystane": 0.0}
    n_kat = {k: 0 for k in kat}
    zysk_kb = 0.0
    for r in wiersze:
        if r["pkt_kb"] is None or r["pkt_wiedza"] is None:
            continue
        d = r["pkt_wiedza"] - r["pkt_kb"]
        if d <= 0:
            zysk_kb += -d
            continue
        k = ("brak w bazie" if r["w_bazie"] < 0.5 else "nieznalezione" if r["pl_opis_en"] < 0.5 else "niewykorzystane")
        kat[k] += d
        n_kat[k] += 1

    srednie = {n: st.mean(r[n] for r in wiersze) for n in warianty}
    srednie_opis = {n: st.mean(r[n] for r in wiersze if r["z_opisem"]) for n in warianty}
    raport = {"n_zadan": len(wiersze), "n_z_opisem": sum(r["z_opisem"] for r in wiersze),
              "pokrycie_srednio": srednie, "pokrycie_z_opisem": srednie_opis,
              "fakty_w_bazie_srednio": st.mean(r["w_bazie"] for r in wiersze),
              "zgodnosc_z_potokiem": sum(r["zgodne_z_potokiem"] for r in wiersze),
              "strata_pkt": kat, "strata_zadan": n_kat, "pkt_gdzie_kb_lepsze": zysk_kb, "wiersze": wiersze}
    (WYN / "raport.json").write_text(json.dumps(raport, ensure_ascii=False, indent=1), encoding="utf-8")

    L = ["# Diagnoza RAG i języka (R1), bez sędziego", "",
         "**Co to jest:** pokrycie faktów idealnych przez 2 znalezione hasła w 4 wariantach zapytania oraz rozkład punktów "
         "straconych między bazą haseł a faktami idealnymi (Qwen3.5-4B UD-IQ3_XXS, dev 2024-maj + 2025-maj, zadania krótkie).",
         "**Po co:** wybrać, co poprawiać: pokrycie bazy, wyszukiwanie (język zapytania) czy użycie wiedzy.",
         "**Co zrobić:** przeliczenie `uv run python scripts/diagnoza_rag.py`; szczegóły w `raport.json`.", "",
         f"Zadania: {len(wiersze)} (z ilustracją: {raport['n_z_opisem']}); faktów idealnych w bazie (pokryte przez "
         f"jakiekolwiek hasło): średnio {100 * raport['fakty_w_bazie_srednio']:.0f}% na zadanie; wariant `pl_opis_en` zgodny "
         f"z hasłami zapisanymi w odpowiedziach potoku w {raport['zgodnosc_z_potokiem']}/{len(wiersze)} zadań.", "",
         "| wariant zapytania | pokrycie faktów (wszystkie zadania) | pokrycie (zadania z ilustracją) |", "|---|---|---|"]
    for n in warianty:
        L.append(f"| `{n}` | {100 * srednie[n]:.0f}% | {100 * srednie_opis[n]:.0f}% |")
    L += ["", "**Punkty stracone przez bazę wobec faktów idealnych** (zadania, w których fakty idealne dały więcej):", "",
          "| przyczyna | zadania | punkty |", "|---|---|---|"]
    for k in kat:
        L.append(f"| {k} | {n_kat[k]} | {kat[k]:g} |")
    L += ["", f"Punkty, w których baza haseł dała więcej niż fakty idealne: {zysk_kb:g}.",
          "Reguła kategorii: brak w bazie = poniżej 50% faktów pokrytych przez jakiekolwiek hasło; nieznalezione = fakty "
          "są w bazie, ale 2 hasła potoku (`pl_opis_en`) pokrywają poniżej 50%; niewykorzystane = pokrywają co najmniej 50%."]
    kr = [r for r in wiersze if r["dlugosc"] <= 600]
    dl = [r for r in wiersze if r["dlugosc"] > 600]
    L += ["", "**Wikipedia (fragmenty BM25, zapytanie jak `pl_bez_opisu`):** pokrycie faktów idealnych", "",
          f"| źródło | wszystkie ({len(wiersze)}) | krótkie ≤ 600 znaków ({len(kr)}) | długie ({len(dl)}) |", "|---|---|---|---|"]
    for n in ("pl_bez_opisu", "wiki2", "wiki4", "hasla+wiki2", "hasla+wiki4"):
        f = lambda rs: f"{100 * st.mean(r[n] for r in rs):.0f}%" if rs else "-"  # noqa: E731
        L.append(f"| `{n}` | {f(wiersze)} | {f(kr)} | {f(dl)} |")
    raport["wiki"] = {n: {"wszystkie": st.mean(r[n] for r in wiersze), "krotkie": st.mean(r[n] for r in kr) if kr else None,
                          "dlugie": st.mean(r[n] for r in dl) if dl else None}
                      for n in ("pl_bez_opisu", "wiki2", "wiki4", "hasla+wiki2", "hasla+wiki4")}
    (WYN / "raport.json").write_text(json.dumps(raport, ensure_ascii=False, indent=1), encoding="utf-8")
    (WYN / "RAPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
