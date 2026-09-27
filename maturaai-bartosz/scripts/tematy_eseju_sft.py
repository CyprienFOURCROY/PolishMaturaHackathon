"""Tematy do esejów nauczyciela dla uczniów SFT (eksperyment 27.09).

Co to jest: `data/esej/sft/tematy_trening.jsonl`, lista tematów (z polem `stanowisko`), na które zamawiane są
eseje nauczyciela.
Po co: dane treningowe bez wycieku: tematy z `data/esej/tematy.jsonl` (228, napisane przez nauczyciela w fazie budowy,
`matura/akapity.py --tematy`), bez tematów sztucznych ewaluacji (`eseje-synt`) i bez tematów CKE 2023-2026.
Filtr: temat odpada, gdy dzieli ≥ 2 8-gramy słów (bez formułek poleceń obecnych w ≥ 3 tematach) albo ma podobieństwo Jaccarda rdzeni słów treści tezy ≥ 0,2 z którymkolwiek
tematem CKE 2023-2026 lub kalibracji; tematy 2026 porównujemy tylko programowo (nie drukujemy ich), liczymy odrzucone.
Stanowisko: każdy temat raz „zgadzam” (jak na egzaminie), a co trzeci drugi raz „nie zgadzam” (model uczy się obu).
Co zrobić: `uv run python scripts/tematy_eseju_sft.py` (deterministyczne, seed 27).
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from matura import devset  # noqa: E402

ZRODLO = ROOT / "data" / "esej" / "tematy.jsonl"
WYJ = ROOT / "data" / "esej" / "sft" / "tematy_trening.jsonl"
RE_W = re.compile(r"\w+")
STOP = set("i w z na do o że się nie jest był była było byli to a od po za przez dla jak oraz lub czy przede wszystkim "
           "wobec powyższej tezy je uzasadnij uwzględniając swojej argumentacji aspekty zajmij stanowisko polityczny "
           "społeczny gospodarczy kulturowy militarny ideologiczny religijny wieku roku latach xix xx xviii xvii xvi "
           "twoja wypowiedź powinna liczyć minimum 300 wyrazów którym które który była najważniejszym".split())
PROG_JACCARD, PROG_8GRAM = 0.2, 2   # 0,2: odpada też ten sam temat w innym ujęciu (rozbicie dzielnicowe, totalitaryzmy)


def slowa(t: str) -> list[str]:
    return RE_W.findall(t.lower())


def ngramy(t: str, n: int = 8) -> set[tuple]:
    w = slowa(t)
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def tresc(t: str) -> set[str]:
    """Rdzenie słów treści (5 pierwszych liter): odmiana nie ukrywa tego samego tematu („nazizm” / „nazizmem”)."""
    return {w[:5] for w in slowa(t) if w not in STOP and len(w) > 2}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def main() -> None:
    chronione = [(e["id"], e["temat"], e.get("rok", 0)) for e in devset.wczytaj(["eseje-cke", "eseje-kalibracja"])]
    synt = [e["temat"] for e in devset.wczytaj(["eseje-synt"])]
    chronione += [(f"synt{k}", t, 0) for k, t in enumerate(synt)]
    ch_ng = [(i, ngramy(t), tresc(t), rok) for i, t, rok in chronione]
    tematy = [json.loads(l) for l in ZRODLO.read_text(encoding="utf-8").splitlines() if l.strip()]
    # formułki poleceń („Zajmij stanowisko wobec powyższej tezy…”): 8-gram w ≥ 3 tematach to wspólny język, nie wyciek
    ile = Counter(g for _, g, _, _ in ch_ng for g in set(g))
    ile.update(g for t in tematy for g in ngramy(t["temat"]))
    formulki = {g for g, c in ile.items() if c >= 3}
    ch_ng = [(i, g - formulki, c, rok) for i, g, c, rok in ch_ng]
    dobre, odrzucone = [], []
    for k, t in enumerate(tematy, 1):
        ng, tr = ngramy(t["temat"]) - formulki, tresc(t["teza"])
        kolizje = [(i, rok, len(ng & g), round(jaccard(tr, c), 2)) for i, g, c, rok in ch_ng
                   if len(ng & g) >= PROG_8GRAM or jaccard(tr, c) >= PROG_JACCARD]
        if kolizje:
            odrzucone.append((f"t{k:03d}", t["teza"], kolizje))
        else:
            dobre.append({"id": f"t{k:03d}", **t, "zrodlo": "data/esej/tematy.jsonl"})
    los = random.Random(27)
    drugie = set(los.sample(range(len(dobre)), len(dobre) // 3))
    wyj = []
    for k, t in enumerate(dobre):
        wyj.append({**t, "id": f"{t['id']}-z", "stanowisko": "zgadzam"})
        if k in drugie:
            wyj.append({**t, "id": f"{t['id']}-n", "stanowisko": "nie zgadzam"})
    los.shuffle(wyj)   # kolejność zamówień: przy przerwaniu generacji zostaje przekrój działów, nie pierwsze epoki
    WYJ.parent.mkdir(parents=True, exist_ok=True)
    WYJ.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in wyj), encoding="utf-8")
    print(f"tematy: {len(tematy)}, chronione: {len(chronione)} (CKE 2023-2026, kalibracja, synt), "
          f"odrzucone: {len(odrzucone)}, zostaje: {len(dobre)}, zamówień: {len(wyj)} → {WYJ.relative_to(ROOT)}")
    for tid, teza, kol in odrzucone:
        widoczne = [c for c in kol if c[1] and c[1] < 2026]
        ukryte = len(kol) - len(widoczne)
        if ukryte:   # kolizja z tematem bez podglądu: nie drukujemy nawet naszej tezy (mogłaby zdradzić temat 2026)
            print(f"  odrzucony {tid}: (teza ukryta) | kolizji bez podglądu (2026, synt, kalibracja): {ukryte}")
        else:
            print(f"  odrzucony {tid}: {teza[:90]} | kolizje 2023-2025: {widoczne[:3]}")


if __name__ == "__main__":
    main()
