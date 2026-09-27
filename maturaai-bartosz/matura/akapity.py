"""Dane dla eksperta akapitu eseju (E3): tematy w stylu CKE → akapity pisane przez nauczyciela z materiału z kart.

Co to jest: generator par (prompt akapitu, akapit wzorcowy) w DOKŁADNIE tym formacie, który harness wysyła na egzaminie
(`esej.prompt_akapitu`, materiał z `esej._material` na kartach kanonicznych, tryb e3).
Po co: mały model uczy się jednej wąskiej rzeczy: z tezy, aspektu i 8 faktów napisać akapit 5–6 zdań z konkretami.
Strukturę całego eseju (wstęp, stanowisko, zakończenie, długość) trzyma kod w `esej.napisz`.
Co zrobić (nauczyciel tylko w fazie budowy):
  uv run python -m matura.akapity --tematy   → data/esej/tematy.jsonl   (5 tematów na dział, 2 rodzaje jak w CKE)
  uv run python -m matura.akapity --akapity  → data/esej/akapity.jsonl  (temat × 3 aspekty/elementy)
  uv run python -m matura.akapity --train    → data/sft/train-akapit.jsonl (prompt/completion dla trening.trenuj)
Tematy podobne do ewaluacyjnych (≥4 wspólne 8-gramy z eseje-cke / eseje-kalibracja) są odrzucane.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import esej

ROOT = Path(__file__).resolve().parent.parent
KAT = ROOT / "data" / "esej"
TEMATY, AKAPITY = KAT / "tematy.jsonl", KAT / "akapity.jsonl"
TRAIN = ROOT / "data" / "sft" / "train-akapit.jsonl"
_lock = threading.Lock()
WAGI_STANOWISK = {"czesciowo": 0.5, "tak": 0.25, "nie": 0.25}

SCHEMA_TEMATY = {"type": "object", "additionalProperties": False, "required": ["tematy"], "properties": {"tematy": {
    "type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["temat", "elementy"],
                               "properties": {"temat": {"type": "string"},
                                              "elementy": {"type": "array", "items": {"type": "string"}}}}}}}
INSTR_TEMATY = """Jesteś autorem arkuszy CKE: matura z historii, poziom rozszerzony, formuła 2023. Dział: {dzial}.
Ułóż 5 tematów wypowiedzi argumentacyjnej, dokładnie w formie CKE: teza w jednym zdaniu, potem „Zajmij stanowisko wobec
powyższej tezy i je uzasadnij, uwzględniając w swojej argumentacji …”.
- 3 tematy z aspektami: zakończ „… uwzględniając w swojej argumentacji aspekty: X, Y i Z.” (np. polityczny, społeczny
  i gospodarczy; militarny, kulturowy i religijny); pole elementy = [] (puste).
- 2 tematy z wyborem: zakończ „… uwzględniając w swojej argumentacji trzy wybrane <rzeczy> z tego okresu.” (np. „trzech
  wybranych władców”, „trzy wybrane wydarzenia”); pole elementy = 3 najlepsze przykłady (nazwy/imiona).
Tezy dyskusyjne (można się zgodzić częściowo), zgodne z zakresem działu. Nie uruchamiaj poleceń. Tylko JSON."""

SCHEMA_AKAPITY = {"type": "object", "additionalProperties": False, "required": ["akapity"], "properties": {"akapity": {
    "type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["id", "akapit"],
                               "properties": {"id": {"type": "string"}, "akapit": {"type": "string"}}}}}}
INSTR_AKAPITY = """Jesteś bardzo dobrym maturzystą. Dla każdego elementu listy napisz akapit wypracowania maturalnego z
historii zgodnie z jego polem „instrukcja” (to reguły dla piszącego) i „zadanie” (teza, aspekt, materiał).
Wymagania: 5–6 zdań, po polsku; tylko wskazany aspekt; w każdym zdaniu konkret (data, postać, wydarzenie, pojęcie)
powiązany z tezą; opieraj się głównie na faktach z materiału, dodaj najwyżej jeden powszechnie znany fakt, którego
jesteś pewien; żadnych dat, których nie jesteś pewien; bez wstępu i zakończenia całej pracy; stanowisko jak w instrukcji.
Nie zaczynaj od „Po pierwsze” (to dopisuje szablon). Zwróć JSON: dla każdego id jego akapit. Nie uruchamiaj poleceń.

LISTA:
"""


def _jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for l in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(l))
        except json.JSONDecodeError:
            continue
    return out


def _dopisz(p: Path, rekordy: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with _lock, open(p, "a", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rekordy)


def rozbior(temat: str, elementy: list[str]) -> tuple[str, list[str], bool] | None:
    """Temat → (teza, 3 aspekty albo 3 elementy, czy_wybor); None, gdy `esej.rozbierz` nie da 3 pozycji."""
    teza, aspekty, wybor = esej.rozbierz(temat)
    if wybor:
        el = [e.strip() for e in elementy if e.strip()][:3]
        return (teza, el, True) if len(el) == 3 else None
    if "aspekt" not in temat:  # rozbierz podstawia wtedy aspekty domyślne; temat spoza obu form CKE odrzucamy
        return None
    return (teza, aspekty, False) if len(aspekty) == 3 else None


def _zakazane_tematy() -> set[tuple]:
    from . import devset, trening
    return trening._zakazane((z.get("temat") or z["polecenie"]) for z in devset.wczytaj(["eseje-cke", "eseje-kalibracja"]))


def tematy(nauczyciel: str, rown: int) -> None:
    from .karty import wymagania
    from .sedzia import llm_json
    from .trening import MIN_WSPOLNYCH, _ngramy
    gotowe = {r["dzial"] for r in _jsonl(TEMATY)}
    zak = _zakazane_tematy()
    dzialy = [d for d in wymagania() if d not in gotowe]
    print(f"tematy: działów do zrobienia {len(dzialy)}", flush=True)

    def jeden(d):
        try:
            w = llm_json(INSTR_TEMATY.format(dzial=d), SCHEMA_TEMATY, nauczyciel)
        except Exception as e:
            print(f"{d[:50]}: błąd {str(e)[:150]}", flush=True); return
        ok = []
        for t in w["tematy"]:
            r = rozbior(t["temat"], t["elementy"])
            if r and len(_ngramy(t["temat"]) & zak) < MIN_WSPOLNYCH:
                ok.append({"dzial": d, "temat": t["temat"], "teza": r[0], "pozycje": r[1], "wybor": r[2]})
        _dopisz(TEMATY, ok)
        print(f"{d[:60]}: tematów {len(ok)}/{len(w['tematy'])}", flush=True)

    with ThreadPoolExecutor(rown) as ex:
        list(ex.map(jeden, dzialy))


def elementy_akapitow(seed: int = 5) -> list[dict]:
    """Temat × pozycja → element do napisania: {id, system, user, stanowisko} w formacie egzaminu (tryb e3)."""
    from .karty import Karty
    karty, rng, out = Karty(), random.Random(seed), []
    for i, t in enumerate(_jsonl(TEMATY)):
        st = rng.choices(list(WAGI_STANOWISK), weights=list(WAGI_STANOWISK.values()))[0]
        kon_st = esej.STANOWISKA[st][1]
        for j, a in enumerate(t["pozycje"]):
            mat = esej._material(None, karty, t["teza"], a, "e3")
            sys_, user = esej.prompt_akapitu(t["teza"], a, mat, kon_st, "e3")
            out.append({"id": f"t{i}-a{j}", "system": sys_, "user": user, "stanowisko": st, "dzial": t["dzial"]})
    return out


def akapity(nauczyciel: str, rown: int, na_partie: int = 6) -> None:
    from .sedzia import llm_json
    gotowe = {r["id"] for r in _jsonl(AKAPITY)}
    el = [e for e in elementy_akapitow() if e["id"] not in gotowe]
    partie = [el[i:i + na_partie] for i in range(0, len(el), na_partie)]
    print(f"akapity: do zrobienia {len(el)} w {len(partie)} partiach", flush=True)

    def jedna(k, pt):
        lista = json.dumps([{"id": e["id"], "instrukcja": e["system"], "zadanie": e["user"]} for e in pt], ensure_ascii=False, indent=1)
        try:
            w = llm_json(INSTR_AKAPITY + lista, SCHEMA_AKAPITY, nauczyciel)
        except Exception as e:
            print(f"partia {k}: błąd {str(e)[:150]}", flush=True); return
        po_id = {e["id"]: e for e in pt}
        ok = [dict(po_id[a["id"]], akapit=a["akapit"].strip()) for a in w["akapity"]
              if a["id"] in po_id and len(a["akapit"].split()) >= 40]
        _dopisz(AKAPITY, ok)
        if k % 10 == 0:
            print(f"partia {k}: +{len(ok)}", flush=True)

    with ThreadPoolExecutor(rown) as ex:
        list(ex.map(lambda kp: jedna(*kp), enumerate(partie)))


def train() -> Path:
    rek = _jsonl(AKAPITY)
    TRAIN.write_text("".join(json.dumps({"prompt": [{"role": "system", "content": r["system"]}, {"role": "user", "content": r["user"]}],
                                          "completion": [{"role": "assistant", "content": r["akapit"]}]}, ensure_ascii=False) + "\n"
                             for r in rek), encoding="utf-8")
    print(f"dane akapitów: {len(rek)} → {TRAIN}")
    return TRAIN


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tematy", action="store_true"); ap.add_argument("--akapity", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--nauczyciel", default="astra:gpt-6-astra"); ap.add_argument("--rownolegle", type=int, default=6)
    a = ap.parse_args()
    if a.tematy:
        tematy(a.nauczyciel, a.rownolegle)
    if a.akapity:
        akapity(a.nauczyciel, a.rownolegle)
    if a.train:
        train()
    sys.exit(0)
