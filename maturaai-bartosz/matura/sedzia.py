"""Sędzia (tylko w fazie budowy, NIGDY w harnessie egzaminacyjnym) i nauczyciel danych.

Backendy (napis „backend:model"):
- "claude:fable"        — `claude -p` (subskrypcja), najlepszy model Claude; DOMYŚLNY sędzia.
- "astra:gpt-6-astra"   — `codex exec` (subskrypcja); tym ocenia organizator → kontrola zgodności.
Podstawa oceny: data/zasady/sedzia-krotkie.md albo sedzia-esej.md (dosłowne zasady CKE, scripts/zasady_oceniania.py)
+ zasady oceniania i przykładowe rozwiązanie danego zadania.
Cache: review/oceny_cache.jsonl, klucz = sha1(sędzia|id|odpowiedź) → wznawialne, różni sędziowie nie mieszają się.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "review" / "oceny_cache.jsonl"
ZASADY = ROOT / "data" / "zasady"
_lock = threading.Lock()

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["oceny"],
          "properties": {"oceny": {"type": "array", "items": {
              "type": "object", "additionalProperties": False,
              "required": ["id", "pkt", "pkt_A", "pkt_B", "bledy_merytoryczne", "uzasadnienie"],
              "properties": {"id": {"type": "string"}, "pkt": {"type": "integer"},
                             "pkt_A": {"type": "integer"}, "pkt_B": {"type": "integer"},
                             "bledy_merytoryczne": {"type": "integer"}, "uzasadnienie": {"type": "string"}}}}}}

INSTR = """Jesteś doświadczonym egzaminatorem CKE z historii (matura, poziom rozszerzony, formuła 2023).
Oceniasz ŚCIŚLE według zasad oceniania CKE podanych w instrukcji systemowej oraz zasad i przykładowego rozwiązania
danego zadania (pola zasady_oceniania, przykladowe_rozwiazanie).
- Akceptuj każdą odpowiedź merytorycznie poprawną i spełniającą warunki zadania, także inną niż przykładowa.
- Nie przyznawaj punktu, gdy brakuje elementu wymaganego w zasadach (np. uzasadnienia, odwołania do źródła).
- Pole pkt: punkty całkowite od 0 do pkt_max.
- Wypowiedź argumentacyjna (esej=true): oceń kryterium A (narracja historyczna, 0–12, z odjęciem punktów za błędy
  merytoryczne wg tabeli) i B (spójność, 0–3; mniej niż 300 słów → B=0); pkt = A + B; podaj liczbę błędów merytorycznych.
  Dla zadań krótkich pkt_A = pkt, pkt_B = 0, bledy_merytoryczne = 0.
- Źródła ikonograficzne nie są pokazane; opieraj się na zasadach i przykładowym rozwiązaniu.
- uzasadnienie: 1–2 zdania po polsku (dla eseju: które aspekty bogato/zadowalająco/powierzchownie, jakie błędy).
Zwróć wyłącznie JSON zgodny ze schematem, jedna ocena na każde id.

ODPOWIEDZI DO OCENY (JSON):
"""


def klucz(zid: str, odp: str, sedzia: str = "") -> str:
    return hashlib.sha1(f"{sedzia}|{zid}|{odp}".encode()).hexdigest()


def wczytaj_cache() -> dict[str, dict]:
    if not CACHE.exists():
        return {}
    out = {}
    for l in CACHE.read_text(encoding="utf-8").splitlines():
        if l.strip():
            d = json.loads(l); out[d["klucz"]] = d
    return out


def _zapisz(wpisy: list[dict]) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with _lock, open(CACHE, "a", encoding="utf-8") as f:
        for w in wpisy:
            f.write(json.dumps(w, ensure_ascii=False) + "\n")


def _claude(prompt: str, schema: dict, model: str, system: str, timeout: int) -> dict:
    cmd = ["claude", "-p", "--model", model, "--output-format", "json", "--json-schema", json.dumps(schema),
           "--no-session-persistence", "--restricted", "--strict-mcp-config", "--disable-slash-commands",
           "--system-prompt", system]
    with tempfile.TemporaryDirectory() as td:  # pusty katalog roboczy: sędzia nie widzi projektu
        r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout, cwd=td)
    if r.returncode != 0:
        raise RuntimeError(f"claude exit {r.returncode}: {(r.stderr or r.stdout)[-800:]}")
    out = json.loads(r.stdout)
    if isinstance(out, dict) and out.get("structured_output") is not None:
        return out["structured_output"]
    wynik = out.get("result", "") if isinstance(out, dict) else ""
    if isinstance(wynik, dict):
        return wynik
    s = wynik[wynik.find("{"): wynik.rfind("}") + 1]
    return json.loads(s)


def _codex(prompt: str, schema: dict, model: str, system: str, timeout: int) -> dict:
    with tempfile.TemporaryDirectory() as td:
        sch, wy = Path(td) / "schema.json", Path(td) / "out.json"
        sch.write_text(json.dumps(schema), encoding="utf-8")
        cmd = ["codex", "exec", "-m", model, "-s", "read-only", "--skip-git-repo-check", "--ephemeral",
               "-C", td, "--output-schema", str(sch), "-o", str(wy), "-"]
        r = subprocess.run(cmd, input=(system + "\n\n" + prompt) if system else prompt,
                           capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0 or not wy.exists():
            raise RuntimeError(f"codex exit {r.returncode}: {r.stderr[-800:]}")
        return json.loads(wy.read_text(encoding="utf-8"))


def llm_json(prompt: str, schema: dict, sedzia: str, system: str = "", timeout: int = 900) -> dict:
    """Jedno wywołanie modelu frontier ze schematem JSON. sedzia = "claude:fable" | "astra:gpt-6-astra"."""
    backend, _, model = sedzia.partition(":")
    if backend == "claude":
        return _claude(prompt, schema, model or "fable", system, timeout)
    if backend in ("astra", "codex"):
        return _codex(prompt, schema, model or "gpt-6-astra", system, timeout)
    raise ValueError(f"nieznany backend sędziego: {sedzia}")


codex_json = lambda prompt, schema, model, timeout=900: llm_json(prompt, schema, f"astra:{model}", "", timeout)  # zgodność wstecz


def system_sedziego(esej: bool) -> str:
    p = ZASADY / ("sedzia-esej.md" if esej else "sedzia-krotkie.md")
    if not p.exists():
        raise FileNotFoundError(f"brak {p} — uruchom scripts/zasady_oceniania.py")
    return p.read_text(encoding="utf-8")


def ocen_partie(pozycje: list[dict], sedzia: str, timeout: int = 900) -> list[dict]:
    """pozycje: {klucz, id, pkt_max, polecenie, zasady, rozwiazanie, odpowiedz, esej}. Partia jednorodna (esej lub nie)."""
    esej = bool(pozycje[0].get("esej"))
    lad = [{"id": p["klucz"][:12], "esej": esej, "pkt_max": p["pkt_max"], "polecenie": p["polecenie"][:2500],
            "zasady_oceniania": p["zasady"][:6000], "przykladowe_rozwiazanie": p["rozwiazanie"][:2000],
            "odpowiedz_zdajacego": p["odpowiedz"][:9000]} for p in pozycje]
    wynik = llm_json(INSTR + json.dumps(lad, ensure_ascii=False, indent=1), SCHEMA, sedzia, system_sedziego(esej), timeout)
    oceny = {o["id"]: o for o in wynik["oceny"]}
    wpisy = []
    for p in pozycje:
        o = oceny.get(p["klucz"][:12])
        if o is None:
            continue
        wpisy.append({"klucz": p["klucz"], "id": p["id"], "pkt": max(0, min(int(o["pkt"]), p["pkt_max"])),
                      "pkt_A": o.get("pkt_A"), "pkt_B": o.get("pkt_B"), "bledy": o.get("bledy_merytoryczne"),
                      "pkt_max": p["pkt_max"], "uzasadnienie": o["uzasadnienie"], "sedzia": sedzia})
    _zapisz(wpisy)
    return wpisy
