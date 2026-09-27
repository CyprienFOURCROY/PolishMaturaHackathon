"""Nocny przebieg: modele × konfiguracje harnesu na dev-secie CKE → odpowiedzi → ocena Astrą → raport.

Uruchomienie (maszyna z GPU):  uv run python -m matura.noc --config noc.toml
Test dymny:                    uv run python -m matura.noc --config noc.toml --limit 5 --modele gemma-270m
Tylko ocena / tylko raport:    --tylko-ocena / --tylko-raport   (wznawia z cache, nic nie liczy od nowa)
Wszystko jest wznawialne: odpowiedzi i oceny zapisywane przyrostowo, gotowe pozycje pomijane.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from . import devset, harness, raport, sedzia

ROOT = Path(__file__).resolve().parent.parent
LOG_LOCK = threading.Lock()


def llama_server() -> str:
    """Ścieżka do llama-server: zmienna LLAMA_SERVER, potem nasza binarka CUDA (data/bin/llama-cuda), potem llama-server
    z PATH (np. `brew install llama.cpp` na Macu albo rozpakowane wydanie llama.cpp na Windows/Linux)."""
    import shutil
    if os.environ.get("LLAMA_SERVER"):
        return os.environ["LLAMA_SERVER"]
    nasza = ROOT / "data" / "bin" / "llama-cuda" / "llama-server"
    if nasza.exists():
        return str(nasza)
    return shutil.which("llama-server") or str(nasza)


def log(*a):
    with LOG_LOCK:
        print(time.strftime("%H:%M:%S"), *a, flush=True)


def pobierz_model(m: dict, katalog: Path) -> tuple[Path, Path | None]:
    if m.get("lokalny"):  # model po treningu: ścieżki względem katalogu projektu
        mm = m.get("mmproj_lokalny")
        return ROOT / m["lokalny"], (ROOT / mm if mm else None)
    from huggingface_hub import hf_hub_download
    g = Path(hf_hub_download(m["repo"], m["plik"], local_dir=katalog / m["nazwa"]))
    mm = Path(hf_hub_download(m["repo"], m["mmproj"], local_dir=katalog / m["nazwa"])) if m.get("mmproj") else None
    return g, mm


class Serwer:
    def __init__(self, bin_: str, gguf: Path, mmproj: Path | None, port: int, rownolegle: int, ctx: int, logf: Path,
                 extra: list[str] | None = None):
        cmd = [bin_, "-m", str(gguf), "--port", str(port), "-ngl", "999", "-np", str(rownolegle),
               "-c", str(ctx * rownolegle), "--jinja", "--no-webui"] + (extra or [])
        if mmproj:
            cmd += ["--mmproj", str(mmproj)]
        self.url = f"http://127.0.0.1:{port}"
        self.p = subprocess.Popen(cmd, stdout=open(logf, "w"), stderr=subprocess.STDOUT)
        for _ in range(600):
            if self.p.poll() is not None:
                raise RuntimeError(f"llama-server padł, log: {logf}")
            try:
                if requests.get(self.url + "/health", timeout=2).status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(1)
        raise RuntimeError("llama-server nie wstał w 10 min")

    def stop(self):
        self.p.terminate()
        try:
            self.p.wait(30)
        except subprocess.TimeoutExpired:
            self.p.kill()


def plik_odp(wyn: Path, model: str, konfig: str) -> Path:
    return wyn / "odpowiedzi" / f"{model}__{konfig}.jsonl"


def wczytaj_odp(p: Path) -> dict[str, dict]:
    if not p.exists():
        return {}
    return {d["id"]: d for d in (json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip())}


KONFIGI_Z_WIKI = {"h0", "h0_en", "h1", "h0_ocr", "h0_opis", "h2", "h2_ocr", "h2_vlm", "e0", "e2", "e3", "e4", "e3_en",
                  "e5", "e5_zgadzam", "e6", "e6_zgadzam", "e7", "e7_nie"}   # e0 = h0 na eseju; e2-e4 szukają per aspekt; e6 → zapas e5
KONFIGI_Z_KARTAMI = {"e3", "e4", "e3_en", "e5", "e5_zgadzam", "e6", "e6_zgadzam", "e7", "e7_nie"}


def potrzebna_wiki(modele: list[dict]) -> bool:
    return any(k in KONFIGI_Z_WIKI for m in modele for k in m["konfiguracje"])


def potrzebne_karty(modele: list[dict]) -> bool:
    return any(k in KONFIGI_Z_KARTAMI for m in modele for k in m["konfiguracje"])


def zadania_kalibracji(konfig: str, zadania: list[dict]) -> list[dict]:
    """wzorzec: tylko zadania z rozwiązaniem CKE; kalibracja_cke: tylko wypracowania z Informatora; pusty: wszystkie."""
    if konfig == "wzorzec":
        return [z for z in zadania if z.get("rozwiazanie")]
    if konfig == "kalibracja_cke":
        return [z for z in zadania if z.get("wzorzec_cke")]
    return zadania


def filtruj_modele(modele: list[dict], nazwy: str | None, konfiguracje: str | None) -> list[dict]:
    """Filtry CLI --modele i --konfiguracje (listy po przecinku); model bez pasujących konfiguracji wypada."""
    out = []
    for m in modele:
        if nazwy and m["nazwa"] not in nazwy.split(","):
            continue
        k = [x for x in m["konfiguracje"] if not konfiguracje or x in konfiguracje.split(",")]
        if k:
            out.append({**m, "konfiguracje": k})
    return out


def filtruj_prefiksy(zadania: list[dict], prefiksy: list[str] | None) -> list[dict]:
    """Tylko zadania, których id zaczyna się od jednego z prefiksów (np. tematy esejów z arkuszy dev bez testu 2026)."""
    return [z for z in zadania if z["id"].startswith(tuple(prefiksy))] if prefiksy else zadania


def generuj(zadania, model: str, konfig: str, wyn: Path, *, url, wiki, obrazy, bez_myslenia, rownolegle, karty=None):
    p = plik_odp(wyn, model, konfig); p.parent.mkdir(parents=True, exist_ok=True)
    gotowe = wczytaj_odp(p)
    todo = [z for z in zadania if z["id"] not in gotowe]
    log(f"[{model}/{konfig}] do zrobienia {len(todo)} z {len(zadania)}")
    lock = threading.Lock()

    def jedno(z):
        try:
            t, s, meta = harness.odpowiedz(konfig, z, url=url, wiki=wiki, obrazy=obrazy, bez_myslenia=bez_myslenia,
                                           karty=karty)
        except Exception as e:  # zapisujemy błąd jako odpowiedź — ocena da 0, raport pokaże liczbę błędów
            t, s, meta = f"(BŁĄD: {e})", 0.0, {}
        with lock, open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps({"id": z["id"], "model": model, "konfig": konfig, "odpowiedz": t, "sekundy": round(s, 2),
                                **meta}, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(rownolegle) as ex:
        list(ex.map(jedno, todo))
    return p


class Oceniacz(threading.Thread):
    """Konsument kolejki plików z odpowiedziami → partie do Astry (równolegle), z cache i ponowieniami."""

    def __init__(self, zadania: list[dict], model: str, rownolegle: int, partia: int):
        super().__init__(daemon=True)
        self.q: queue.Queue = queue.Queue(); self.z = {z["id"]: z for z in zadania}
        self.model, self.rown, self.partia = model, rownolegle, partia
        self.bledy = 0

    def dodaj(self, p: Path | None):
        self.q.put(p)

    def run(self):
        with ThreadPoolExecutor(self.rown) as ex:
            while True:
                p = self.q.get()
                if p is None:
                    break
                from .ocen_wszystko import pozycje_do_oceny  # klucz cache z nazwą sędziego (T2)
                poz = pozycje_do_oceny(p, self.z, sedzia.wczytaj_cache(), self.model)
                partie = [[x] for x in poz if x["esej"]]
                krotkie = [x for x in poz if not x["esej"]]
                partie += [krotkie[i:i + self.partia] for i in range(0, len(krotkie), self.partia)]
                log(f"[ocena] {p.name}: {len(poz)} do oceny w {len(partie)} partiach")
                futs = [ex.submit(self._partia, pt) for pt in partie]
                for f in as_completed(futs):
                    f.result()
                log(f"[ocena] {p.name}: gotowe")

    def _partia(self, pt):
        for proba in range(3):
            try:
                return sedzia.ocen_partie(pt, self.model)
            except Exception as e:
                log(f"[ocena] błąd (próba {proba+1}/3): {str(e)[:300]}")
                time.sleep(30 * (proba + 1))
        self.bledy += 1


def wczytaj_config(sciezka: str) -> dict:
    """TOML faz; `dziedzicz = "noc/wspolne.toml"` w [ogolne] dokleja wspólne ustawienia (lokalne wygrywają)."""
    cfg = tomllib.loads((ROOT / sciezka).read_text(encoding="utf-8"))
    baza = cfg["ogolne"].get("dziedzicz")
    if baza:
        cfg["ogolne"] = {**tomllib.loads((ROOT / baza).read_text(encoding="utf-8"))["ogolne"], **cfg["ogolne"]}
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="noc.toml")
    ap.add_argument("--limit", type=int, help="tylko N pierwszych zadań (test dymny)")
    ap.add_argument("--modele", help="lista nazw po przecinku (filtr)")
    ap.add_argument("--konfiguracje", help="lista konfiguracji po przecinku (filtr)")
    ap.add_argument("--tylko-ocena", action="store_true")
    ap.add_argument("--tylko-raport", action="store_true")
    ap.add_argument("--bez-oceny", action="store_true")
    a = ap.parse_args()
    cfg = wczytaj_config(a.config)
    og = cfg["ogolne"]
    wyn = ROOT / og["katalog_wynikow"]; wyn.mkdir(parents=True, exist_ok=True)
    zadania = devset.wczytaj(og["sesje"])
    if og.get("bez_esejow"):
        zadania = [z for z in zadania if not z["esej"]]
    zadania = filtruj_prefiksy(zadania, og.get("prefiksy_id"))
    if a.limit:
        zadania = zadania[:a.limit]
    modele = filtruj_modele(cfg["model"], a.modele, a.konfiguracje)
    log(f"zadania: {len(zadania)} ({sum(z['pkt_max'] for z in zadania)} pkt), modele: {[m['nazwa'] for m in modele]}")

    if a.tylko_raport:
        raport.zapisz(wyn, zadania, cfg); return 0

    oc = Oceniacz(zadania, og["sedzia_model"], og.get("sedzia_rownolegle", 3), og.get("sedzia_partia", 12))
    if not a.bez_oceny:
        oc.start()

    if a.tylko_ocena:
        for p in sorted((wyn / "odpowiedzi").glob("*.jsonl")):
            oc.dodaj(p)
    else:
        for konfig in og.get("kalibracja", ["wzorzec", "pusty"]):
            oc.dodaj(generuj(zadania_kalibracji(konfig, zadania), "_kalibracja", konfig, wyn, url=None, wiki=None,
                             obrazy=False, bez_myslenia=True, rownolegle=1))
        wiki = None
        if potrzebna_wiki(modele):
            from .retrieval import IDX, Wikipedia
            if not (IDX / "offsety.npy").exists():
                log("brak indeksu BM25 → buduję (jednorazowo)")
                from .retrieval import buduj
                buduj()
            wiki = Wikipedia(); log("indeks Wikipedii wczytany")
        karty = None
        if potrzebne_karty(modele):
            from .karty import KAT, Karty
            if not (KAT / "karty.jsonl").exists():
                raise SystemExit("brak data/karty/karty.jsonl: uruchom uv run python -m matura.karty --buduj")
            karty = Karty(); log(f"karty faktów wczytane ({len(karty.k)})")
        for m in modele:
            try:
                gguf, mm = pobierz_model(m, ROOT / og.get("katalog_modeli", "data/modele"))
                rozm = gguf.stat().st_size + (mm.stat().st_size if mm else 0)
                (wyn / "rozmiary.json").write_text(json.dumps({**(json.loads((wyn / "rozmiary.json").read_text()) if (wyn / "rozmiary.json").exists() else {}),
                                                               m["nazwa"]: {"gguf": gguf.stat().st_size, "mmproj": mm.stat().st_size if mm else 0, "razem": rozm}}, indent=1))
                (wyn / "logi").mkdir(exist_ok=True)
                rown = m.get("rownolegle", og["rownolegle"])
                s = Serwer(os.environ.get("LLAMA_SERVER", og["llama_server"]), gguf, mm, og["port"], rown,
                           m.get("ctx", og.get("ctx", 8192)), wyn / "logi" / f"serwer_{m['nazwa']}.log",
                           m.get("serwer_args", ["--image-min-tokens", "1024"] if mm else []))
                log(f"[{m['nazwa']}] serwer gotowy ({rozm/1e6:.0f} MB)")
                try:
                    for k in m["konfiguracje"]:
                        p = generuj(zadania, m["nazwa"], k, wyn, url=s.url, wiki=wiki, obrazy=bool(mm) and m.get("obrazy", True),
                                    bez_myslenia=m.get("bez_myslenia", True), rownolegle=rown, karty=karty)
                        oc.dodaj(p)
                finally:
                    s.stop()
            except Exception as e:
                log(f"[{m['nazwa']}] POMINIĘTY: {e}")
            raport.zapisz(wyn, zadania, cfg)  # raport cząstkowy po każdym modelu
    oc.dodaj(None)
    if not a.bez_oceny:
        oc.join()
    raport.zapisz(wyn, zadania, cfg)
    log(f"KONIEC. Raport: {wyn / 'RAPORT.md'}  (nieudane partie oceny: {oc.bledy})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
