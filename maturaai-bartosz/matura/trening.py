"""Trening małych modeli (pełny fine-tuning SFT) + eksport do GGUF w kilku kwantyzacjach.

Kroki (każdy wznawialny, wywoływany z scripts/noc.sh):
  dane      data/sft/zadania.jsonl + CKE 2020–2022 → data/sft/train.jsonl (prompt/completion, format h0)
            z kontekstem BM25 w 70% przykładów (model uczy się korzystać z Wikipedii i radzić sobie bez niej);
            filtr wycieku: odrzuca przykład, gdy dzieli 8-gram słów z zadaniem dev/test/synt.
            --filtr typ=zamkniete|otwarte|rozstrzygnij albo epoka=<jedna z router.EPOKI>
                    → data/sft/train-<routing>-<wartość>.jsonl, np. train-typ-zamkniete.jsonl (dane ekspertów H1/H2).
            Klucz jak w router.klucz_eksperta, czyli ten sam wybór eksperta co na egzaminie. Pliki jednego routingu
            dzielą train.jsonl bez reszty: te same przykłady z tą samą decyzją o kontekście (D19: łącznie te same dane).
  trenuj    --baza Qwen/Qwen3-0.6B --nazwa qwen3-0.6b-sft [--dane data/sft/train.jsonl] → data/trening/<nazwa>/hf
  eksport   --nazwa ... --kwanty Q4_K_M,Q6_K,Q8_0          → data/trening/<nazwa>/gguf/<nazwa>-<Q>.gguf
Eksperci H1/H2: ta sama baza co generalista (D9), nazwa = <nazwa generalisty>-<filtr z „-" zamiast „=">, np.
  dane --filtr typ=zamkniete
  trenuj --baza Qwen/Qwen3.5-0.8B --nazwa qwen3.5-0.8b-sft-typ-zamkniete --dane data/sft/train-typ-zamkniete.jsonl
  eksport --nazwa qwen3.5-0.8b-sft-typ-zamkniete --kwanty Q8_0
            → data/trening/qwen3.5-0.8b-sft-typ-zamkniete/gguf/qwen3.5-0.8b-sft-typ-zamkniete-Q8_0.gguf
Zasada regulaminu (D8): w kategorii „Mały" liczy się rozmiar bazy PRZED fine-tuningiem; eksport w kilku
kwantyzacjach mierzy, ile wyuczonego przetrwa kwantyzację.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SFT = ROOT / "data" / "sft"
TREN = ROOT / "data" / "trening"
LLAMA_CPP = ROOT / "data" / "llama.cpp"          # źródła (convert_hf_to_gguf.py)
RE_W = re.compile(r"\w+")
TYPY = ("zamkniete", "otwarte", "rozstrzygnij")  # router.klucz_eksperta(z, "typ") bez „esej" (bez treningu esejowego, D18)
MIN_WSPOLNYCH = 4  # wyciek = co najmniej tyle wspólnych 8-gramów (skopiowane zdanie źródła daje ich wiele, styk formułek 1–2)
WIKI_BM25 = "bm25"  # domyślne `wiki` w dane(): indeks data/wiki/bm25 (retrieval.Wikipedia), jak dotąd


def _ngramy(t: str, n: int = 8) -> set[tuple]:
    w = RE_W.findall(t.lower())
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def _zakazane(teksty, prog: int = 3, treningowe=()) -> set[tuple]:
    """8-gramy zadań ewaluacyjnych bez formułek poleceń: n-gram obecny w >= `prog` różnych zadaniach (ewaluacyjnych
    albo treningowych) to wspólny język arkuszy („Zaznacz P, jeśli informacja jest prawdziwa…”), a nie wyciek treści.
    Wyciek = konkretne zdanie/źródło, które nie powtarza się w wielu zadaniach."""
    ile, ile_tr = Counter(), Counter()
    for t in teksty:
        ile.update(_ngramy(t))
    for t in treningowe:
        ile_tr.update(_ngramy(t))
    return {g for g, c in ile.items() if c < prog and ile_tr[g] < prog}


def _oczysc_cke(r: str) -> str:
    """Rozwiązanie z klucza CKE → odpowiedź ucznia (bez „Przykładowe…", pierwsza z alternatyw)."""
    r = re.sub(r"Przykładow[ea] (uzasadnieni[ea]|odpowied[źz]i?|rozwiązani[ea])\s*:?", "Uzasadnienie:", r)
    r = r.strip()
    if r.startswith("•"):
        r = r.lstrip("• ")
    r = r.split("\n•")[0]
    return re.sub(r"\n{2,}", "\n", r).strip()


def _filtr(filtr: str) -> tuple[str, str]:
    """„typ=zamkniete" → ("typ", "zamkniete"); nieznany routing albo wartość → ValueError."""
    from .router import EPOKI
    dozwolone = {"typ": TYPY, "epoka": EPOKI}
    routing, _, wartosc = filtr.partition("=")
    if wartosc not in dozwolone.get(routing, ()):
        raise ValueError(f"nieznany filtr {filtr!r}; dozwolone: " + "; ".join(f"{r}=<{'|'.join(w)}>" for r, w in dozwolone.items()))
    return routing, wartosc


def _rozklad(licznik: Counter, kolejnosc) -> str:
    n = sum(licznik.values()) or 1
    return ", ".join(f"{k} {licznik[k]} ({licznik[k] / n:.0%})" for k in [*kolejnosc, *sorted(set(licznik) - set(kolejnosc))])


def dane(udzial_kontekstu: float = 0.7, seed: int = 11, filtr: str | None = None, wiki=WIKI_BM25,
         wejscie: Path = SFT / "zadania.jsonl", katalog: Path = SFT, prompt: str = "h0") -> Path:
    """Dane SFT → katalog/train.jsonl, a z filtrem „typ=…"/„epoka=…" → katalog/train-typ-….jsonl (eksperci H1/H2).

    wiki: WIKI_BM25 = wczytaj retrieval.Wikipedia() (domyślnie, jak dotąd); None = bez kontekstu; inny obiekt z .szukaj()
    użyty wprost. Losowanie kontekstu idzie po wszystkich przykładach także przy filtrze, więc ekspert dostaje dokładnie
    te przykłady z train.jsonl (z tym samym kontekstem albo bez), które router skieruje do niego na egzaminie.
    prompt="goly": format gołego modelu (SYS_GOLY + treść zadania, bez kontekstu) → katalog/train-goly.jsonl; tak
    odpowiada zestaw na egzaminie, gdy najlepszy jest goły prompt (Qwen3.5-4B, etap Q 26.09).
    """
    from . import devset, harness, router
    routing, wartosc = _filtr(filtr) if filtr is not None else (None, None)
    ewal = [devset.tresc_dla_modelu(z) for z in devset.wczytaj(
        ["2023-maj", "2023-czerwiec", "2024-maj", "2024-czerwiec", "2025-maj", "2025-czerwiec", "2026-maj", "2026-czerwiec", "synt"])]
    if wiki == WIKI_BM25:
        from .retrieval import Wikipedia
        wiki = Wikipedia()
    rng = random.Random(seed)
    zrodla = []
    for z in (json.loads(l) for l in Path(wejscie).read_text(encoding="utf-8").splitlines() if l.strip()):
        z["typ"] = "zamkniete" if z.get("rodzaj") == "zamkniete" else "otwarte"
        zrodla.append((z, z["odpowiedz_wzorowa"]))
    for z in devset.pula_treningowa():
        zrodla.append((z, _oczysc_cke(z["rozwiazanie"])))
    zakazane = _zakazane(ewal, treningowe=[devset.tresc_dla_modelu(z) for z, _ in zrodla])
    out, odrz, po_wycieku = [], 0, 0
    rozklad = {"typ": Counter(), "epoka": Counter()}
    for z, odp in zrodla:
        if not odp.strip() or len(_ngramy(devset.tresc_dla_modelu(z)) & zakazane) >= MIN_WSPOLNYCH:
            odrz += 1
            continue
        po_wycieku += 1
        z_kontekstem = rng.random() < udzial_kontekstu  # losowane przed filtrem: te same decyzje co w train.jsonl
        klucze = {r: router.klucz_eksperta(z, r) for r in rozklad}
        if routing and klucze[routing] != wartosc:
            continue
        if prompt == "goly":
            pr = [{"role": "system", "content": harness.SYS_GOLY}, {"role": "user", "content": devset.tresc_dla_modelu(z)}]
        else:
            pr = harness.prompt_treningowy(z, harness._kontekst(wiki, z, 4)[0] if z_kontekstem else "")
        out.append({"prompt": pr, "completion": [{"role": "assistant", "content": odp}]})
        for r, k in klucze.items():
            rozklad[r][k] += 1
    rng.shuffle(out)
    katalog = Path(katalog)
    p = katalog / (f"train-{routing}-{wartosc}.jsonl" if routing else ("train-goly.jsonl" if prompt == "goly" else "train.jsonl"))
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    naglowek = f"dane SFT {filtr}: {len(out)} z {po_wycieku}" if routing else f"dane SFT: {len(out)}"
    print(f"{naglowek} przykładów (odrzucone: {odrz}, wyciek lub brak odpowiedzi) → {p}")
    print(f"  typ: {_rozklad(rozklad['typ'], TYPY)}")
    print(f"  epoka: {_rozklad(rozklad['epoka'], router.EPOKI)}")
    gen = katalog / "train.jsonl"
    if routing and gen.exists():
        n_gen = sum(1 for l in gen.read_text(encoding="utf-8").splitlines() if l.strip())
        if n_gen != po_wycieku:
            print(f"UWAGA: {gen} ma {n_gen} przykładów, a to wejście daje {po_wycieku} po filtrze wycieku: ekspert i "
                  f"generalista nie mają łącznie tych samych danych (D19); zbuduj oba pliki z tego samego {Path(wejscie).name}")
    return p


def konfiguracja_sft(nazwa: str, epoki: float, lr: float, batch: int, max_len: int, max_kroki: int = -1):
    """Parametry SFT (TRL 1.14 / transformers 5: rozgrzewka przez `warmup_steps` jako ułamek kroków, bez `warmup_ratio`).
    max_kroki > 0: test dymny (tyle kroków optymalizatora, potem koniec)."""
    from trl import SFTConfig
    return SFTConfig(output_dir=str(TREN / nazwa / "ckpt"), num_train_epochs=epoki, learning_rate=lr, max_steps=max_kroki,
                     per_device_train_batch_size=batch, gradient_accumulation_steps=max(1, 16 // batch),
                     lr_scheduler_type="cosine", warmup_steps=0.05, bf16=True, max_length=max_len,
                     gradient_checkpointing=True, logging_steps=10, eval_strategy="epoch", save_strategy="no",
                     report_to="none", completion_only_loss=True)


def konfiguracja_lora(r: int):
    """LoRA na warstwach liniowych modelu językowego; bez wieży wizji (Qwen3.5: oryginalny mmproj pozostaje zgodny) i bez
    warstw uwagi liniowej `linear_attn` (llama.cpp convert_lora_to_gguf nie umie przestawić ich głowic w macierzach
    LoRA: NotImplementedError w _reorder_v_heads, 26.09). Adapter zapisywany osobno i scalany z wagami."""
    from peft import LoraConfig
    return LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.05, target_modules="all-linear",
                      exclude_modules=r".*(visual|vision|linear_attn).*", task_type="CAUSAL_LM")


def ma_mtp(katalog: Path) -> bool:
    """Czy zapis modelu ma tensory warstwy MTP. Qwen3.5: klasa HF jej nie wczytuje, więc zapis po treningu jej nie ma,
    a config nadal ją deklaruje → eksport GGUF musi dostać `--no-mtp` (inaczej llama-server: brak blk.N.attn_norm).
    MTP służy tylko przyspieszaniu generacji."""
    from safetensors import safe_open
    for f in Path(katalog).glob("*.safetensors"):
        with safe_open(str(f), "pt") as s:
            if any("mtp" in k for k in s.keys()):
                return True
    return False


def trenuj(baza: str, nazwa: str, epoki: float, lr: float, batch: int, max_len: int,
           plik_danych: Path = SFT / "train.jsonl", lora: int = 0, max_kroki: int = -1) -> Path:
    wy = TREN / nazwa / "hf"
    if (wy / "config.json").exists():
        print(f"{nazwa}: już wytrenowany ({wy})"); return wy
    plik_danych = Path(plik_danych)
    if not plik_danych.is_file():  # przed importem torch i ładowaniem modelu: zły --dane widać od razu
        raise FileNotFoundError(f"{nazwa}: brak pliku danych SFT {plik_danych} "
                                "(najpierw: python -m matura.trening dane [--filtr typ=…|epoka=…])")
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer
    tok = AutoTokenizer.from_pretrained(baza)
    try:
        model = AutoModelForCausalLM.from_pretrained(baza, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    except (ValueError, KeyError):  # modele multimodalne (Qwen3.5): klasa image-text-to-text, trenujemy tylko tekst
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(baza, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
        for n, prm in model.named_parameters():
            if "visual" in n or "vision" in n:
                prm.requires_grad = False  # wieża wizji zamrożona → oryginalny mmproj pozostaje zgodny
    ds = load_dataset("json", data_files=str(plik_danych))["train"].train_test_split(test_size=0.05, seed=1)
    cfg = konfiguracja_sft(nazwa, epoki, lr, batch, max_len, max_kroki)
    tr = SFTTrainer(model=model, args=cfg, train_dataset=ds["train"], eval_dataset=ds["test"], processing_class=tok,
                    peft_config=konfiguracja_lora(lora) if lora else None)
    tr.train()
    wy.mkdir(parents=True, exist_ok=True)
    if lora:  # sam adapter (llama.cpp --lora, nie liczy się do rozmiaru, D7) + scalone wagi (GGUF jak po pełnym treningu)
        tr.model.save_pretrained(str(TREN / nazwa / "adapter"))
        tr.model.merge_and_unload().save_pretrained(str(wy))
        zrodlo = Path(baza)
        for f in (zrodlo.glob("*.json") if zrodlo.is_dir() else []):   # preprocessor/chat template bazy (VLM)
            if not (wy / f.name).exists() and not f.name.endswith(".index.json"):   # indeks wag bazy wskazałby jej pliki
                shutil.copy(f, wy / f.name)
    else:
        tr.save_model(str(wy))
    tok.save_pretrained(str(wy))
    (TREN / nazwa / "metryki.json").write_text(json.dumps(tr.state.log_history, indent=1))
    print(f"{nazwa}: zapisany → {wy}")
    return wy


def eksport(nazwa: str, kwanty: list[str], quantize_bin: str) -> None:
    hf, gg = TREN / nazwa / "hf", TREN / nazwa / "gguf"
    gg.mkdir(parents=True, exist_ok=True)
    f16 = gg / f"{nazwa}-F16.gguf"
    if not f16.exists():
        mtp = [] if ma_mtp(hf) or "mtp_num_hidden_layers" not in (hf / "config.json").read_text() else ["--no-mtp"]
        subprocess.run([sys.executable, str(LLAMA_CPP / "convert_hf_to_gguf.py"), str(hf), "--outtype", "f16",
                        "--outfile", str(f16), *mtp], check=True)
    for q in kwanty:
        cel = gg / f"{nazwa}-{q}.gguf"
        if not cel.exists():
            subprocess.run([quantize_bin, str(f16), str(cel), q], check=True)
        print(f"{cel.name}: {cel.stat().st_size/1e6:.0f} MB")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("krok", choices=["dane", "trenuj", "eksport"])
    ap.add_argument("--filtr", help="dane: typ=zamkniete|otwarte|rozstrzygnij albo epoka=<router.EPOKI> (eksperci H1/H2)")
    ap.add_argument("--prompt", default="h0", choices=["h0", "goly"], help="dane: format promptu (goly = SYS_GOLY, bez kontekstu)")
    ap.add_argument("--dane", default=str(SFT / "train.jsonl"), help="trenuj: plik danych SFT (eksperci: train-<filtr>.jsonl)")
    ap.add_argument("--baza"); ap.add_argument("--nazwa")
    ap.add_argument("--epoki", type=float, default=2); ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--lora", type=int, default=0, help="trenuj: rząd LoRA (0 = pełny fine-tuning)")
    ap.add_argument("--max-kroki", type=int, default=-1, help="trenuj: test dymny, tyle kroków")
    ap.add_argument("--kwanty", default="Q4_K_M,Q6_K,Q8_0")
    ap.add_argument("--quantize", default="llama-quantize")
    a = ap.parse_args(argv)
    if a.krok == "dane":
        if a.prompt == "goly":
            dane(filtr=a.filtr, prompt="goly", wiki=None)
        else:
            dane(filtr=a.filtr)
    elif a.krok == "trenuj":
        trenuj(a.baza, a.nazwa, a.epoki, a.lr, a.batch, a.max_len, Path(a.dane), a.lora, a.max_kroki)
    else:
        eksport(a.nazwa, a.kwanty.split(","), a.quantize)


if __name__ == "__main__":
    main()
