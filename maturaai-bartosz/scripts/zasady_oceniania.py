"""Plik wszystkich zasad oceniania matury z historii z OFICJALNEGO źródła (cke.gov.pl) + kalibracja esejów.

Wejście (pobierane z cke.gov.pl, jeśli brak lokalnie):
- Informator o egzaminie maturalnym z historii 2025/2026 (CKE, Warszawa 2024),
- zasady oceniania rozwiązań zadań, sesje majowe 2023–2026 (strony arkuszy CKE formuły 2023).
Wyjście (poza gitem — dokumenty CKE; w paczce tar):
- data/zasady/ZASADY-OCENIANIA-HISTORIA.md   — dosłowne teksty: rozdz. 1 Informatora (opis egzaminu, typy zadań,
  zasady oceniania wszystkich typów zadań, kryteria wypowiedzi argumentacyjnej), uwagi ogólne i kryteria
  wypracowania z zasad oceniania 2026, lista źródeł z SHA256;
- data/zasady/sedzia-krotkie.md, data/zasady/sedzia-esej.md — części pliku podawane sędziemu (bez przykładów zadań);
- data/zasady/kalibracja_esejow.json — 8 przykładowych wypracowań z Informatora z oceną CKE (A, B, komentarz).
Uruchomienie: uv run python scripts/zasady_oceniania.py
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path

import pypdf
import requests

ROOT = Path(__file__).resolve().parent.parent
Z = ROOT / "data" / "zasady"
SRC = Z / "zrodla"
CKE = "https://cke.gov.pl/images/_EGZAMIN_MATURALNY_OD_2023"
ZRODLA = {
    "informator-historia-2025-2026.pdf": f"{CKE}/Informatory/2024/Informator_EM2025_historia_2025_2026.pdf",
    **{f"zasady-{r}-maj.pdf": f"{CKE}/Arkusze_egzaminacyjne/{r}/Historia/MHIP-R0-100-{str(r)[2:]}05-zasady.pdf" for r in (2023, 2024, 2026)},
    "zasady-2025-maj.pdf": f"{CKE}/Arkusze_egzaminacyjne/2025/zasady_oceniania/MHIP-R0-100-2505-zasady.pdf",
}
STOPKA = re.compile(r"^\s*(Informator o egzaminie maturalnym z historii.*|\s*oraz w technikum od roku szkolnego.*|"
                    r"\d+\s*$|Opis egzaminu maturalnego z historii\s+\d+|Przykładowe zadania z rozwiązaniami\s+\d+|"
                    r"\d+\s+Informator.*|Strona \d+ z \d+|Zasady oceniania rozwiązań zadań|Egzamin maturalny z historii – termin.*)\s*$", re.M)


def pobierz() -> dict[str, dict]:
    SRC.mkdir(parents=True, exist_ok=True)
    meta = {}
    for plik, url in ZRODLA.items():
        p = SRC / plik
        if not p.exists():
            r = requests.get(url, timeout=120); r.raise_for_status(); p.write_bytes(r.content)
        meta[plik] = {"url": url, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bajty": p.stat().st_size}
    return meta


def strony(plik: str) -> list[str]:
    return [STOPKA.sub("", p.extract_text() or "") for p in pypdf.PdfReader(str(SRC / plik)).pages]


def czysc(t: str) -> str:
    t = re.sub(r"[ \t]+\n", "\n", t)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def kalibracja(inf: list[str]) -> list[dict]:
    """Przykładowe wypracowania (str. 81–101) z oceną CKE."""
    tekst = "\n".join(inf[80:101])
    wyn = []
    for m_t in re.finditer(r"Przykład (\d+)\.?[ \t]*\n(.*?)(?=\n\s*Wymagania ogólne)", tekst, re.S):
        temat = " ".join(m_t.group(2).split())
        koniec = tekst.find("Przykład ", m_t.end())
        blok = tekst[m_t.end(): koniec if koniec > 0 else len(tekst)]
        for m_r in re.finditer(r"Realizacja (\d)\.?[ \t]*\n(.*?)(?=Realizacja \d\.?[ \t]*\n|\Z)", blok, re.S):
            r = m_r.group(2)
            m_s = re.search(r"ok\.\s*(\d+)\s*słów", r)
            m_a = re.search(r"A\. Narracja historyczna:(.*?)(\d+)\s*pkt", r, re.S)
            m_b = re.search(r"B\. Spójność[^:]*:(.*?)(\d+)\s*pkt", r, re.S)
            if not (m_s and m_a and m_b):
                continue
            wyn.append({"id": f"informator-p{m_t.group(1)}-r{m_r.group(1)}", "temat": temat,
                        "wypracowanie": czysc(r[:m_s.start()]), "slow_cke": int(m_s.group(1)),
                        "pkt_A": int(m_a.group(2)), "pkt_B": int(m_b.group(2)),
                        "pkt": int(m_a.group(2)) + int(m_b.group(2)),
                        "komentarz_cke": czysc(r[m_s.end():])[:3000]})
    return wyn


def main() -> int:
    meta = pobierz()
    inf = strony("informator-historia-2025-2026.pdf")
    rozdz1 = czysc("\n".join(inf[4:14]))            # str. 5–14: opis egzaminu + zasady oceniania
    z26 = strony("zasady-2026-maj.pdf")
    uwagi = czysc(z26[1].split("Zadanie 1.")[0])      # uwagi ogólne na początku zasad
    pelne = "\n".join(z26)
    i = pelne.find("NARRACJA HISTORYCZNA")
    rubryka = czysc(pelne[max(0, pelne.rfind("KRYTERIA", 0, i)) if pelne.rfind("KRYTERIA", 0, i) > 0 else i - 200:])
    kal = kalibracja(inf)
    dzis = dt.date.today().isoformat()
    zrodla_md = "\n".join(f"- `{k}` — {v['url']} (SHA256 {v['sha256'][:16]}…, {v['bajty']} B, pobrano {dzis})" for k, v in meta.items())
    naglowek = (f"# Zasady oceniania — matura z historii, poziom rozszerzony (formuła 2023)\n\n"
                f"**Co to jest:** dosłowne teksty z oficjalnych dokumentów CKE, jedyna podstawa pracy sędziego.\n"
                f"**Źródła (cke.gov.pl):**\n{zrodla_md}\n\n"
                f"Rubryka wypracowania w zasadach oceniania sesji 2023–2026 zgodna w ~99% (pomiar 2026-09-25).\n")
    czesc_ogolna = f"## 1. Informator CKE, rozdział 1: opis egzaminu i zasady oceniania (str. 5–14)\n\n{rozdz1}\n"
    czesc_uwagi = f"## 2. Zasady oceniania 2026 (maj): uwagi ogólne\n\n{uwagi}\n"
    czesc_esej = f"## 3. Zasady oceniania 2026 (maj): kryteria oceniania wypowiedzi argumentacyjnej\n\n{rubryka}\n"
    (Z / "ZASADY-OCENIANIA-HISTORIA.md").write_text(naglowek + "\n" + czesc_ogolna + "\n" + czesc_uwagi + "\n" + czesc_esej
        + f"\n## 4. Przykładowe wypracowania z oceną CKE (Informator str. 81–101)\n\nZob. `kalibracja_esejow.json` ({len(kal)} realizacji).\n",
        encoding="utf-8")
    (Z / "sedzia-krotkie.md").write_text(naglowek + "\n" + czesc_ogolna + "\n" + czesc_uwagi, encoding="utf-8")
    (Z / "sedzia-esej.md").write_text(naglowek + "\n" + czesc_ogolna + "\n" + czesc_esej, encoding="utf-8")
    (Z / "kalibracja_esejow.json").write_text(json.dumps(kal, ensure_ascii=False, indent=1), encoding="utf-8")
    for f in ("ZASADY-OCENIANIA-HISTORIA.md", "sedzia-krotkie.md", "sedzia-esej.md"):
        print(f, len((Z / f).read_text(encoding="utf-8")), "znaków")
    print("kalibracja:", [(k["id"], k["pkt_A"], k["pkt_B"], k["slow_cke"]) for k in kal])
    return 0


if __name__ == "__main__":
    sys.exit(main())
