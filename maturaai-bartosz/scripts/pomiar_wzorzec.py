"""Pomiar sufitu narzędzia do obrazów: h0 + opis wzorcowy ilustracji (Claude z wizją, scripts/opisy_wzorcowe.py).

Co to jest: rejestruje w czasie uruchomienia konfigurację `h0_wzorzec` (= h0 + blok obrazy.blok_opisu(z, "wzorzec"),
ten sam mechanizm co h0_ocr) i uruchamia matura.noc z podanym plikiem konfiguracji.
Po co: bez zmiany harnessu egzaminacyjnego (tylko pomiar na dev; na egzaminie zamknięte API są zakazane).
Co zrobić: `uv run python scripts/pomiar_wzorzec.py --config noc/b0_wzorzec.toml --bez-oceny [--rejestruj wzorzec,vlm_q2b_en]`,
potem matura.pelna_matura.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from matura import harness, noc  # noqa: E402

# --rejestruj w1,w2: konfiguracje h0_<w> dla wariantów opisu z cache (domyślnie tylko wzorzec)
argv, warianty = sys.argv[1:], ["wzorzec"]
if "--rejestruj" in argv:
    i = argv.index("--rejestruj"); warianty = argv[i + 1].split(","); argv = argv[:i] + argv[i + 2:]
for w in warianty:
    harness.KONFIGI_OPIS_ILUSTRACJI[f"h0_{w}"] = w
    noc.KONFIGI_Z_WIKI.add(f"h0_{w}")
sys.argv = ["matura.noc"] + argv
sys.exit(noc.main())
